"""Single-process SDAPT, local DINO/iBOT and split MoCo-v3 across DINO ViTs.

Clients remain separate batches and model instances where required. Tensor casts
simulate the wire format; this module is not a network transport or a privacy
mechanism. The command-line layer owns epochs, schedules and persistence.
"""
from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import SplitEncoder, token_tensor, cast_smashed, concat_smashed, smashed_bytes
from .objectives import dino_loss, ibot_loss, koleo_loss, moco_v3_loss, sinkhorn_knopp


class DINOHead(nn.Module):
    """Three-layer DINO projection MLP with normalized bottleneck/prototypes."""
    def __init__(self, input_dim: int, output_dim: int = 65536,
                 hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, bottleneck_dim))
        self.prototypes = nn.Parameter(torch.empty(output_dim, bottleneck_dim))
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(self.prototypes, std=.02)

    def forward(self, features: Tensor) -> Tensor:
        x = F.normalize(self.mlp(features.to(self.prototypes.dtype)), dim=-1, eps=1e-6)
        return F.linear(x, F.normalize(self.prototypes, dim=-1))


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """Accumulate EMA in float32; promote reduced-precision targets if needed."""
    if not 0 <= momentum <= 1:
        raise ValueError('EMA momentum must lie in [0, 1]')
    if any(p.dtype in (torch.float16, torch.bfloat16) for p in teacher.parameters()):
        teacher.float()
    source, target = student.state_dict(), teacher.state_dict()
    if source.keys() != target.keys():
        raise ValueError('EMA teacher/student state structures differ')
    for name, dest in target.items():
        src = source[name]
        if src.shape != dest.shape:
            raise ValueError(f'EMA shape mismatch: {name}')
        if dest.is_floating_point():
            # A single interpolation avoids two bf16 roundings that can drift
            # even when the online and teacher parameters are identical.
            dest.lerp_(src.to(dest), 1 - momentum)
        else:
            dest.copy_(src)


class StateAccumulator:
    """Streaming, sample-weighted FedAvg; retains one CPU sum, not all clients."""
    def __init__(self):
        self.total_weight = 0.
        self.sums: OrderedDict[str, Tensor] = OrderedDict()
        self.dtypes: dict[str, torch.dtype] = {}

    @torch.no_grad()
    def add(self, state: Mapping[str, Tensor], weight: float) -> None:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('FedAvg client weights must be finite and positive')
        if self.sums and self.sums.keys() != state.keys():
            raise ValueError('FedAvg state keys differ')
        for name, tensor in state.items():
            value = tensor.detach().cpu()
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite FedAvg tensor: {name}')
            if name not in self.sums:
                self.dtypes[name] = value.dtype
                self.sums[name] = value.to(torch.float64 if value.dtype == torch.float64 else torch.float32).mul(weight) if value.is_floating_point() else value.clone()
            elif value.shape != self.sums[name].shape or value.dtype != self.dtypes[name]:
                raise ValueError(f'FedAvg shape/dtype mismatch: {name}')
            elif value.is_floating_point():
                self.sums[name].add_(value, alpha=weight)
            elif name.endswith('num_batches_tracked'):
                self.sums[name] = torch.maximum(self.sums[name], value)
            elif not torch.equal(self.sums[name], value):
                raise ValueError(f'Cannot average unequal nonfloating state: {name}')
        self.total_weight += weight

    @torch.no_grad()
    def average(self) -> OrderedDict[str, Tensor]:
        if self.total_weight <= 0:
            raise ValueError('No positive-weight clients were accumulated')
        return OrderedDict((name, (value / self.total_weight).to(self.dtypes[name])
                            if value.is_floating_point() else value.clone())
                           for name, value in self.sums.items())


def weighted_average_states(states: Sequence[Mapping[str, Tensor]],
                            weights: Sequence[float]) -> OrderedDict[str, Tensor]:
    if len(states) != len(weights) or not states:
        raise ValueError('FedAvg requires equally sized nonempty state/weight lists')
    result = StateAccumulator()
    for state, weight in zip(states, weights):
        result.add(state, weight)
    return result.average()


def block_masks(batch: int, patches: int, *, device: torch.device,
                probability: float = .5, ratio: tuple[float, float] = (.1, .5)) -> Tensor:
    """Random rectangular patch masks for a fraction of global images.

    A bounded fill completes the requested cardinality when rectangles overlap.
    The caller can provide explicit masks to train_step for other protocols.
    """
    side = math.isqrt(patches)
    if side * side != patches or not 0 <= probability <= 1 or not 0 <= ratio[0] <= ratio[1] <= 1:
        raise ValueError('Masking requires a square patch grid and valid probability/ratio')
    masks = torch.zeros(batch, patches, dtype=torch.bool)
    chosen = torch.randperm(batch)[:int(batch * probability)]
    for row in chosen.tolist():
        count = int(patches * (ratio[0] + torch.rand(()).item() * (ratio[1] - ratio[0])))
        grid = masks[row].reshape(side, side)
        for _ in range(20):
            remaining = count - int(grid.sum())
            if remaining <= 0:
                break
            area = max(1, int(remaining * (.5 + .5 * torch.rand(()).item())))
            aspect = math.exp(math.log(.3) + torch.rand(()).item() * math.log(1 / .09))
            height = min(side, max(1, round(math.sqrt(area * aspect))))
            width = min(side, max(1, round(math.sqrt(area / aspect))))
            top = torch.randint(side - height + 1, ()).item()
            left = torch.randint(side - width + 1, ()).item()
            candidate = grid[top:top + height, left:left + width]
            if int((~candidate).sum()) <= remaining:
                candidate.fill_(True)
        remaining = count - int(grid.sum())
        if remaining:
            empty = (~masks[row]).nonzero().flatten()
            masks[row, empty[torch.randperm(len(empty))[:remaining]]] = True
    return masks.to(device)


def _validate_crops(client_crops: Sequence[Sequence[Tensor]], expected_views: int | None = None) -> tuple[int, int]:
    if not client_crops:
        raise ValueError('At least one participating client is required')
    views = len(client_crops[0])
    if views < 2 or (expected_views is not None and views != expected_views):
        raise ValueError('Expected aligned global views followed by optional locals')
    total = 0
    for crops in client_crops:
        if len(crops) != views or not crops or crops[0].ndim != 4 or len(crops[0]) == 0:
            raise ValueError('Each client must supply the same nonempty list of BCHW crops')
        if any(x.ndim != 4 or len(x) != len(crops[0]) for x in crops):
            raise ValueError('Views within each client must contain the same image batch')
        if any(x.shape[1:] != client_crops[0][i].shape[1:] for i, x in enumerate(crops)):
            raise ValueError('Crop shapes must agree across clients')
        if crops[0].shape[1:] != crops[1].shape[1:]:
            raise ValueError('The two global crops must have equal shape')
        total += len(crops[0])
    return views, total


class _AccumulatingTrainer(nn.Module):
    def __init__(self, accumulation_steps: int, max_grad_norm: float | None):
        super().__init__()
        if accumulation_steps < 1 or (max_grad_norm is not None and max_grad_norm <= 0):
            raise ValueError('Invalid accumulation steps or gradient clip norm')
        self.accumulation_steps, self.max_grad_norm = accumulation_steps, max_grad_norm
        self.pending_steps, self.pending_examples = 0, 0
        self.optimizer_steps = 0

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def _backward(self, loss: Tensor, examples: int, optimizer: torch.optim.Optimizer,
                  momentum: float, flush: bool, freeze_head: bool = False) -> bool:
        if not 0 <= momentum <= 1:
            raise ValueError('EMA momentum must lie in [0, 1]')
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            self.pending_steps = self.pending_examples = 0
            raise FloatingPointError('Nonfinite loss; pending gradients discarded')
        if self.pending_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        (loss * examples).backward()
        if freeze_head:
            # None also suppresses optimizer weight decay on the prototype layer.
            self.head.prototypes.grad = None
        self.pending_steps += 1
        self.pending_examples += examples
        if self.pending_steps >= self.accumulation_steps or flush:
            return self.flush(optimizer, ema_momentum=momentum)
        return False

    def flush(self, optimizer: torch.optim.Optimizer, ema_momentum: float = .996) -> bool:
        """Apply a pending partial window with its actual example count."""
        if not self.pending_steps:
            return False
        if not 0 <= ema_momentum <= 1:
            raise ValueError('EMA momentum must lie in [0, 1]')
        params = list(self.trainable_parameters())
        for param in params:
            if param.grad is not None:
                param.grad.div_(self.pending_examples)
                if not torch.isfinite(param.grad).all():
                    optimizer.zero_grad(set_to_none=True)
                    self.pending_steps = self.pending_examples = 0
                    raise FloatingPointError('Nonfinite gradients; optimizer/EMA update skipped')
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self._update_teacher(ema_momentum)
        self.optimizer_steps += 1
        self.pending_steps = self.pending_examples = 0
        return True


class SDAPTTrainer(_AccumulatingTrainer):
    """Frozen clients, server LoRA, globally centered DINO+iBOT+0.1 KoLeo."""
    def __init__(self, student: SplitEncoder, *, head_dim: int = 65536,
                 hidden_dim: int = 2048, bottleneck_dim: int = 256,
                 accumulation_steps: int = 7, transport_dtype: torch.dtype = torch.bfloat16,
                 max_grad_norm: float | None = 3., student_temperature: float = .1,
                 ibot_weight: float = 1., koleo_weight: float = .1):
        super().__init__(accumulation_steps, max_grad_norm)
        if student.method != 'sdapt':
            raise ValueError('SDAPTTrainer requires an SDAPT split encoder')
        self.student = student
        self.head = DINOHead(student.num_features, head_dim, hidden_dim, bottleneck_dim).to(next(student.parameters()).device)
        self.teacher_server = copy.deepcopy(student.server).float().requires_grad_(False).eval()
        self.teacher_head = copy.deepcopy(self.head).requires_grad_(False).eval()
        self.transport_dtype = transport_dtype
        self.student_temperature = student_temperature
        self.ibot_weight, self.koleo_weight = ibot_weight, koleo_weight

    def train(self, mode: bool = True):
        super().train(mode)
        self.student.client.eval()
        self.teacher_server.eval(); self.teacher_head.eval()
        return self

    def _update_teacher(self, momentum: float):
        ema_update(self.teacher_server, self.student.server, momentum)
        ema_update(self.teacher_head, self.head, momentum)

    @torch.no_grad()
    def teacher_state_dict(self) -> OrderedDict[str, Tensor]:
        return self.student.encoder_state_dict(server=self.teacher_server)

    def _loss(self, student_outputs: list[Tensor], teacher_globals: Tensor,
              mask: Tensor, teacher_temperature: float) -> tuple[Tensor, dict[str, float]]:
        batch = len(student_outputs[0])
        prefix = self.student.num_prefix_tokens
        with torch.no_grad():
            cls_targets = sinkhorn_knopp(self.teacher_head(teacher_globals[:, 0]), teacher_temperature).split(batch)
            teacher_masked = teacher_globals[:, prefix:][mask]
            patch_targets = sinkhorn_knopp(self.teacher_head(teacher_masked), teacher_temperature) if len(teacher_masked) else None
        cls_logits = [self.head(tokens[:, 0]) for tokens in student_outputs]
        dino = dino_loss(cls_logits, cls_targets, self.student_temperature)
        global_outputs = torch.cat(student_outputs[:2])
        student_masked = global_outputs[:, prefix:][mask]
        ibot = ibot_loss(self.head(student_masked), patch_targets, mask, self.student_temperature) if patch_targets is not None else global_outputs.sum() * 0
        koleo = sum(koleo_loss(tokens[:, 0]) for tokens in student_outputs[:2])
        loss = dino + self.ibot_weight * ibot + self.koleo_weight * koleo
        return loss, {'dino_loss': float(dino.detach()), 'ibot_loss': float(ibot.detach()),
                      'koleo_loss': float(koleo.detach()), 'loss': float(loss.detach())}

    def train_step(self, client_crops: Sequence[Sequence[Tensor]], *,
                   optimizer: torch.optim.Optimizer, teacher_temperature: float = .07,
                   ema_momentum: float = .996, masks: Tensor | None = None,
                   flush: bool = False, freeze_head: bool = False) -> dict:
        views, examples = _validate_crops(client_crops)
        self.train()
        dtype = next(self.student.parameters()).dtype
        device = next(self.student.parameters()).device
        uplink, paper_uplink = 0, 0
        combined = []
        with torch.no_grad():
            # View first, client second: teacher/student indices stay aligned.
            for view in range(views):
                pieces = []
                for crops in client_crops:
                    wire = cast_smashed(self.student.client(crops[view].to(device)), self.transport_dtype)
                    uplink += smashed_bytes(wire)
                    paper_uplink += smashed_bytes(wire, include_positions=False)
                    pieces.append(cast_smashed(wire, dtype))
                combined.append(concat_smashed(pieces))
            globals_ = concat_smashed(combined[:2])
            teacher_global = self.teacher_server(globals_)
        prefix = self.student.num_prefix_tokens
        global_tensor = token_tensor(globals_)
        mask = block_masks(2 * examples, global_tensor.shape[1] - prefix, device=device) if masks is None else masks.to(device)
        if mask.shape != global_tensor[:, prefix:, 0].shape or mask.dtype != torch.bool:
            raise ValueError('Masks must be crop-major [2*global_batch, patches] booleans')
        student_globals = self.student.server(globals_, mask).split(examples)
        outputs = list(student_globals) + [self.student.server(x) for x in combined[2:]]
        loss, metrics = self._loss(outputs, teacher_global, mask, teacher_temperature)
        stepped = self._backward(loss, examples, optimizer, ema_momentum, flush, freeze_head)
        return {**metrics, 'examples': examples, 'optimizer_step': stepped,
                'uplink_bytes': uplink, 'paper_uplink_bytes': paper_uplink, 'downlink_bytes': 0}


class DINOLocalTrainer(SDAPTTrainer):
    """Full-model DINO local update for the FedAvg baseline.

    Call with exactly one client's views. The federation driver broadcasts and
    sample-averages the complete state_dict (student, EMA teacher, shared head).
    No optimizer moments are federated, and no adaptive server optimizer is used.
    """
    def __init__(self, student: SplitEncoder, *, head_dim: int = 65536,
                 hidden_dim: int = 2048, bottleneck_dim: int = 256,
                 accumulation_steps: int = 7, transport_dtype: torch.dtype = torch.bfloat16,
                 max_grad_norm: float | None = 3., student_temperature: float = .1,
                 ibot_weight: float = 1., koleo_weight: float = .1):
        _AccumulatingTrainer.__init__(self, accumulation_steps, max_grad_norm)
        if student.method != 'fedavg':
            raise ValueError('DINOLocalTrainer requires a fully trainable FedAvg encoder')
        self.student = student
        self.head = DINOHead(student.num_features, head_dim, hidden_dim, bottleneck_dim).to(next(student.parameters()).device)
        self.teacher = copy.deepcopy(student).float().requires_grad_(False).eval()
        self.teacher_head = copy.deepcopy(self.head).requires_grad_(False).eval()
        self.transport_dtype = transport_dtype
        self.student_temperature = student_temperature
        self.ibot_weight, self.koleo_weight = ibot_weight, koleo_weight

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.teacher.eval(); self.teacher_head.eval()
        return self

    def _update_teacher(self, momentum: float):
        ema_update(self.teacher, self.student, momentum)
        ema_update(self.teacher_head, self.head, momentum)

    @torch.no_grad()
    def teacher_state_dict(self) -> OrderedDict[str, Tensor]:
        return self.teacher.encoder_state_dict()

    def train_step(self, client_crops: Sequence[Sequence[Tensor]], *,
                   optimizer: torch.optim.Optimizer, teacher_temperature: float = .07,
                   ema_momentum: float = .996, masks: Tensor | None = None,
                   flush: bool = False, freeze_head: bool = False) -> dict:
        _, examples = _validate_crops(client_crops)
        if len(client_crops) != 1:
            raise ValueError('FedAvg local DINO step accepts one client at a time')
        self.train()
        device = next(self.student.parameters()).device
        crops = [x.to(device) for x in client_crops[0]]
        with torch.no_grad():
            teacher_globals = self.teacher.forward_tokens(torch.cat(crops[:2]))
        mask = block_masks(2 * examples, teacher_globals.shape[1] - self.student.num_prefix_tokens, device=device) if masks is None else masks.to(device)
        globals_ = self.student.forward_tokens(torch.cat(crops[:2]), mask).split(examples)
        outputs = list(globals_) + [self.student.forward_tokens(x) for x in crops[2:]]
        loss, metrics = self._loss(outputs, teacher_globals, mask, teacher_temperature)
        stepped = self._backward(loss, examples, optimizer, ema_momentum, flush, freeze_head)
        return {**metrics, 'examples': examples, 'optimizer_step': stepped,
                'uplink_bytes': 0, 'paper_uplink_bytes': 0, 'downlink_bytes': 0}


def _moco_mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
    parts: list[nn.Module] = []
    for index in range(layers):
        source = input_dim if index == 0 else hidden_dim
        destination = output_dim if index == layers - 1 else hidden_dim
        parts.append(nn.Linear(source, destination, bias=False))
        parts.append(nn.BatchNorm1d(destination, affine=index != layers - 1))
        if index != layers - 1:
            parts.append(nn.ReLU())
    return nn.Sequential(*parts)


class MoCoSFLTrainer(_AccumulatingTrainer):
    """MoCo-v3 with distinct trainable client models and a shared ViT server.

    Both online and EMA client features are transmitted. ``paper_uplink_bytes``
    separately reports the paper's two-view online-activation proxy; the actual
    ``uplink_bytes`` includes both branches. Gradients return only to online
    clients. Optional synchronize_clients is explicit, never automatic.
    """
    def __init__(self, student: SplitEncoder, *, num_clients: int = 10,
                 project_dim: int = 256, hidden_dim: int = 4096,
                 accumulation_steps: int = 7, transport_dtype: torch.dtype = torch.bfloat16,
                 max_grad_norm: float | None = 3., temperature: float = .2):
        super().__init__(accumulation_steps, max_grad_norm)
        if student.method != 'mocosfl' or num_clients < 1:
            raise ValueError('MoCoSFL requires a trainable split encoder and positive client count')
        self.student = student
        self.extra_clients = nn.ModuleList(copy.deepcopy(student.client) for _ in range(num_clients - 1))
        self.teacher = copy.deepcopy(student).float().requires_grad_(False).eval()
        self.extra_teacher_clients = nn.ModuleList(copy.deepcopy(self.teacher.client) for _ in range(num_clients - 1))
        device = next(student.parameters()).device
        self.projector = _moco_mlp(student.num_features, hidden_dim, project_dim, 3).to(device)
        self.predictor = _moco_mlp(project_dim, hidden_dim, project_dim, 2).to(device)
        self.teacher_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.transport_dtype, self.temperature = transport_dtype, temperature

    @property
    def client_models(self):
        return [self.student.client, *self.extra_clients]

    @property
    def teacher_client_models(self):
        return [self.teacher.client, *self.extra_teacher_clients]

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.extra_teacher_clients.eval()
        # MoCo-v3 target projector uses batch statistics, even without gradients.
        self.teacher_projector.train(mode)
        return self

    def _update_teacher(self, momentum: float):
        ema_update(self.teacher, self.student, momentum)
        for target, source in zip(self.extra_teacher_clients, self.extra_clients):
            ema_update(target, source, momentum)
        ema_update(self.teacher_projector, self.projector, momentum)

    @torch.no_grad()
    def synchronize_clients(self, weights: Sequence[float]) -> None:
        """Optional weighted client averaging at a caller-selected boundary."""
        if self.pending_steps:
            raise RuntimeError('Flush accumulated gradients before synchronizing clients')
        for clients in (self.client_models, self.teacher_client_models):
            average = weighted_average_states([m.state_dict() for m in clients], weights)
            for client in clients:
                client.load_state_dict(average, strict=True)

    @torch.no_grad()
    def teacher_state_dict(self, client_id: int = 0) -> OrderedDict[str, Tensor]:
        if not 0 <= client_id < len(self.client_models):
            raise ValueError('Unknown client id')
        return self.teacher.encoder_state_dict(client=self.teacher_client_models[client_id])

    def train_step(self, client_crops: Sequence[Sequence[Tensor]], *,
                   optimizer: torch.optim.Optimizer, teacher_temperature: float = .07,
                   ema_momentum: float = .996, masks: Tensor | None = None,
                   flush: bool = False, client_ids: Sequence[int] | None = None) -> dict:
        _, examples = _validate_crops(client_crops, expected_views=2)
        if examples < 2:
            raise ValueError('MoCo-v3 needs at least two samples for negatives and BatchNorm')
        if masks is not None:
            raise ValueError('MoCo-v3 does not use patch masks')
        ids = list(range(len(client_crops))) if client_ids is None else list(client_ids)
        if len(ids) != len(client_crops) or len(set(ids)) != len(ids) or any(i < 0 or i >= len(self.client_models) for i in ids):
            raise ValueError('Participating client ids must be unique and valid')
        self.train()
        device = next(self.student.parameters()).device
        dtype = next(self.student.parameters()).dtype
        online, target = [], []
        online_bytes, target_bytes, paper_uplink, downlink = 0, 0, 0, 0
        for view in range(2):
            queries, keys = [], []
            for client_id, crops in zip(ids, client_crops):
                images = crops[view].to(device)
                wire = cast_smashed(self.client_models[client_id](images), self.transport_dtype)
                online_bytes += smashed_bytes(wire)
                paper_uplink += smashed_bytes(wire, include_positions=False)
                downlink += smashed_bytes(wire, include_positions=False)
                queries.append(cast_smashed(wire, dtype))
                with torch.no_grad():
                    key_wire = cast_smashed(self.teacher_client_models[client_id](images), self.transport_dtype)
                    target_bytes += smashed_bytes(key_wire)
                    keys.append(cast_smashed(key_wire, dtype))
            features = self.student.server(concat_smashed(queries))[:, 0]
            online.append(self.predictor(self.projector(features.to(next(self.projector.parameters()).dtype))))
            with torch.no_grad():
                features = self.teacher.server(concat_smashed(keys))[:, 0]
                target.append(self.teacher_projector(features.to(next(self.teacher_projector.parameters()).dtype)))
        loss = moco_v3_loss(online[0], online[1], target[0], target[1], self.temperature)
        stepped = self._backward(loss, examples, optimizer, ema_momentum, flush)
        return {'loss': float(loss.detach()), 'infonce_loss': float(loss.detach()),
                'examples': examples, 'optimizer_step': stepped,
                'uplink_bytes': online_bytes + target_bytes, 'paper_uplink_bytes': paper_uplink,
                'downlink_bytes': downlink}
