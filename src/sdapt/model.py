"""Strict DINO ViT construction, register-aware splits and rotary positions."""
from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass

import timm
import torch
from safetensors.torch import load_file
from timm.models.vision_transformer import VisionTransformer
from timm.models.eva import Eva
from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODEL_NAME = 'vit_base_patch14_dinov2.lvd142m'


@dataclass(frozen=True)
class SmashedTokens:
    """DINOv3 activations plus the matching rotary-position embedding.

    Keeping positions explicit avoids mutable, last-crop server state when
    differently sized global and local views are interleaved.
    """
    tokens: Tensor
    rope: Tensor | None


def token_tensor(features: Tensor | SmashedTokens) -> Tensor:
    return features.tokens if isinstance(features, SmashedTokens) else features


def cast_smashed(features: Tensor | SmashedTokens, dtype: torch.dtype) -> Tensor | SmashedTokens:
    if isinstance(features, SmashedTokens):
        return SmashedTokens(features.tokens.to(dtype), features.rope)
    return features.to(dtype)


def concat_smashed(features: list[Tensor | SmashedTokens]) -> Tensor | SmashedTokens:
    if not features:
        raise ValueError('Cannot concatenate an empty smashed batch')
    if isinstance(features[0], SmashedTokens):
        first = features[0]
        for other in features[1:]:
            if not isinstance(other, SmashedTokens) or (first.rope is None) != (other.rope is None):
                raise ValueError('Inconsistent rotary metadata across client batches')
            if first.rope is not None and not torch.equal(first.rope, other.rope):
                raise ValueError('Rotary positions differ across concatenated views')
        return SmashedTokens(torch.cat([item.tokens for item in features]), first.rope)
    if any(isinstance(item, SmashedTokens) for item in features):
        raise ValueError('Cannot mix rotary and absolute-position batches')
    return torch.cat(features)


def smashed_bytes(features: Tensor | SmashedTokens, include_positions: bool = True) -> int:
    tokens = token_tensor(features)
    count = tokens.numel() * tokens.element_size()
    if include_positions and isinstance(features, SmashedTokens) and features.rope is not None:
        count += features.rope.numel() * features.rope.element_size()
    return count


def _blocks(blocks, tokens: Tensor, rope: Tensor | None, is_eva: bool, checkpointing: bool) -> Tensor:
    for block in blocks:
        projection = block.attn.qkv
        if isinstance(projection, QVLoRALinear):
            projection = projection.base
        # Frozen blocks can use bf16 storage while fully trainable blocks keep
        # their original fp32 weights and receive fp32 activations.
        tokens = tokens.to(projection.weight.dtype)
        kwargs = {'rope': rope} if is_eva else {}
        if checkpointing and torch.is_grad_enabled():
            tokens = checkpoint(block, tokens, use_reentrant=False, **kwargs)
        else:
            tokens = block(tokens, **kwargs)
    return tokens


class QVLoRALinear(nn.Module):
    """Low-rank updates to Q and V of timm's fused QKV; K stays unchanged."""
    def __init__(self, base: nn.Linear, rank: int = 64, alpha: float = 32):
        super().__init__()
        if rank < 1 or alpha <= 0 or base.out_features != 3 * base.in_features:
            raise ValueError('Expected fused QKV projection and positive LoRA rank/alpha')
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        dim = base.in_features
        self.q_a = nn.Parameter(base.weight.new_empty(rank, dim))
        self.q_b = nn.Parameter(base.weight.new_zeros(dim, rank))
        self.v_a = nn.Parameter(base.weight.new_empty(rank, dim))
        self.v_b = nn.Parameter(base.weight.new_zeros(dim, rank))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base(x.to(self.base.weight.dtype))
        q, k, v = base_output.chunk(3, dim=-1)
        scale = self.alpha / self.rank
        adapter_input = x.to(self.q_a.dtype)
        q = q + F.linear(F.linear(adapter_input, self.q_a), self.q_b) * scale
        v = v + F.linear(F.linear(adapter_input, self.v_a), self.v_b) * scale
        return torch.cat((q, k, v), dim=-1).to(base_output.dtype)

    @property
    def weight(self) -> Tensor:
        # Eva's optional separated Q/V bias path calls F.linear(qkv.weight).
        dim = self.base.in_features
        delta = torch.cat((self.q_b @ self.q_a, self.base.weight.new_zeros(dim, dim),
                           self.v_b @ self.v_a))
        return (self.base.weight + delta * (self.alpha / self.rank)).to(self.base.weight.dtype)

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        weight = self.base.weight.detach().to(self.q_a.dtype).clone()
        dim = self.base.in_features
        weight[:dim].add_(self.q_b @ self.q_a, alpha=self.alpha / self.rank)
        weight[2 * dim:].add_(self.v_b @ self.v_a, alpha=self.alpha / self.rank)
        return weight


class ViTClient(nn.Module):
    """Patch embedding, positional encoding and the configured client blocks."""
    def __init__(self, backbone: VisionTransformer | Eva, split_depth: int = 1):
        super().__init__()
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.reg_token = backbone.reg_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop, self.patch_drop, self.norm_pre = backbone.pos_drop, backbone.patch_drop, backbone.norm_pre
        self.blocks = nn.Sequential(*list(backbone.blocks)[:split_depth])
        self.dynamic_img_size = backbone.dynamic_img_size
        self.no_embed_class, self.num_prefix_tokens = backbone.no_embed_class, backbone.num_prefix_tokens
        self.num_features = backbone.num_features
        self.is_eva = isinstance(backbone, Eva)
        self.grad_checkpointing = getattr(backbone, 'grad_checkpointing', False)
        if self.is_eva:
            self.rope = backbone.rope
            self.rope_mixed = getattr(backbone, 'rope_mixed', False)
            if self.rope_mixed:
                raise ValueError('Depth-dependent mixed rotary embeddings are not a DINO architecture')

    def forward(self, images: Tensor, mask: Tensor | None = None,
                mask_token: Tensor | None = None) -> Tensor | SmashedTokens:
        tokens = self.patch_embed(images.to(self.patch_embed.proj.weight.dtype))
        if mask is not None:
            if mask_token is None or mask.shape != (tokens.shape[0], tokens.shape[1] * tokens.shape[2]):
                raise ValueError('Input mask must match the patch embedding grid')
            tokens = torch.where(mask.reshape(*tokens.shape[:3], 1), mask_token.to(tokens), tokens)
        if self.is_eva:
            tokens, rope = Eva._pos_embed(self, tokens)
            tokens = _blocks(self.blocks, self.norm_pre(tokens), rope, True, self.grad_checkpointing)
            return SmashedTokens(tokens, rope)
        tokens = VisionTransformer._pos_embed(self, tokens)
        return _blocks(self.blocks, self.norm_pre(self.patch_drop(tokens)), None, False, self.grad_checkpointing)


class ViTServer(nn.Module):
    """Remaining transformer blocks; SDAPT masking happens at this boundary."""
    def __init__(self, backbone: VisionTransformer | Eva, split_depth: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(*list(backbone.blocks)[split_depth:])
        self.norm = backbone.norm
        self.mask_token = nn.Parameter(backbone.cls_token.new_zeros(1, 1, backbone.num_features))
        self.num_features = backbone.num_features
        self.num_prefix_tokens = backbone.num_prefix_tokens
        self.is_eva = isinstance(backbone, Eva)
        self.grad_checkpointing = getattr(backbone, 'grad_checkpointing', False)

    def forward(self, tokens: Tensor | SmashedTokens, mask: Tensor | None = None) -> Tensor:
        rope = tokens.rope if isinstance(tokens, SmashedTokens) else None
        if self.is_eva and not isinstance(tokens, SmashedTokens):
            raise ValueError('DINOv3 server requires rotary metadata from its client')
        tokens = token_tensor(tokens)
        prefix = self.num_prefix_tokens
        if mask is not None:
            if mask.dtype != torch.bool or mask.shape != tokens[:, prefix:, 0].shape:
                raise ValueError('Server mask must match [batch, smashed patches]')
            patches = torch.where(mask.unsqueeze(-1), self.mask_token.to(tokens), tokens[:, prefix:])
            tokens = torch.cat((tokens[:, :prefix], patches), dim=1)
        return self.norm(_blocks(self.blocks, tokens, rope, self.is_eva, self.grad_checkpointing))


class SplitEncoder(nn.Module):
    """A partition of an already initialized timm ViT, consuming its modules.

    Production construction goes through build_split_encoder, which always
    strictly loads pretrained weights. Direct construction supports inspecting
    an existing compatible ViT and small independent test fixtures.
    """
    def __init__(self, backbone: VisionTransformer | Eva, method: str = 'sdapt',
                 unfreeze_last: int = 1, lora_rank: int = 64, lora_alpha: float = 32,
                 split_depth: int = 1, gradient_checkpointing: bool = False):
        super().__init__()
        if method not in {'sdapt', 'fedavg', 'mocosfl'}:
            raise ValueError(f'Unknown adaptation method: {method}')
        if backbone.cls_token is None or not backbone.dynamic_img_size:
            raise ValueError('Split encoder needs dynamic ViT images and a CLS token')
        if not 0 <= split_depth < len(backbone.blocks):
            raise ValueError('split_depth must leave at least one server block')
        if not 1 <= unfreeze_last <= len(backbone.blocks) - split_depth:
            raise ValueError('unfreeze_last must fit the server block count')
        self.method, self.split_depth, self.num_features = method, split_depth, backbone.num_features
        self.num_prefix_tokens = backbone.num_prefix_tokens
        backbone.set_grad_checkpointing(gradient_checkpointing)
        self.client = ViTClient(backbone, split_depth)
        self.server = ViTServer(backbone, split_depth)
        if method == 'sdapt':
            self.client.requires_grad_(False)
            self.server.requires_grad_(False)
            for block in list(self.server.blocks)[:-unfreeze_last]:
                block.attn.qkv = QVLoRALinear(block.attn.qkv, lora_rank, lora_alpha)
            for block in list(self.server.blocks)[-unfreeze_last:]:
                block.requires_grad_(True)
            self.server.mask_token.requires_grad_(True)
            for module in self.server.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
        elif method == 'mocosfl':
            self.server.mask_token.requires_grad_(False)

    def forward_tokens(self, images: Tensor, mask: Tensor | None = None) -> Tensor:
        if self.method == 'fedavg':
            return self.server(self.client(images, mask, self.server.mask_token))
        return self.server(self.client(images), mask)

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_tokens(images)[:, 0]

    def encoder_state_dict(self, *, server: ViTServer | None = None,
                           client: ViTClient | None = None) -> OrderedDict[str, Tensor]:
        """Export final timm weights, merging adapters without mutation.

        CPU tensors share storage until serialization; call after training ends.
        This avoids another full-model allocation for huge and 7B encoders.
        """
        client = self.client if client is None else client
        server = self.server if server is None else server
        state = OrderedDict((key, value.detach().cpu()) for key, value in client.state_dict().items())
        server_state = OrderedDict((key, value.detach().cpu()) for key, value in server.state_dict().items()
                                   if key != 'mask_token')
        for name, module in server.named_modules():
            if isinstance(module, QVLoRALinear):
                for suffix in ('base.weight', 'base.bias', 'q_a', 'q_b', 'v_a', 'v_b'):
                    server_state.pop(f'{name}.{suffix}', None)
                server_state[f'{name}.weight'] = module.merged_weight().cpu()
                if module.base.bias is not None:
                    server_state[f'{name}.bias'] = module.base.bias.detach().cpu()
        for name, value in server_state.items():
            if name.startswith('blocks.'):
                _, index, suffix = name.split('.', 2)
                name = f'blocks.{int(index) + self.split_depth}.{suffix}'
            state[name] = value
        return state


class DINOEncoder(nn.Module):
    """Frozen downstream encoder returning final LayerNorm CLS features."""
    def __init__(self, backbone: VisionTransformer | Eva):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.num_features = backbone.num_features

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_tokens(images)[:, 0]

    def forward_tokens(self, images: Tensor) -> Tensor:
        return self.backbone.forward_features(images.to(self.backbone.patch_embed.proj.weight.dtype))


def _loaded_backbone(pretrained_path: str | Path | None = None,
                     state_dict: Mapping[str, Tensor] | None = None,
                     model: str = 'dinov2-vitb14', parameter_dtype: torch.dtype | str | None = None):
    if (pretrained_path is None) == (state_dict is None):
        raise ValueError('Provide exactly one pretrained safetensors path or encoder state_dict')
    if pretrained_path is not None:
        path = Path(pretrained_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix != '.safetensors':
            raise ValueError('Pretrained backbone must be a safetensors file')
        state_dict = load_file(str(path), device='cpu')
    from .registry import get_model_spec
    parameter_dtype = _parameter_dtype(parameter_dtype)
    # safetensors uses memory-mapped storage. Meta construction avoids allocating
    # and random-initializing another complete 7B model before loading weights.
    with torch.device('meta'):
        backbone = timm.create_model(get_model_spec(model).timm_name, pretrained=False,
                                    num_classes=0, dynamic_img_size=True)
    state_dict = {key: value.to(parameter_dtype) if value.is_floating_point() else value
                  for key, value in state_dict.items()}
    backbone.load_state_dict(state_dict, strict=True, assign=True)
    # Rotary periods and Eva's zero key bias are absent from checkpoints.
    for module in backbone.modules():
        if isinstance(module, RotaryEmbeddingDinoV3):
            module.periods = module._compute_periods(device='cpu', dtype=torch.float32)
            module.pos_embed_cached = None
            module.feat_shape = None
        for name, value in list(module._buffers.items()):
            if value is not None and name == 'k_bias':
                module._buffers[name] = torch.zeros(value.shape, dtype=parameter_dtype)
                continue
            if value is not None and value.is_meta:
                if name != 'k_bias':
                    raise RuntimeError(f'Uninitialized nonpersistent model buffer: {name}')
                module._buffers[name] = torch.zeros(value.shape, dtype=parameter_dtype)
    return backbone


def _parameter_dtype(parameter_dtype: torch.dtype | str | None) -> torch.dtype:
    if isinstance(parameter_dtype, str):
        if parameter_dtype not in {'float32', 'bfloat16'}:
            raise ValueError('parameter_dtype must be float32 or bfloat16')
        parameter_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[parameter_dtype]
    if parameter_dtype is None:
        parameter_dtype = torch.float32
    if parameter_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('parameter_dtype must be float32 or bfloat16')
    return parameter_dtype


def build_split_encoder(pretrained_path: str | Path, method: str = 'sdapt',
                        unfreeze_last: int = 1, lora_rank: int = 64,
                        lora_alpha: float = 32, split_depth: int = 1,
                        model: str = 'dinov2-vitb14', parameter_dtype: torch.dtype | str | None = None,
                        gradient_checkpointing: bool = False) -> SplitEncoder:
    frozen_dtype = _parameter_dtype(parameter_dtype)
    # Determine trainability before any storage reduction: pretrained trainable
    # weights must not be quantized and then merely promoted back to float32.
    encoder = SplitEncoder(_loaded_backbone(pretrained_path, model=model, parameter_dtype=torch.float32),
                           method, unfreeze_last, lora_rank, lora_alpha, split_depth, gradient_checkpointing)
    for parameter in encoder.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(frozen_dtype)
    for module in encoder.modules():
        if getattr(module, 'q_bias', None) is not None and getattr(module, 'k_bias', None) is not None:
            module.k_bias = module.k_bias.to(module.q_bias.dtype)
    return encoder


def build_encoder(pretrained_path: str | Path | None = None,
                  state_dict: Mapping[str, Tensor] | None = None,
                  model: str = 'dinov2-vitb14', parameter_dtype: torch.dtype | str | None = None,
                  gradient_checkpointing: bool = False) -> DINOEncoder:
    backbone = _loaded_backbone(pretrained_path, state_dict, model, parameter_dtype)
    backbone.set_grad_checkpointing(gradient_checkpointing)
    return DINOEncoder(backbone)
