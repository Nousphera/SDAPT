import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.dinov2.utils.vit_construction import construct_vit_base_backbone, construct_vit_moco_base_backbone
from models.dinov2.utils.param_groups import fuse_params_groups, get_params_groups_with_decay

logger = logging.getLogger("moco")


def _mlp(in_dim, hidden_dim, out_dim, num_layers=3, use_bn=True, last_bn=True):
    """MoCo v3 style MLP (BN + ReLU between layers, optional affine-less BN at the end)."""
    layers = []
    dim = in_dim
    for i in range(num_layers - 1):
        layers.append(nn.Linear(dim, hidden_dim, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim, bias=not last_bn))
    if last_bn:
        # affine=False: no learnable scale/shift on the output, as in MoCo v3
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)


@torch.no_grad()
def concat_all_gather(tensor):
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return tensor
    gathered = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor, async_op=False)
    return torch.cat(gathered, dim=0)


class MoCoBackbone(nn.Module):
    def __init__(self, device, global_args, embed_dim=768):
        super().__init__()
        self.global_args = global_args

        moco_cfg = global_args.get("moco", {})
        self.out_dim = moco_cfg.get("out_dim", 256)
        self.hidden_dim = moco_cfg.get("hidden_dim", 4096)
        self.pred_hidden_dim = moco_cfg.get("pred_hidden_dim", 4096)
        self.proj_layers = moco_cfg.get("proj_nlayers", 3)
        self.pred_layers = moco_cfg.get("pred_nlayers", 2)
        self.temperature = moco_cfg.get("temperature", 0.2)
        self.queue_size = moco_cfg.get("queue_size", 4096)
        self.symmetric = moco_cfg.get("symmetric_loss", True)
        self.use_in_batch_negatives = moco_cfg.get("use_in_batch_negatives", True)
        self.load_dino_weight = global_args.get("dino_weight", False)

        # ---------------------------------------------------------- backbones
        if self.load_dino_weight:
            self.student_backbone = construct_vit_base_backbone(
                device,
                global_args["pretrained_weight_path"],
                global_args["checkpoint_key"],
                patch_size=global_args.get("patch_size", 14),
                embed_dim=embed_dim,
                depth=global_args.get("depth", 12),
                num_heads=global_args.get("num_heads", 12),
                mlp_ratio=global_args.get("mlp_ratio", 4),
                init_values=global_args.get("init_values", 1.0e-05),
            )
            self.teacher_backbone = construct_vit_base_backbone(
                device,
                global_args["pretrained_weight_path"],
                global_args["checkpoint_key"],
                patch_size=global_args.get("patch_size", 14),
                embed_dim=embed_dim,
                depth=global_args.get("depth", 12),
                num_heads=global_args.get("num_heads", 12),
                mlp_ratio=global_args.get("mlp_ratio", 4),
                init_values=global_args.get("init_values", 1.0e-05),
            )
        else:
            self.student_backbone = construct_vit_moco_base_backbone(
                device,
                global_args["pretrained_weight_path"],
                global_args["checkpoint_key"],
                patch_size=global_args.get("patch_size", 16),
                embed_dim=embed_dim,
                depth=global_args.get("depth", 12),
                num_heads=global_args.get("num_heads", 12),
                mlp_ratio=global_args.get("mlp_ratio", 4),
            )
            self.teacher_backbone = construct_vit_moco_base_backbone(
                device,
                global_args["pretrained_weight_path"],
                global_args["checkpoint_key"],
                patch_size=global_args.get("patch_size", 16),
                embed_dim=embed_dim,
                depth=global_args.get("depth", 12),
                num_heads=global_args.get("num_heads", 12),
                mlp_ratio=global_args.get("mlp_ratio", 4),
            )

        # ---------------------------------------------------- projection MLPs
        self.student_proj_head = _mlp(
            embed_dim, self.hidden_dim, self.out_dim, num_layers=self.proj_layers
        )
        self.teacher_proj_head = _mlp(
            embed_dim, self.hidden_dim, self.out_dim, num_layers=self.proj_layers
        )

        # ------------------------------------- prediction MLP (query side ONLY)
        self.student_pred_head = _mlp(
            self.out_dim, self.pred_hidden_dim, self.out_dim,
            num_layers=self.pred_layers, last_bn=False,
        )

        self.student = nn.ModuleDict(
            {
                "backbone": self.student_backbone,
                "proj_head": self.student_proj_head,
                "pred_head": self.student_pred_head,
            }
        )
        self.teacher = nn.ModuleDict(
            {
                "backbone": self.teacher_backbone,
                "proj_head": self.teacher_proj_head,
            }
        )

        self._sync_teacher_modules()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------- queue
        # Pre-allocated at FULL size and filled with random *normalised* vectors.
        # It does not "grow": after queue_size / batch_size steps it has been
        # fully overwritten with real keys.
        self.register_buffer("queue", F.normalize(torch.randn(self.out_dim, self.queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------ init

    @torch.no_grad()
    def _sync_teacher_modules(self):
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_proj_head.load_state_dict(self.student_proj_head.state_dict())

    # --------------------------------------------------------------- forward

    def _cls_token(self, tokens, is_key):
        backbone = self.teacher_backbone if is_key else self.student_backbone
        out = backbone(tokens, is_training=True)
        return out["x_norm_clstoken"]

    def forward_query(self, client_tokens):
        """tokens -> backbone -> proj -> pred -> L2-normalised query embedding."""
        cls = self._cls_token(client_tokens, is_key=False)
        z = self.student_proj_head(cls)
        q = self.student_pred_head(z)
        return F.normalize(q, dim=1)

    @torch.no_grad()
    def forward_key(self, client_tokens):
        """tokens -> momentum backbone -> momentum proj -> L2-normalised key. No grad, ever."""
        cls = self._cls_token(client_tokens, is_key=True)
        k = self.teacher_proj_head(cls)
        return F.normalize(k, dim=1).detach()

    def forward(self, client_tokens, is_key=False):
        return self.forward_key(client_tokens) if is_key else self.forward_query(client_tokens)

    # ------------------------------------------------------------------ loss

    def contrastive_loss(self, q, k):
        """InfoNCE for one direction: q against its positive k, plus queue negatives.

        q: (B, D) query embeddings (grad)
        k: (B, D) positive key embeddings (no grad); k[i] is the positive for q[i]
        """
        # positive logits: (B, 1)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        logits_parts = [l_pos]

        # in-batch negatives (MoCo v3 flavour): every other key in the batch.
        if self.use_in_batch_negatives and k.shape[0] > 1:
            l_batch = q @ k.t()                       # (B, B)
            eye = torch.eye(k.shape[0], dtype=torch.bool, device=q.device)
            l_batch = l_batch.masked_fill(eye, float("-inf"))  # drop the positive
            logits_parts.append(l_batch)

        # queue negatives (MoCo v2 flavour): keys from previous iterations.
        queue = self.queue.clone().detach().to(q.dtype)   # (D, K)
        l_neg = q @ queue                                 # (B, K)
        logits_parts.append(l_neg)

        logits = torch.cat(logits_parts, dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute_loss(self, q1, q2, k1, k2):
        """Symmetric MoCo v3 loss: view1's query vs view2's key, and vice versa."""
        if not self.symmetric:
            return self.contrastive_loss(q1, k2)
        return 0.5 * (self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1))

    # ----------------------------------------------------------------- queue

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """FIFO update. Call AFTER the loss, never before (the positives would
        otherwise appear among their own negatives)."""
        keys = concat_all_gather(keys.detach())
        batch_size = keys.shape[0]
        K = self.queue.shape[1]

        ptr = int(self.queue_ptr)
        idx = (torch.arange(batch_size, device=keys.device) + ptr) % K
        self.queue[:, idx] = keys.t().to(self.queue.dtype)
        self.queue_ptr[0] = (ptr + batch_size) % K
        self.queue_filled[0] = min(int(self.queue_filled) + batch_size, K)

    @property
    def queue_warm(self) -> bool:
        """True once the queue holds only real keys (no random init vectors left)."""
        return int(self.queue_filled) >= self.queue.shape[1]

    # ----------------------------------------------------------- param groups

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=1.0,
            patch_embed_lr_mult=0.2,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        """Only the query (student) branch is optimised."""
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups