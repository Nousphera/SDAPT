import logging
from typing import Any, Dict

from models.dinov2.utils.vit_construction import construct_vit_base_backbone
from models.dinov2.utils.param_groups import fuse_params_groups, get_params_groups_with_decay
import torch
import torch.nn as nn
import torch.distributed as dist

from models.dinov2.layers import DINOHead
from models.dinov2.loss import DINOLoss, iBOTPatchLoss

logger = logging.getLogger("dinov2")


class DinoBackbone(nn.Module):
    def __init__(self, device,global_args, embed_dim=768):
        super().__init__()
        self.global_args = global_args
        self.split_depth = int(global_args.get("split_depth", 0))
        self.model_depth = int(global_args.get("depth", 12))
        if not 0 <= self.split_depth < self.model_depth:
            raise ValueError(f"split_depth must be between 0 and {self.model_depth - 1}")

        self.do_dino = global_args["dino"]["loss_weight"] > 0
        self.do_koleo = global_args["dino"]["koleo_loss_weight"] > 0
        self.do_ibot = global_args["ibot"]["loss_weight"] > 0
        # Split learning uses the shared DINO head for iBOT as well.
        self.ibot_separate_head = False
        self.n_global_crops = 2
        self.centering_mode = global_args["train"]["centering"]
        self.dino_out_dim = global_args["dino"]["head_n_prototypes"]
        self.ibot_out_dim = self.dino_out_dim
        self.teacher_temp = global_args["teacher"]["teacher_temp"]
        
        student_model_dict = dict()
        teacher_model_dict = dict()

        self.student_backbone = construct_vit_base_backbone(
            device,
            global_args['pretrained_weight_path'],
            global_args['checkpoint_key'],
            depth=self.model_depth - self.split_depth,
            block_offset=self.split_depth,
            patch_size=global_args.get("patch_size", 14),
            embed_dim=global_args.get("embed_dim", embed_dim),
            num_heads=global_args.get("num_heads", 12),
            mlp_ratio=global_args.get("mlp_ratio", 4),
            init_values=global_args.get("init_values", 1.0e-05),
        )
        self.teacher_backbone = construct_vit_base_backbone(
            device,
            global_args['pretrained_weight_path'],
            global_args['checkpoint_key'],
            depth=self.model_depth - self.split_depth,
            block_offset=self.split_depth,
            patch_size=global_args.get("patch_size", 14),
            embed_dim=global_args.get("embed_dim", embed_dim),
            num_heads=global_args.get("num_heads", 12),
            mlp_ratio=global_args.get("mlp_ratio", 4),
            init_values=global_args.get("init_values", 1.0e-05),
        )

        student_model_dict["backbone"] =  self.student_backbone
        teacher_model_dict["backbone"] =  self.teacher_backbone

        embed_dim = global_args.get("embed_dim", embed_dim)
        self.student_dino_head = DINOHead(
            in_dim=embed_dim,
            out_dim=self.dino_out_dim,
            hidden_dim=global_args["dino"]["head_hidden_dim"],
            bottleneck_dim=global_args["dino"]["head_bottleneck_dim"],
            nlayers=global_args["dino"]["head_nlayers"],
        )
        self.teacher_dino_head = DINOHead(
            in_dim=embed_dim,
            out_dim=self.dino_out_dim,
            hidden_dim=global_args["dino"]["head_hidden_dim"],
            bottleneck_dim=global_args["dino"]["head_bottleneck_dim"],
            nlayers=global_args["dino"]["head_nlayers"] ,
        )

        student_model_dict["dino_head"] = self.student_dino_head
        teacher_model_dict["dino_head"] = self.teacher_dino_head

        self.student_ibot_head = None
        self.teacher_ibot_head = None

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        self._sync_teacher_modules()

        self.dino_target_builder = DINOLoss(self.dino_out_dim)
        self.ibot_target_builder = iBOTPatchLoss(self.ibot_out_dim) if self.do_ibot else None

    @torch.no_grad()
    def _sync_teacher_modules(self):
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_dino_head.load_state_dict(self.student_dino_head.state_dict())

    def forward_features(self, client_embeddings, is_teacher):
        if is_teacher:
            return self.teacher_backbone(client_embeddings, is_training=True)
        else:
            return self.student_backbone(client_embeddings, is_training=True)

    def forward(self, client_embeddings, is_teacher = False):
        return self.forward_features(client_embeddings, is_teacher=is_teacher)

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=1.0,
            patch_embed_lr_mult=0.2,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups
    


