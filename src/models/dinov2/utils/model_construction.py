from functools import partial

from torch import Block, distributed, nn
import torch

from models.dinov2.dino_backbone import DinoBackbone
from models.dinov2.dino_head import DinoHead 
from models.dinov2.layers.attention import MemEffAttention
from models.dinov2.layers.attention import MemEffAttention
from models.dinov2.lp_tail import AllClassifiers, LinearClassifier, create_linear_input
from models.dinov2.utils.misc import activate_lora_for_model, configure_trainable_params_for_model, load_pretrained_weights
from models.dinov2.vit_backbone import ViTBackbone
from models.dinov2.vit_head import ViTHead

def construct_dino_backbone(
    device, pretrained_weight_path, checkpoint_key='model', global_args=None, apply_trainable_policy=True, **kwargs
 
) -> DinoBackbone:
    backbone_model = DinoBackbone(device, global_args=global_args)
    # load_vit_backbone_pretrained(backbone_model, global_args['pretrained_path'], map_location=map_location)

    activate_lora_for_model(
        backbone_model.student["backbone"],
        global_args["lora"]["unfreeze_blocks"],
        global_args["lora"]["layers"],
        global_args["lora"]["rank"],
        global_args["lora"]["attn_key"],    
        global_args["lora"]["attn_proj"],
    )
    activate_lora_for_model(
        backbone_model.teacher["backbone"],
        global_args["lora"]["unfreeze_blocks"],
        global_args["lora"]["layers"],
        global_args["lora"]["rank"],
        global_args["lora"]["attn_key"],
        global_args["lora"]["attn_proj"],
    )

    for name, param in backbone_model.student["backbone"].named_parameters():
        if not (
            "lora" in name
            or "register_token" in name
            or (global_args["lora"]["unfreeze_norm"] and "norm" in name)
            or (global_args["lora"]["unfreeze_cls_token"] and "cls_token" in name)
            or (getattr(global_args["lora"], "unfreeze_mask_token", False) and "mask_token" in name)
            or (
                global_args["lora"]["unfreeze_blocks"] is not None
                and any([f"blocks.{idx}." in name for idx in global_args["lora"]["unfreeze_blocks"]])
            )
        ):
            param.requires_grad_(False)

    # there is no backpropagation through the teacher, so no need for gradients (reset this after activating LoRA)
    for p in backbone_model.teacher.parameters():
        p.requires_grad = False
    backbone_model.to(device)
    return backbone_model

def construct_dino_head(device, pretrained_weight_path, checkpoint_key='model', **kwargs) -> DinoHead:
    client_model = DinoHead().to(device)

    for param in client_model.parameters():
        param.requires_grad_(False)

    # if pretrained_path:
    #     load_dino_client_pretrained_initial_layers(client_model, pretrained_path)
    load_pretrained_weights(client_model, pretrained_weight_path, checkpoint_key=checkpoint_key)

    return client_model





