
from functools import partial
from venv import logger

from requests.compat import urlparse
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from models.dinov2.dino_backbone import DinoBackbone
from models.dinov2.dino_head import DinoHead
from models.dinov2.utils.misc import activate_lora_for_model, load_moco_vit_backbone, load_pretrained_weights
from models.dinov2.vision_transformer_layers import Block, MemEffAttention
from models.dinov2.vit_backbone import ViTBackbone
from models.dinov2.vit_head import ViTHead


def _make_vit_base_backbone(**kwargs) -> ViTBackbone:
    kwargs.setdefault("patch_size", 14)
    kwargs.setdefault("embed_dim", 768)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 12)
    kwargs.setdefault("mlp_ratio", 4)
    kwargs.setdefault("num_register_tokens", 0)
    return ViTBackbone(
        drop_path_uniform=True,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )

def _make_vit_base_moco_backbone(**kwargs) -> ViTBackbone:
    return ViTBackbone(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_uniform=True,
        init_values=None,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=0,
        **kwargs,
    )

def _apply_lora_trainable_policy(model, lora_args):
    activate_lora_for_model(
        model,
        lora_args["unfreeze_blocks"],
        lora_args["layers"],
        lora_args["rank"],
        lora_args["attn_key"],
        lora_args["attn_proj"],
    )

    for name, param in model.named_parameters():
        if not (
            "lora" in name
            or "register_token" in name
            or (lora_args["unfreeze_norm"] and "norm" in name)
            or (lora_args["unfreeze_cls_token"] and "cls_token" in name)
            or (getattr(lora_args, "unfreeze_mask_token", False) and "mask_token" in name)
            or (
                lora_args["unfreeze_blocks"] is not None
                and any([f"blocks.{idx}." in name for idx in lora_args["unfreeze_blocks"]])
            )
        ):
            param.requires_grad_(False)

def load_pretrained_weightss(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # --- expand patch_embed from 3->4 channels if the model expects more channels
    #     than the checkpoint provides (e.g. HPA 4-channel warm-start from RGB) ---
    pe_key = "patch_embed.proj.weight"
    if pe_key in state_dict:
        ckpt_w = state_dict[pe_key]                       # [embed_dim, C_ckpt, P, P]
        model_w = model.state_dict()[pe_key]              # [embed_dim, C_model, P, P]
        c_ckpt, c_model = ckpt_w.shape[1], model_w.shape[1]
        if c_model > c_ckpt:
            n_extra = c_model - c_ckpt
            extra = ckpt_w.mean(dim=1, keepdim=True).repeat(1, n_extra, 1, 1)
            state_dict[pe_key] = torch.cat([ckpt_w, extra], dim=1)
            logger.info(f"Expanded {pe_key}: {c_ckpt} -> {c_model} channels "
                        f"(new channels = mean of pretrained channels)")

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

    # --- guard against the silent-skip failure mode ---
    if pe_key in state_dict and any(pe_key in m for m in msg.missing_keys):
        raise RuntimeError(
            f"{pe_key} was NOT loaded (still in missing_keys) — patch_embed is random. "
            f"Check checkpoint nesting / key names."
        )
    
# def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
#     if urlparse(pretrained_weights).scheme:  # If it looks like an URL
#         state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
#     else:
#         state_dict = torch.load(pretrained_weights, map_location="cpu")
#     if checkpoint_key is not None and checkpoint_key in state_dict:
#         logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
#         state_dict = state_dict[checkpoint_key]
#     # remove `module.` prefix
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     # remove `backbone.` prefix induced by multicrop wrapper
#     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
#     msg = model.load_state_dict(state_dict, strict=False)
#     print(msg)
#     logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
def construct_vit_base_moco_head(device, pretrained_weight_path, checkpoint_key="model", global_args={}, **kwargs
) -> ViTHead:
    head_model = ViTHead(
        patch_size=16,
        embed_dim=768,
        num_register_tokens=0,
        **global_args)   
    
    load_moco_vit_backbone(head_model, pretrained_weight_path)
    
    return head_model

def construct_vit_base_head(device, pretrained_weight_path, checkpoint_key="model", global_args={}, **kwargs
) -> ViTHead:
    head_model = ViTHead(
        embed_dim=768,
        num_register_tokens=0,
        **global_args)   
    
    if global_args["dataset"] == "hpa":
        load_pretrained_weightss(head_model, pretrained_weight_path, checkpoint_key=checkpoint_key)
    else:
        load_pretrained_weights(head_model, pretrained_weight_path, checkpoint_key=checkpoint_key)
    
    return head_model

def construct_vit_base_head_to_device(device, pretrained_weight_path, checkpoint_key="model", **kwargs
) -> ViTHead:
    head_model = construct_vit_base_head(device=device, pretrained_weight_path=pretrained_weight_path, checkpoint_key=checkpoint_key, **kwargs
    )
    return head_model.to(device)


def construct_vit_base_backbone(device, pretrained_weight_path, checkpoint_key="model", **kwargs
) -> ViTBackbone:
    # print("Where are loading from huggingface!!!!!")

    backbone_model = _make_vit_base_backbone(**kwargs)
    load_pretrained_weights(backbone_model, pretrained_weight_path, checkpoint_key=checkpoint_key)

    if(pretrained_weight_path == "rad-dino"):
        path = hf_hub_download("microsoft/rad-dino", "backbone_compatible.safetensors")
        sd = load_file(path)
        print(sd)
        msg = backbone_model.load_state_dict(sd, strict=False)
        print(msg)
    return backbone_model

def construct_vit_base_backbone_to_device(device, pretrained_weight_path, checkpoint_key="model", **kwargs
) -> ViTBackbone:
    backbone_model = construct_vit_base_backbone(
        device=device,
        pretrained_weight_path=pretrained_weight_path,
        checkpoint_key=checkpoint_key,
        **kwargs,
    )
    return backbone_model.to(device)

def construct_vit_base_backbone_lora_to_device(device, global_args, pretrained_weight_path, checkpoint_key="model", **kwargs
) -> ViTBackbone:

    backbone_model = _make_vit_base_backbone(**kwargs)
    load_pretrained_weights(backbone_model, pretrained_weight_path, checkpoint_key=checkpoint_key)

    _apply_lora_trainable_policy(backbone_model, global_args["lora"])

    backbone_model.to(device)
    return backbone_model

def construct_dino_backbone(
    device, pretrained_weight_path, checkpoint_key='model', global_args=None, apply_trainable_policy=True, **kwargs
 
) -> DinoBackbone:
    backbone_model = DinoBackbone(device, global_args=global_args)
    # load_vit_backbone_pretrained(backbone_model, global_args['pretrained_path'], map_location=map_location)

    server_lora_args = dict(global_args["lora"])
    split_depth = backbone_model.split_depth
    if split_depth:
        original_unfreeze_blocks = server_lora_args.get("unfreeze_blocks")
        if original_unfreeze_blocks is not None:
            server_lora_args["unfreeze_blocks"] = [
                block_index - split_depth
                for block_index in original_unfreeze_blocks
                if split_depth <= block_index < backbone_model.model_depth
            ]

    _apply_lora_trainable_policy(backbone_model.student["backbone"], server_lora_args)
    _apply_lora_trainable_policy(backbone_model.teacher["backbone"], server_lora_args)

    # there is no backpropagation through the teacher, so no need for gradients (reset this after activating LoRA)
    for p in backbone_model.teacher.parameters():
        p.requires_grad = False
    backbone_model.to(device)
    return backbone_model

def construct_dino_head(device, pretrained_weight_path, checkpoint_key='model', in_chans=3, img_size=224, split_depth=0, global_args=None, **kwargs) -> DinoHead:
    model_args = global_args or {}
    model_depth = int(model_args.get("depth", 12))
    client_model = DinoHead(
        in_chans=in_chans,
        img_size=img_size,
        patch_size=model_args.get("patch_size", 14),
        embed_dim=model_args.get("embed_dim", 768),
        num_heads=model_args.get("num_heads", 12),
        mlp_ratio=model_args.get("mlp_ratio", 4),
        init_values=model_args.get("init_values", 1.0e-05),
        depth=split_depth,
    ).to(device)

    for param in client_model.parameters():
        param.requires_grad_(False)

    load_pretrained_weights(
        client_model,
        pretrained_weight_path,
        checkpoint_key=checkpoint_key,
        block_limit=split_depth if split_depth else None,
    )

    return client_model