from charset_normalizer.api import logger
import torch
from torch import nn

from models.dinov2.utils.param_groups import fuse_params_groups, get_params_groups_with_decay
from models.dinov2.utils.train_lora_util import activate_lora, deactivate_lora, delete_qkv, delete_qkv, is_merged


def _get_backbone_blocks(model):
    if not hasattr(model, "blocks"):
        return []


    if getattr(model, "chunked_blocks", False):
        blocks = []
        for chunk in model.blocks:
            for block in chunk:
                if isinstance(block, nn.Identity):
                    continue
                blocks.append(block)
        return blocks

    return list(model.blocks)

def normalize_backbone_key(key: str):
    prefixes = (
        "module.",
        "student_backbone.",
        "teacher_backbone.",
        "student.backbone.",
        "teacher.backbone.",
        "backbone.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key

def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate positional embeddings for different resolutions."""
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            logger.info("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
                antialias=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

def load_moco_vit_backbone(backbone, ckpt_path, map_location="cpu", verbose=True):
    """
    Loads MoCo v3 ViT-B `base_encoder` weights (trunk only) into a ViTBackbone.
    Drops momentum_encoder, predictor, and the base_encoder projection head.
    Position embeddings are interpolated if the checkpoint resolution differs.
    """
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    prefix = "module.base_encoder."
    new_state_dict = {}

    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue  # skip momentum_encoder.*, predictor.*
        stripped = k[len(prefix):]
        if stripped.startswith("head."):
            continue  # skip the MoCo projection head, not part of the backbone
        new_state_dict[stripped] = v

    if verbose:
        print(f"Extracted {len(new_state_dict)} backbone tensors "
              f"(from {len(state_dict)} total in checkpoint)")

    # must run after prefix stripping: it looks for a bare "pos_embed" key,
    # and mutates new_state_dict in place
    if hasattr(backbone, "patch_embed") and hasattr(backbone, "pos_embed"):
        interpolate_pos_embed(backbone, new_state_dict)

    missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)

    if verbose:
        print(f"\nMissing keys ({len(missing)}):")
        for m in missing:
            print(" ", m)
        print(f"\nUnexpected keys ({len(unexpected)}):")
        for u in unexpected:
            print(" ", u)

    return backbone

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, block_offset=0, block_limit=None):
   
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k.replace("backbone.", "")

        if new_k.startswith("blocks.0."):
            parts = new_k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                inner_idx = parts[2]
                rest = ".".join(parts[3:])
                new_k = f"blocks.{inner_idx}.{rest}"

        if new_k.startswith("blocks."):
            parts = new_k.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                block_index = int(parts[1])
                if block_limit is not None and not (block_offset <= block_index < block_limit):
                    continue
                new_k = ".".join(["blocks", str(block_index - block_offset), *parts[2:]])

        new_state_dict[new_k] = v
    # Interpolate pos_embed to match model dimensions
    if hasattr(model, "patch_embed") and hasattr(model, "pos_embed"):
        interpolate_pos_embed(model, new_state_dict)
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(msg)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def load_pretrained_moco_weights(model, pretrained_weights, checkpoint_key):
   
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k.replace("backbone.", "")

        if new_k.startswith("blocks.0."):
            parts = new_k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                inner_idx = parts[2]
                rest = ".".join(parts[3:])
                new_k = f"blocks.{inner_idx}.{rest}"

        new_state_dict[new_k] = v
    # Interpolate pos_embed to match model dimensions
    if hasattr(model, "patch_embed") and hasattr(model, "pos_embed"):
        interpolate_pos_embed(model, new_state_dict)
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(msg)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

def activate_lora_for_model(model, unfreeze_blocks, lora_layers, lora_rank, lora_attn_key, lora_attn_proj):
    if unfreeze_blocks is not None:
        lora_blocks = [idx for idx in list(range(len(model.blocks))) if idx not in unfreeze_blocks]
        for block_idx in lora_blocks:
            activate_lora(
                model.blocks[block_idx],
                lora_layers,
                lora_rank,
                include_attn_key=lora_attn_key,
                include_attn_proj=lora_attn_proj,
            )
    else:
        activate_lora(model, lora_layers, lora_rank, include_attn_key=lora_attn_key, include_attn_proj=lora_attn_proj)


def configure_trainable_params_for_model(
    model,
    *,
    unfreeze_blocks=None,
    unfreeze_embed=False,
    unfreeze_channel_embed=False,
    unfreeze_norm=False,
    unfreeze_cls_token=False,
    unfreeze_mask_token=False,
    unfreeze_register_tokens=True,
):
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
            continue

        if unfreeze_register_tokens and ("register_token" in name or "register_tokens" in name):
            param.requires_grad_(True)
            continue

        if unfreeze_embed and ("patch_embed" in name or "decoder_pred" in name):
            param.requires_grad_(True)
            continue

        if unfreeze_channel_embed and "channel_embed" in name:
            param.requires_grad_(True)
            continue

        if unfreeze_norm and "norm" in name:
            param.requires_grad_(True)
            continue

        if unfreeze_cls_token and "cls_token" in name:
            param.requires_grad_(True)
            continue

        if unfreeze_mask_token and "mask_token" in name:
            param.requires_grad_(True)

    if unfreeze_blocks is not None:
        blocks = _get_backbone_blocks(model)
        for block_idx in unfreeze_blocks:
            if 0 <= block_idx < len(blocks):
                for param in blocks[block_idx].parameters():
                    param.requires_grad_(True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("NUMBER OF TRAINABLE BACKBONE PARAMS: %d", trainable_params)
    return trainable_params


def _get_lora_save_settings(lora_config):
    if lora_config is None:
        return [11], ["attn"]
    return lora_config.get("unfreeze_blocks"), lora_config.get("layers", ["attn"])


def deactivate_lora_before_save(model, lora_config=None):
    model.train(False)
    assert is_merged(model)

    unfreeze_blocks, lora_layers = _get_lora_save_settings(lora_config)
    blocks = _get_backbone_blocks(model)
    if unfreeze_blocks is not None:
        unfreeze_set = set(unfreeze_blocks)
        lora_blocks = [idx for idx in range(len(blocks)) if idx not in unfreeze_set]
        for block_idx in lora_blocks:
            deactivate_lora(blocks[block_idx], activate_layers=lora_layers, delete_separate_proj=False)
    else:
        deactivate_lora(model, activate_layers=lora_layers, delete_separate_proj=False)


def reactivate_lora_after_save(model, lora_config=None):
    unfreeze_blocks, lora_layers = _get_lora_save_settings(lora_config)
    blocks = _get_backbone_blocks(model)
    if unfreeze_blocks is not None:
        unfreeze_set = set(unfreeze_blocks)
        lora_blocks = [idx for idx in range(len(blocks)) if idx not in unfreeze_set]
        for block_idx in lora_blocks:
            delete_qkv(blocks[block_idx], layers=lora_layers)
    else:
        delete_qkv(model, layers=lora_layers)
    model.train(True)
    assert not is_merged(model)

def get_params_groups_finetuning(models_dict, linear_layer, global_args):
    all_params_groups = []
    
    # ViT backbone (with LoRA) — layerwise decay, patch embed mult
    all_params_groups += get_maybe_fused_params_for_submodel(models_dict)
    
    # Linear head — higher LR, no decay splitting needed
    linear_params = {
        "params": [p for p in linear_layer.parameters() if p.requires_grad],
        "lr": global_args['lr'],
        "weight_decay": 0.0,              # small head, usually skip decay
        "foreach": True,
        "name": "linear_head",            # useful for logging
    }
    all_params_groups.append(linear_params)
    
    return all_params_groups

def get_maybe_fused_params_for_submodel(m):
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

