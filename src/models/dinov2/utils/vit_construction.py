from functools import partial

from torch import distributed, nn
from models.dinov2.vision_transformer_layers import Block
from models.dinov2.layers.attention import MemEffAttention
from models.dinov2.lp_tail import AllClassifiers, LinearClassifier
from models.dinov2.utils.misc import activate_lora_for_model, load_moco_vit_backbone, load_pretrained_weights
from models.dinov2.vit_backbone import ViTBackbone
from models.dinov2.vit_head import ViTHead

def scale_lr(learning_rates, batch_size):
    return learning_rates

def construct_lp_tail(device, embed_dim=768, n_last_blocks_list=[1, 4], learning_rates=[1e-2], batch_size=32, num_classes=1000, **kwargs):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [False, True]:
            for _lr in learning_rates:
                lr =  scale_lr(_lr, batch_size)
                out_dim = n * embed_dim + (embed_dim if avgpool else 0)
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier
                linear_classifiers_dict[f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")] = (
                    linear_classifier
                )
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict, optim_param_groups=optim_param_groups).to(device)

    return linear_classifiers


def construct_segmentation_tail(device, embed_dim=768, n_last_blocks_list=[4], learning_rates=[1e-3], batch_size=32, num_classes=32, **kwargs):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [False, True]:
            for _lr in learning_rates:
                lr =  scale_lr(_lr, batch_size)
                out_dim = embed_dim * 4
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier
                linear_classifiers_dict[f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")] = (
                    linear_classifier
                )
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict, optim_param_groups=optim_param_groups).to(device)

    return linear_classifiers

def construct_vit_base_backbone(
    device,
    pretrained_weight_path,
    checkpoint_key="model",
    depth=12,
    block_offset=0,
    patch_size=14,
    embed_dim=768,
    num_heads=12,
    mlp_ratio=4,
    init_values=1.0e-05,
    **kwargs,
) -> ViTBackbone:
    backbone_model = ViTBackbone(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_uniform=True,
        init_values=init_values,
        block_fn=partial(Block, attn_class= MemEffAttention),
        num_register_tokens=kwargs.pop("num_register_tokens", 0),
        **kwargs)

    load_pretrained_weights(
        backbone_model,
        pretrained_weight_path,
        checkpoint_key=checkpoint_key,
        block_offset=block_offset,
        block_limit=block_offset + depth if block_offset else None,
    )
    return backbone_model

def construct_vit_moco_base_backbone(
    device,
    pretrained_weight_path,
    checkpoint_key="model",
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    **kwargs,
) -> ViTBackbone:
    backbone_model = ViTBackbone(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_uniform=True,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        block_fn=partial(Block, attn_class= MemEffAttention),
        num_register_tokens=kwargs.pop("num_register_tokens", 0),
        **kwargs)
    load_moco_vit_backbone(backbone_model, pretrained_weight_path)
    return backbone_model

def construct_vit_base_backbone_to_device(device, pretrained_weight_path, checkpoint_key="model", **kwargs
) -> ViTBackbone:
    backbone_model = ViTBackbone(
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_uniform=True,
        init_values=1.0e-05,  # for layerscale: None or 0 => no layerscale
        block_fn=partial(Block, attn_class= MemEffAttention),
        num_register_tokens=0,
        **kwargs).to(device)
  
    load_pretrained_weights(backbone_model, pretrained_weight_path, checkpoint_key=checkpoint_key)
    return backbone_model


def construct_vit_base_backbone_lora_to_device(device, global_args, pretrained_weight_path, checkpoint_key="model", **kwargs
) -> ViTBackbone:
    backbone_model = ViTBackbone(
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_uniform=True,
        init_values=1.0e-05,  # for layerscale: None or 0 => no layerscale
        block_fn=partial(Block, attn_class= MemEffAttention),
        num_register_tokens=0,
        **kwargs)
    
    load_pretrained_weights(backbone_model, pretrained_weight_path, checkpoint_key=checkpoint_key)
    
    activate_lora_for_model(
        backbone_model,
        global_args["lora"]["unfreeze_blocks"],
        global_args["lora"]["layers"],
        global_args["lora"]["rank"],
        global_args["lora"]["attn_key"],
        global_args["lora"]["attn_proj"],
    )

    for name, param in backbone_model.named_parameters():
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

    backbone_model.to(device)
    return backbone_model

def construct_vit_base_head(device, pretrained_weight_path, checkpoint_key="model", **kwargs
) -> ViTHead:
    head_model = ViTHead(
        patch_size=14,
        embed_dim=768,
        in_chans=3,
        num_register_tokens=0,
        **kwargs)
    
    load_pretrained_weights(head_model, pretrained_weight_path, checkpoint_key=checkpoint_key)
    
    return head_model


def construct_vit_base_head_to_device(device, pretrained_weight_path, checkpoint_key="model", **kwargs
) -> ViTHead:
    head_model = ViTHead(
        patch_size=14,
        embed_dim=768,
        in_chans=3,
        num_register_tokens=0,
        **kwargs).to(device)
    
    load_pretrained_weights(head_model, pretrained_weight_path, checkpoint_key=checkpoint_key)
    
    return head_model