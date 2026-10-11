import copy
import os
from argparse import Namespace
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist

from data.dataset import SamplerType, make_dataset
from data.distributed_dataloader import make_distributed_data_loader
from data.partitioners.partition_factory import PartitionFactory
from models import _apply_lora_trainable_policy
from models.dinov2.utils import datasets
from models.dinov2.utils.misc import load_moco_vit_backbone, load_pretrained_weights
from models.mocov3.mocov3_backbone import MoCoBackbone
from models.mocov3.mocov3_head import MoCoHead
from models.mocov3.mocov3_schedulars import build_moco_schedulers
from models.mocov3.mocov3_transform import get_moco_vit_transforms, moco_collate_fn
from trainers.implementations.pretraining.mocosfl_training import MoCoSLTrainer
from utils.argument_utils import (
    build_base_argument_parser,
    expand_argument_parser_with_distributed_learning_parameters,
    namespace_to_dict,
    set_env_variables,
)
from utils.config_utils import set_random_seed
from utils.scheduler_utils import build_dino_optimizer, build_dino_schedulers


def setup_arguments() -> dict:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_distributed_learning_parameters(parser)
    parser.add_argument("--test_num_workers", type=int, default=8)
    args: Namespace = parser.parse_args()
    set_env_variables(args)
    args_dict = namespace_to_dict(args)

    moco_config = args_dict.get("moco_config", args_dict.get("dino_config", {}))
    if isinstance(moco_config, dict):
        merged = {}
        for key, value in args_dict.items():
            if key not in ("moco_config", "dino_config") and not str(key).startswith("_"):
                merged[key] = value
        merged.update(moco_config)
        args_dict["moco_config"] = merged

    return args_dict


if __name__ == "__main__":
    global_args = setup_arguments()["moco_config"]
    set_random_seed(global_args["random_seed"])

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
                    dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------- data
    partitioner = PartitionFactory.create(global_args["partitioner"])
    print(global_args.get("img_size", 224))
    print(global_args.get("dino_weight", False))


    train_transform = get_moco_vit_transforms(
        img_size=global_args.get("img_size", 224),
        in_chans=global_args.get("in_chans", 3),
        mean=global_args.get("norm_mean"),
        std=global_args.get("norm_std"),
        dataset=global_args.get("dataset")
    )
    train_dataset = make_dataset(
        dataset_str=global_args["dataset"],
        transform=train_transform,
        dataset_paths=global_args.get("dataset_paths"),
    )
    train_data_loaders = make_distributed_data_loader(
        dataset=train_dataset,
        batch_size=global_args["batch_size"],
        num_workers=global_args.get("num_workers", 4),
        partitioner_cls=partitioner,
        partitioner_kwargs_extra={"alpha": global_args["alpha"]},
        num_partitions=global_args["nr_of_clients"],
        sampler_type=SamplerType.INFINITE,
        drop_last=True,  # keep batch sizes uniform: BatchNorm in the MLP heads
        shuffle=True,
        seed=0,
        collate_fn=moco_collate_fn,
        persistent_workers=False,
    )

    client_train_iterators = [iter(loader) for loader in train_data_loaders]

    # ------------------------------------------- client heads (query + key)
    print(global_args.get("patch_size", 16))
    base_head = MoCoHead(
        img_size=global_args.get("img_size", 224),
        patch_size=global_args.get("patch_size", 16),
        in_chans=global_args.get("in_chans", 3),
        embed_dim=global_args.get("embed_dim", 768),
        freeze_patch_embed=global_args.get("moco", {}).get("freeze_patch_embed", True),
    ).to(device)
    if global_args.get("dino_weight", False):
        load_pretrained_weights(base_head, global_args["pretrained_weight_path"], checkpoint_key=global_args["checkpoint_key"])
    else:
        load_moco_vit_backbone(base_head, global_args["moco_checkpoint_path"])
         
    client_heads = {}
    client_key_heads = {}
    for client_id in range(global_args["nr_of_clients"]):
        head = base_head if client_id == 0 else copy.deepcopy(base_head)
        client_heads[client_id] = head

        key_head = copy.deepcopy(head)
        for p in key_head.parameters():
            p.requires_grad = False
        client_key_heads[client_id] = key_head

    # ------------------------------------------------------ server backbone
    server_backbone = MoCoBackbone(
        device,
        global_args=global_args,
        embed_dim=global_args.get("embed_dim", 768),
    )

    _apply_lora_trainable_policy(server_backbone.student["backbone"], global_args["lora"])
    _apply_lora_trainable_policy(server_backbone.teacher["backbone"], global_args["lora"])

    # there is no backpropagation through the teacher, so no need for gradients (reset this after activating LoRA)
    for p in server_backbone.teacher.parameters():
        p.requires_grad = False
    server_backbone.to(device)
    
    lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule = build_dino_schedulers(global_args)
    trainer = MoCoSLTrainer()
    trainer.train(
        device=device,
        server_model=server_backbone,
        server_optimizer=build_dino_optimizer(global_args, server_backbone.get_params_groups()),
        server_scheduler=(lr_schedule, wd_schedule, momentum_schedule, last_layer_lr_schedule),
        client_data_iterators=client_train_iterators,
        client_models=client_heads,
        client_key_models=client_key_heads,
        global_args=global_args,
    )   