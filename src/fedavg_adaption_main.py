import os
import copy
from functools import partial
from pyexpat import model
from argparse import Namespace
from collections import Counter
from charset_normalizer import cli
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import Linear
from data.dataset import SamplerType, make_dataset
from data.datasets.FMoW_wilds_dataset import dino_collate_fn, get_dinov2_pretrain_vit_transforms
from data.distributed_dataloader import make_distributed_data_loader
from data.partitioners.partition_factory import PartitionFactory
import models
from models.dinov2.utils import datasets
from models.dinov2.utils.transforms import get_transform
from trainers.implementations.pretraining.sl_dinov2_training import Dinov2SLTrainer
from trainers.implementations.pretraining.fl_dinov2_training import Dinov2FLTrainer
from trainers.implementations.u_shaped.u_shaped_psl import UShapedPSL
from utils.argument_utils import build_base_argument_parser, \
    expand_argument_parser_with_distributed_learning_parameters, set_env_variables, \
    namespace_to_dict
from utils.config_utils import set_random_seed
from utils.scheduler_utils import build_dino_optimizer, build_dino_schedulers, get_u_shaped_split_avg_client_optimizer_and_scheduler

def setup_arguments() -> dict:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_distributed_learning_parameters(parser)
    parser.add_argument('--test_num_workers', type=int, default=8,
                        help='num_workers provided to the test Dataloader. For Split Learning, we differentiate between num_workers for the train Dataloader and test_num_workers for the test Dataloader.')
    args: Namespace = parser.parse_args()
    set_env_variables(args)
    args_dict = namespace_to_dict(args)

    # Keep the merged YAML section for existing callers. CLI values are used as a
    # base so keys like pretrained_weight_path_server remain available, but the
    # config file (dino_config) takes precedence over CLI values on conflicts.
    dino_config = args_dict.get("dino_config", {})
    if isinstance(dino_config, dict):
        merged_dino_config = {}
        for key, value in args_dict.items():
            if key != "dino_config" and not str(key).startswith("_"):
                merged_dino_config[key] = value
        merged_dino_config.update(dino_config)
        args_dict["dino_config"] = merged_dino_config

    return args_dict

if __name__ == '__main__':
    print("starting")
    global_args = setup_arguments()["dino_config"]
    set_random_seed(global_args['random_seed'])

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device("cpu")


    ## partition dataset into clients
    print(global_args['alpha'])
    partitioner = PartitionFactory.create(global_args["partitioner"])
    train_transform = get_dinov2_pretrain_vit_transforms(True, global_args['dataset'])
    train_dataset = make_dataset(
        dataset_str=global_args['dataset'],
        transform=train_transform,
        dataset_paths=global_args.get("dataset_paths"),
    )
    train_data_loaders = make_distributed_data_loader(
        dataset=train_dataset,
        batch_size=global_args['batch_size'],
        num_workers=global_args.get("num_workers", 4),
        partitioner_cls=partitioner,
        partitioner_kwargs_extra={"alpha": global_args['alpha']},
        num_partitions=global_args['nr_of_clients'],
        sampler_type=SamplerType.INFINITE,
        drop_last=True,
        shuffle=True,
        seed=0,
        collate_fn=partial(
            dino_collate_fn,
            img_size=global_args.get("img_size", 224),
            patch_size=global_args.get("patch_size", 14),
            n_tokens=global_args.get("n_tokens"),
        ),
        persistent_workers=False,
    )
    client_train_iterators = [
        iter(loader) for loader in train_data_loaders
    ]
    print("loaded data")


    ## Dinov2 UDSL: n VIT heads, one DINOV2 backone
    client_heads = {}
    client_backbones = {}
    client_backbone_optimizers = {}
    client_backbone_schedulers = {}
    client_head_dinov2 = models.construct_dino_head(
        device,
        global_args['pretrained_weight_path'],
        checkpoint_key=global_args['checkpoint_key'],
        img_size=global_args.get('img_size', 224),
        split_depth=global_args.get('split_depth', 0),
        global_args=global_args,
    ).to(device)
    base_backbone = None
    for client_id in range(global_args['nr_of_clients']):
        _client_head_dinov2 = client_head_dinov2 if client_id == 0 else copy.deepcopy(client_head_dinov2)
   
        _client_backbone_dinov2 = models.construct_dino_backbone(
            device,
            global_args['pretrained_weight_path'],
            checkpoint_key=global_args['checkpoint_key'],
            global_args=global_args,
        ).to(device)

        if base_backbone is None:
            base_backbone = {k: v.detach().clone() for k, v in _client_backbone_dinov2.state_dict().items()}
        else:
            _client_backbone_dinov2.load_state_dict(base_backbone, strict=True)

        for name, p in _client_backbone_dinov2.named_parameters():
            if "lora_A" in name:
                p.requires_grad_(False)
        
        client_backbone_optimizers[client_id] = build_dino_optimizer(global_args, _client_backbone_dinov2.get_params_groups())
        client_backbone_schedulers[client_id] = build_dino_schedulers(global_args)
        client_heads[client_id] = _client_head_dinov2
        client_backbones[client_id] = _client_backbone_dinov2

    trainer = Dinov2FLTrainer()
    trainer.train(
        device=device,
        client_backbones=client_backbones,
        client_backbones_schedulers=client_backbone_schedulers,
        client_backbones_optimizers=client_backbone_optimizers,
        client_heads=client_heads,
        client_data_iterators=client_train_iterators,
        global_args=global_args
    )