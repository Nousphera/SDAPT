import os
from argparse import Namespace

import torch
import torch.distributed as dist
from data.dataset import SamplerType, make_dataset
from data.distributed_dataloader import make_distributed_data_loader
from data.partitioners.partition_factory import PartitionFactory
from evaluators.evaluator_factory import build_evaluator
from evaluators.hpa_evaluator import KaggleHPASubmissionEvaluator
import models
from models.dinov2.utils.datasets import PROTEIN_COLS
from models.dinov2.utils.transforms import get_transform
from models.u_shaped_client.u_shaped_client_factory import UShapedClientsFactory
from trainers.implementations.u_shaped.u_shaped_psl import UShapedPSL
from utils.argument_utils import build_base_argument_parser, \
    expand_argument_parser_with_distributed_learning_parameters, set_env_variables, \
    namespace_to_dict
from utils.config_utils import set_random_seed
from utils.scheduler_utils import get_u_shaped_split_avg_optimizer_and_scheduler

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

def prevalence_thresholds(train_targets, t_min=0.25, t_max=0.5):
    prevalence = train_targets.mean(axis=0)
    counts = train_targets.sum(axis=0).astype(int)   # raw positive counts
    for name, p, c in sorted(zip(PROTEIN_COLS, prevalence, counts), key=lambda x: -x[1]):
        print(f"{name:32s} {p:6.3%}  ({c} imgs)")
    prevalence_clipped = np.clip(prevalence, t_min, t_max)
    print(prevalence_clipped)
    return prevalence_clipped

if __name__ == '__main__':
    global_args = setup_arguments()["dino_config"]
    partition_count = 25
    clients_to_train = 25
    set_random_seed(global_args['random_seed'])
    partition_seed = global_args.get('partition_seed')
    if partition_seed is None:
        partition_seed = global_args['random_seed']

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device("cpu")

    partitioner = PartitionFactory.create(global_args["partitioner"])
    train_transform = get_transform(is_train=True, dataset_name=global_args['dataset'])
    train_dataset = make_dataset(
        dataset_str=global_args['dataset'],
        transform=train_transform,
        train=True,
        subsample_fraction=global_args['subsample_fraction'],
        dataset_paths=global_args.get("dataset_paths"),
    )
    train_data_loaders = make_distributed_data_loader(
        dataset=train_dataset,
        batch_size=global_args['batch_size'],
        num_workers=1,
        partitioner_cls=partitioner,
        partitioner_kwargs_extra={"alpha": global_args["alpha"]},
        num_partitions=partition_count,
        sampler_type=SamplerType.INFINITE,
        drop_last=False,
        shuffle=True,
        seed=partition_seed,
        persistent_workers=False,
    )

    train_datasets = {}

    test_dataset = make_dataset(
        dataset_str=global_args['dataset'],
        transform=get_transform(is_train=False, dataset_name=global_args['dataset']),
        train=False,
        dataset_paths=global_args.get("dataset_paths"),
    )
    test_data_loaders = make_distributed_data_loader(
        dataset=test_dataset,
        batch_size=global_args['batch_size'],
        num_workers=1,
        partitioner_cls=partitioner,
        partitioner_kwargs_extra={"alpha": global_args["alpha"]},
        num_partitions=partition_count,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        seed=partition_seed,
        persistent_workers=False,
    )

    # Keep the 25-way data split, but instantiate and train only the first five clients.
    global_args['nr_of_clients'] = clients_to_train

    if global_args['freeze_server_backbone']:
        server_backbone_vit = models.construct_vit_base_backbone(device=device, pretrained_weight_path=global_args['pretrained_weight_path_server'], checkpoint_key=global_args['checkpoint_key_server'], patch_size=global_args['patch_size'],init_values=global_args['init_values']).to(device)
        for param in server_backbone_vit.parameters():
                param.requires_grad = False
        server_optimizer, server_schedular = None, None
    else:
        server_backbone_vit = models.construct_vit_base_backbone_lora_to_device(device=device, global_args=global_args, pretrained_weight_path=global_args['pretrained_weight_path_server'], checkpoint_key=global_args['checkpoint_key_server']).to(device)
        server_optimizer, server_schedular = get_u_shaped_split_avg_optimizer_and_scheduler(server_backbone_vit, global_args)
        
    evaluator = build_evaluator(global_args)
    criterion = None
    clients = UShapedClientsFactory().create(global_args['dataset'], global_args['nr_of_clients'], train_data_loaders, test_data_loaders, device, global_args, criterion)
    print(type(evaluator))
    trainer = UShapedPSL()
    trainer.train(
        device=device,
        server_model=server_backbone_vit,
        server_optimizer=server_optimizer,
        server_scheduler=server_schedular,
        clients=clients,
        evaluator=evaluator,
        global_args=global_args
    )


