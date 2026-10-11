import os
from argparse import Namespace

import numpy as np
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

# Shot counts to evaluate. Edit this list to run whichever configurations you want.
SHOT_COUNTS = [1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128]


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


def build_server_backbone(global_args, device):

    server_backbone_vit = models.construct_vit_base_backbone(
        device=device,
        pretrained_weight_path=global_args['pretrained_weight_path_server'],
        checkpoint_key=global_args['checkpoint_key_server'],
    ).to(device)
    for param in server_backbone_vit.parameters():
        param.requires_grad = False
    return server_backbone_vit


def run_few_shot(max_shots, global_args, partitioner, train_transform, test_data_loaders, device):
    """Run one full few-shot experiment for a given number of shots and return the
    average accuracy across clients (plus the per-client accuracies)."""
    # Reset the seed so every shot count starts from the same random state.
    max_shots_used = max_shots * global_args['nr_of_clients']
    print("making dataset...")
    train_dataset = make_dataset(
        dataset_str=global_args['dataset'],
        transform=train_transform,
        train=True,
        subsample_fraction=global_args['subsample_fraction'],
        max_shots=max_shots_used,
        seed=max_shots_used,
        dataset_paths=global_args.get("dataset_paths"),
    )
    print("Finished making dataset")

    train_data_loaders = make_distributed_data_loader(
        dataset=train_dataset,
        batch_size=global_args['batch_size'],
        num_workers=1,
        partitioner_cls=partitioner,
        partitioner_kwargs_extra={"alpha": 1000},
        num_partitions=global_args['nr_of_clients'],
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=True,
        seed=0,
        persistent_workers=False,
    )

    # Fresh backbone for this run (see build_server_backbone docstring).
    server_backbone_vit = build_server_backbone(global_args, device)
    if max_shots == 1:
        global_args['k'] = 1
    elif max_shots <= 20:
        global_args['k'] = max_shots - 1
    else:
        global_args['k'] = 20
    clients = UShapedClientsFactory().create(
        'few_shot', global_args['nr_of_clients'],
        train_data_loaders, test_data_loaders, device, global_args
    )

    client_accuracies = []
    for client_id, client in clients.items():
        client.tail.fit(client.head, server_backbone_vit, iter(client.train_iterator))
        print(f"[{max_shots}-shot] Client {client_id} finished training. Evaluating on test set...")
        correct = 0
        total = 0

        for x, y in iter(client.test_dataloader):
            x = x.to(client.tail.device)
            y = y.to(client.tail.device)

            preds = client.tail.predict(client.head, server_backbone_vit, x)

            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total
        client_accuracies.append(acc)

    average_accuracy = sum(client_accuracies) / len(client_accuracies)
    return average_accuracy, client_accuracies


if __name__ == '__main__':
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

    partitioner = PartitionFactory.create(global_args["partitioner"])
    train_transform = get_transform(is_train=False, dataset_name=global_args['dataset'])

    # The test set does not depend on the number of shots, so build it once and
    # reuse it across all shot counts.
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
        partitioner_kwargs_extra={"alpha": 1000},
        num_partitions=global_args['nr_of_clients'],
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
    )

    results = {}
    for shots in SHOT_COUNTS:
        print(f"\n===== Running {shots}-shot few-shot learning =====")
        average_accuracy, client_accuracies = run_few_shot(
            shots, global_args, partitioner, train_transform, test_data_loaders, device
        )
        results[shots] = average_accuracy
        print(f"[{shots}-shot] Per-client accuracies: "
              f"{[round(a, 4) for a in client_accuracies]}")
        print(f"[{shots}-shot] Average accuracy: {average_accuracy:.4f}")

    print("\n===== Summary =====")
    for shots in SHOT_COUNTS:
        print(f"{shots}-shot average accuracy: {results[shots]:.4f}")