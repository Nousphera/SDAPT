import os
from argparse import Namespace
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.distributed as dist
from data.dataset import SamplerType, make_dataset
from data.datasets.cheXpert_dataset import CheXpertFromHuggingFace
from data.distributed_dataloader import make_distributed_data_loader
from data.partitioners.partition_factory import PartitionFactory
from evaluators.evaluator_factory import build_evaluator
from evaluators.hpa_evaluator import KaggleHPASubmissionEvaluator
import models
from models.dinov2.utils.transforms import get_transform
from models.u_shaped_client.u_shaped_client_factory import UShapedClientsFactory
from trainers.implementations.u_shaped.u_shaped_psl import UShapedPSL
from utils.argument_utils import build_base_argument_parser, \
    expand_argument_parser_with_distributed_learning_parameters, set_env_variables, \
    namespace_to_dict
from utils.config_utils import set_random_seed
from utils.scheduler_utils import get_u_shaped_split_avg_optimizer_and_scheduler

# Shot counts to evaluate. Edit this list to run whichever configurations you want.
SHOT_COUNTS = [4, 8, 16]

# The five CheXpert competition tasks. Edit this list to run a subset.
CHEXPERT_COMPETITION_TASKS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
    "Atelectasis",
    "Pleural Effusion",
    "Support Devices"
]


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


def build_test_data_loaders(task, global_args, partitioner, device):
    """Build the (task-specific) test data loaders, one per client.

    The test labels depend on the task, so this must be rebuilt for every task.
    It does NOT depend on the number of shots, so it can be reused across all
    shot counts for the same task."""
    test_dataset = CheXpertFromHuggingFace(
        csv_path="",
        transform=get_transform(is_train=False, dataset_name=global_args['dataset']),
        train=False,
        few_shot_task=task,
        uncertain_policy="zeros",   # or "ones"
        seed=0,
    )

    test_data_loaders = []
    for client_id in range(global_args['nr_of_clients']):
        test_data_loader = make_distributed_data_loader(
            dataset=test_dataset,
            batch_size=global_args['batch_size'],
            num_workers=1,
            partitioner_cls=partitioner,
            partitioner_kwargs_extra={"alpha": 1000},
            num_partitions=1,
            sampler_type=SamplerType.DISTRIBUTED,
            drop_last=False,
            shuffle=False,
            seed=0,
            persistent_workers=False,
        )
        test_data_loaders.append(test_data_loader[0])
    return test_data_loaders



def run_few_shot(task, max_shots, global_args, partitioner, train_transform, test_data_loaders, train_dataset_global, device):
    """Run one full few-shot experiment for a given task and number of shots and
    return the average F1 across clients (plus the per-client F1s)."""
    # Reset the seed so every (task, shot) combination starts from the same random state.
    set_random_seed(global_args['random_seed'])
    train_data_loaders = []
    for client_id in range(global_args['nr_of_clients']):
        train_dataset = train_dataset_global.make_few_shot_dataset(task=task, max_shots=max_shots, seed=client_id)
        
        train_data_loader = make_distributed_data_loader(
            dataset=train_dataset,
            batch_size=global_args['batch_size'],
            num_workers=1,
            partitioner_cls=partitioner,
            partitioner_kwargs_extra={"alpha": 1000},
            num_partitions=1,
            sampler_type=SamplerType.DISTRIBUTED,
            drop_last=False,
            shuffle=True,
            seed=0,
            persistent_workers=False,
        )
        train_data_loaders.append(train_data_loader[0])

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

    client_f1s = []
    client_aurocs = []
    for client_id, client in clients.items():
        client.tail.fit(client.head, server_backbone_vit, iter(client.train_iterator))
        print(f"[{task} | {max_shots}-shot] Client {client_id} finished training. Evaluating on test set...")

        all_scores = []
        all_labels = []

        for x, y in iter(client.test_dataloader):
            x = x.to(client.tail.device)
            y = y.to(client.tail.device)

            scores = client.tail.predict_proba(client.head, server_backbone_vit, x)

            all_scores.append(scores.detach().cpu())
            all_labels.append(y.detach().cpu())

        all_scores = torch.cat(all_scores).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Drop masked/uncertain rows (label -1) if uncertain_policy="ignore" leaked in.
        keep = all_labels != -1
        all_scores, all_labels = all_scores[keep], all_labels[keep]

        all_preds = (all_scores >= 0.5).astype(int)

        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        # roc_auc_score raises if the test split has only one class present.
        auroc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")

        client_f1s.append(f1)
        client_aurocs.append(auroc)
        print(f"[{task} | {max_shots}-shot] Client {client_id}: AUROC = {auroc:.4f}, F1 = {f1:.4f}")

    average_f1 = sum(client_f1s) / len(client_f1s) if client_f1s else float("nan")
    average_auroc = sum(client_aurocs) / len(client_aurocs) if client_aurocs else float("nan")
    return average_auroc, average_f1, client_aurocs, client_f1s


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

    partitioner = PartitionFactory.create('random')
    train_transform = get_transform(is_train=False, dataset_name=global_args['dataset'])

    # results[task][shots] = average F1 across clients
    results = {}
    train_data_loader =  CheXpertFromHuggingFace(
                    csv_path="",
                    transform=train_transform,
                    train=True,
                    uncertain_policy="zeros",
                )
    for task in CHEXPERT_COMPETITION_TASKS:
        print(f"\n########## Task: {task} ##########")
        results[task] = {}

        # The test set depends on the task but not on the number of shots, so
        # build it once per task and reuse it across all shot counts.
        test_data_loaders = build_test_data_loaders(task, global_args, partitioner, device)
      
        for shots in SHOT_COUNTS:
            print(f"\n===== Running {shots}-shot few-shot learning for '{task}' =====")
            average_auroc, average_f1, client_aurocs, client_f1s = run_few_shot(
                task, shots, global_args, partitioner, train_transform, test_data_loaders, train_data_loader, device
            )
            results[task][shots] = average_f1
            print(f"[{task} | {shots}-shot] Per-client AUROCs: {[round(a, 4) for a in client_aurocs]}")
            print(f"[{task} | {shots}-shot] Per-client F1s: {[round(a, 4) for a in client_f1s]}")
            print(f"[{task} | {shots}-shot] Average AUROC: {average_auroc:.4f}, Average F1: {average_f1:.4f}")

    print("\n===== Summary =====")
    header = "Task".ljust(20) + "".join(f"{s}-shot".rjust(12) for s in SHOT_COUNTS)
    print(header)
    print("-" * len(header))
    for task in CHEXPERT_COMPETITION_TASKS:
        row = task.ljust(20) + "".join(f"{results[task][s]:.4f}".rjust(12) for s in SHOT_COUNTS)
        print(row)

    # Mean over tasks for each shot count.
    mean_row = "MEAN".ljust(20)
    for s in SHOT_COUNTS:
        vals = [results[t][s] for t in CHEXPERT_COMPETITION_TASKS]
        mean_row += f"{sum(vals) / len(vals):.4f}".rjust(12)
    print("-" * len(header))
    print(mean_row)