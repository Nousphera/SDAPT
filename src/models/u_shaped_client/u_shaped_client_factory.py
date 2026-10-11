import copy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from models import construct_vit_base_head, construct_vit_base_moco_head
from models.dinov2.lp_tail import KnnHead
from models.u_shaped_client.cheXpert_loss import MaskedBCEWithLogitsLoss
from models.u_shaped_client.u_shaped_client import UShapedClient
from utils.scheduler_utils import get_u_shaped_split_avg_client_optimizer_and_scheduler

MOCO_WEIGHT_KEY = "moco/vit-b"
VIT_BASE_EMBED_DIM = 768


@dataclass(frozen=True)
class _DatasetSpec:
    """Everything that differs between datasets."""

    build_tail: Callable[[str, dict], nn.Module]
    criterion: Optional[Callable[[], nn.Module]] = None
    # Only chexpert/fmow ever looked at the MoCo checkpoint in the original code.
    allow_moco: bool = False
    # few_shot builds clients without an optimizer and without wrapping the
    # train dataset in iter().
    knn: bool = False
    # hpa takes the criterion passed in by the caller instead of building one.
    use_given_criterion: bool = False
    # See the note in create(): "rvl" never returned any clients.
    build_clients: bool = True


class UShapedClientsFactory:
    # ------------------------------------------------------------------ heads

    @staticmethod
    def _build_head(device, global_args, allow_moco=False):
        if allow_moco and global_args['pretrained_weight_path_client'] == MOCO_WEIGHT_KEY:
            moco_checkpoint = global_args.get("moco_checkpoint_path")
            if not moco_checkpoint:
                raise ValueError("moco_checkpoint_path is required when using the MoCo client head")
            return construct_vit_base_moco_head(device, moco_checkpoint).to(device)

        return construct_vit_base_head(
            device,
            global_args['pretrained_weight_path_client'],
            checkpoint_key=global_args['checkpoint_key_client'],
            global_args=global_args,
        ).to(device)

    # ------------------------------------------------------------------ tails

    @staticmethod
    def _feature_dim(global_args):
        n_layers = global_args.get("feature_layers", 1)
        if isinstance(n_layers, (list, tuple)):
            n_layers = len(n_layers)
        tokens_per_layer = 2 if global_args.get("use_patch_mean_pooling", True) else 1
        return VIT_BASE_EMBED_DIM * tokens_per_layer * n_layers  # 6144 for the default

    @classmethod
    def _build_bn_linear_tail(cls, device, global_args):
        feature_dim = cls._feature_dim(global_args)
        return nn.Sequential(
            nn.BatchNorm1d(feature_dim, affine=False),
            nn.Linear(feature_dim, global_args['nr_of_classes']),
        ).to(device)

    @staticmethod
    def _build_linear_tail(device, global_args):
        return nn.Linear(
            in_features=VIT_BASE_EMBED_DIM,
            out_features=global_args['nr_of_classes'],
        ).to(device)

    @staticmethod
    def _build_knn_tail(device, global_args):
        return KnnHead(k=global_args['k'])

    # --------------------------------------------------------------- registry

    @classmethod
    def _specs(cls):
        return {
            "chexpert": _DatasetSpec(
                build_tail=cls._build_bn_linear_tail,
                criterion=MaskedBCEWithLogitsLoss,
                allow_moco=True,
            ),
            "fmow": _DatasetSpec(
                build_tail=cls._build_bn_linear_tail,
                criterion=nn.CrossEntropyLoss,
                allow_moco=True,
            ),
            "rvl": _DatasetSpec(
                build_tail=cls._build_linear_tail,
                criterion=nn.CrossEntropyLoss,
                build_clients=False,
            ),
            "domain_quickdraw": _DatasetSpec(
                build_tail=cls._build_bn_linear_tail,
                criterion=nn.CrossEntropyLoss,
            ),
            "domain_sketch": _DatasetSpec(
                build_tail=cls._build_linear_tail,
                criterion=nn.CrossEntropyLoss,
            ),
            "hpa": _DatasetSpec(
                build_tail=cls._build_linear_tail,
                use_given_criterion=True,
            ),
            "few_shot": _DatasetSpec(
                build_tail=cls._build_knn_tail,
                criterion=nn.CrossEntropyLoss,
                knn=True,
            ),
        }

    # ------------------------------------------------------------------ build

    @staticmethod
    def _build_clients(nr_of_clients, client_head, client_tail, client_train_dataset,
                       client_test_dataset, criterion, global_args, knn=False,
                       client_head_factory=None, client_tail_factory=None):
        clients = {}
        for client_id in range(nr_of_clients):
            if client_head_factory is None:
                # Preserve the behavior of the legacy create_clients API.
                head = client_head if client_id == 0 else copy.deepcopy(client_head)
            else:
                head = client_head_factory(client_id)
            if client_tail_factory is None:
                # Preserve the behavior of the legacy create_clients API.
                tail = client_tail if client_id == 0 else copy.deepcopy(client_tail)
            else:
                tail = client_tail_factory(client_id)

            if knn:
                optimizer, scheduler = None, None
                train_iterator = client_train_dataset[client_id]
            else:
                optimizer, scheduler = get_u_shaped_split_avg_client_optimizer_and_scheduler(
                    tail, global_args
                )
                train_iterator = iter(client_train_dataset[client_id])

            clients[client_id] = UShapedClient(
                client_id=client_id,
                client_head=head,
                client_tail=tail,
                client_optimizer=optimizer,
                client_scheduler=scheduler,
                client_train_iterator=train_iterator,
                client_test_dataloader=client_test_dataset[client_id],
                criterion=criterion,
            )
        return clients

    # Kept for backwards compatibility with external callers. `client_optimizer`
    # and `client_schedulers` were ignored in the original code too: every client
    # gets its own optimizer built from its own tail.
    def create_clients(self, dataset, nr_of_clients, client_head, client_tail, client_optimizer,
                       client_schedulers, client_train_dataset, client_test_dataset, criterion,
                       global_args):
        return self._build_clients(
            nr_of_clients, client_head, client_tail, client_train_dataset,
            client_test_dataset, criterion, global_args, knn=False,
        )

    def create_clients_knn(self, dataset, nr_of_clients, client_head, client_tail, client_optimizer,
                           client_schedulers, client_train_dataset, client_test_dataset, criterion,
                           global_args):
        return self._build_clients(
            nr_of_clients, client_head, client_tail, client_train_dataset,
            client_test_dataset, criterion, global_args, knn=True,
        )

    def create(self, dataset, nr_of_clients, client_train_dataset, client_test_dataset, device,
               global_args, criterion_given=None):
        spec = self._specs().get(dataset)
        if spec is None:
            raise ValueError(f"Unknown dataset")

        client_head = self._build_head(device, global_args, allow_moco=spec.allow_moco)
        client_tail = spec.build_tail(device, global_args)
        criterion = criterion_given if spec.use_given_criterion else spec.criterion()

        if not spec.build_clients:
            # Behaviour preserved from the original: the "rvl" branch built the
            # head, tail, optimizer and criterion but never called create_clients,
            # so create() returned None. Probably a bug -- flip build_clients to
            # True in the spec above to fix it.
            return None

        def build_client_head(client_id):
            client_seed = global_args.get("random_seed", 0) + client_id
            with torch.random.fork_rng():
                torch.manual_seed(client_seed)
                return self._build_head(device, global_args, allow_moco=spec.allow_moco)

        def build_client_tail(client_id):
            client_seed = global_args.get("random_seed", 0) + client_id
            with torch.random.fork_rng():
                torch.manual_seed(client_seed)
                return spec.build_tail(device, global_args)

        return self._build_clients(
            nr_of_clients, client_head, client_tail, client_train_dataset,
            client_test_dataset, criterion, global_args, knn=spec.knn,
            client_head_factory=build_client_head,
            client_tail_factory=build_client_tail,
        )