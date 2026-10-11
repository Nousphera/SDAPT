# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import os
import logging
from collections import Counter
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np
import torch
from torch.utils.data import Sampler

from data.dataloader import make_data_loader
from data.dataset import SamplerType
from data.partitioners.dirichlet_partitioner import DirichletDataPartitioner
from data.partitioners.partitioner import Partitioner
from data.partitioners.random_partitioner import RandomPartitioner

from models.dinov2.utils.samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
T = TypeVar("T")

def make_distributed_data_loader(
    *,
    dataset,
    num_partitions: int,
    partitioner_cls: None,
    partitioner_kwargs_extra: dict = None,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    alpha: float = 100000,
):
    
    partitioner_kwargs = {
    "dataset": dataset,
    "num_partitions": num_partitions,
    "seed": seed,
    "shuffle": shuffle,
    **(partitioner_kwargs_extra or {}),
    }
    
    partitioner = None
    if partitioner_cls is not None: 
        partitioner = partitioner_cls(**(partitioner_kwargs))
    data_loaders = []
    if num_partitions == 1:
        data_loader = make_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            seed=seed,
            sampler_type=sampler_type,
            sampler_size=sampler_size,
            sampler_advance=sampler_advance,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn
        )
        return [data_loader]
    
    partitions = partitioner.load_partitions()
    if isinstance(partitioner, DirichletDataPartitioner):
        targets = np.asarray(dataset.targets)
        print("Dirichlet partition class counts:")
        for client_id, partition in enumerate(partitions):
            class_counts = Counter(targets[partition.indices].tolist())
            print(f"  client {client_id}: {dict(sorted(class_counts.items()))}")

    for partition in partitions:
        data_loader = make_data_loader(
            dataset=partition,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            seed=seed,
            sampler_type=sampler_type,
            sampler_size=sampler_size,
            sampler_advance=sampler_advance,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn
        )
        data_loaders.append(data_loader)

    return data_loaders