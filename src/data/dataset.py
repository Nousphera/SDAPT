# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import logging
import os
import inspect
from enum import Enum
from typing import Callable, Optional 
import torch
from torch.utils.data import Sampler
from data.datasets.FMoW_HF_dataset import FMoWFromHuggingFace
from data.datasets.HPA_dataset import HPADataset
from data.datasets.FMoW_wilds_dataset import FMoWFromWILDS
from data.datasets.RVL_dataset import RVLCDIPFromHuggingFace
from data.datasets.cheXpert_dataset import CheXpertFromHuggingFace
from data.datasets.quickdraw_dataset import DomainNetQuickdrawDataset
from data.datasets.rxrx1_dataset import RxRx1FromWILDS
from data.datasets.sketch_dataset import DomainNetSketchDataset
from models.dinov2.utils.datasets import (
    BDDDataset,
    DiorFromHuggingFace,
    ViperWeatherDataset,
)
from models.dinov2.utils.samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler


logger = logging.getLogger("linprobe")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4

def _parse_dataset_str(dataset_str: str):
    name = dataset_str

    dataset_map = {
        "fmow": FMoWFromHuggingFace,
        "chexpert": CheXpertFromHuggingFace,
        "dior": DiorFromHuggingFace,
        "domain_sketch": DomainNetSketchDataset,
        "viper": ViperWeatherDataset,
        "bdd": BDDDataset,
        "hpa": HPADataset,
        "domain_quickdraw": DomainNetQuickdrawDataset,
        "rxrx1": RxRx1FromWILDS,
        "rvl": RVLCDIPFromHuggingFace,
    }

    if name in dataset_map:
        return dataset_map[name], {"csv_path": name}

    raise ValueError(f'Unsupported dataset "{name}"')

def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    max_shots: int = 0,
    seed: int = 0,
    train = True,
    subsample_fraction: float = 1.0,
    isMoco: bool = False,
    dataset_paths: Optional[dict] = None,
):
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    kwargs["train"] = train
    kwargs["subsample_fraction"] = subsample_fraction
    dataset_paths = {
        key: os.path.expandvars(os.path.expanduser(str(value)))
        for key, value in (dataset_paths or {}).items()
        if value is not None
    }
    constructor_parameters = inspect.signature(class_.__init__).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in constructor_parameters.values()
    )
    if "dataset_paths" in constructor_parameters or accepts_kwargs:
        kwargs["dataset_paths"] = dataset_paths

    if class_ is FMoWFromHuggingFace:
        kwargs["max_shots"] = max_shots
        kwargs["seed"] = seed
    if class_ is RxRx1FromWILDS:
        kwargs["max_shots"] = max_shots
    if class_ is DomainNetSketchDataset:
        kwargs["max_shots"] = max_shots
    if class_ is DomainNetQuickdrawDataset:
        kwargs["max_shots"] = max_shots
    if class_ is ViperWeatherDataset:
        kwargs["max_shots"] = max_shots
    if class_ is DiorFromHuggingFace:
        kwargs["max_shots"] = max_shots
    if class_ is RVLCDIPFromHuggingFace:
        kwargs["max_shots"] = max_shots
    if class_ is BDDDataset:
        kwargs["max_shots"] = max_shots
    dataset = class_(transform=transform, target_transform=target_transform, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None