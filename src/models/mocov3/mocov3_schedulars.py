"""Schedulers for MoCo split-learning training.

Returns the same 4-tuple shape the trainer expects:
    (lr_schedule, wd_schedule, momentum_schedule, last_layer_lr_schedule)

There is no teacher-temperature schedule here — that's a DINO/iBOT concept.
MoCo's temperature is a fixed hyperparameter on the InfoNCE logits.
"""

import math

import numpy as np


def cosine_scheduler(
    base_value,
    final_value,
    total_iters,
    warmup_iters=0,
    start_warmup_value=0.0,
    freeze_iters=0,
):
    freeze_schedule = np.zeros(freeze_iters)
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iters - warmup_iters - freeze_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / max(len(iters), 1))
    )
    schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule


def moco_momentum_scheduler(base_momentum, total_iters):
    """m goes from base_momentum -> 1.0 on a cosine ramp (MoCo v3)."""
    iters = np.arange(total_iters)
    return 1.0 - (1.0 - base_momentum) * (np.cos(np.pi * iters / total_iters) + 1) * 0.5


def build_moco_schedulers(global_args):
    optim_cfg = global_args["optim"]
    epoch_len = global_args.get("train", {}).get(
        "OFFICIAL_EPOCH_LENGTH", global_args["OFFICIAL_EPOCH_LENGTH"]
    )
    total_epochs = global_args["nr_of_epochs"]
    total_iters = epoch_len * total_epochs

    # LR is specified per 256 samples, as is standard for MoCo/SimCLR recipes.
    effective_batch = (
        global_args["batch_size"]
        * global_args["nr_of_clients"]
        * optim_cfg["accum_iter"]
    )
    lr = optim_cfg["lr"] * effective_batch / 256.0

    lr_schedule = cosine_scheduler(
        base_value=lr,
        final_value=optim_cfg.get("min_lr", 1e-6),
        total_iters=total_iters,
        warmup_iters=optim_cfg.get("warmup_epochs", 10) * epoch_len,
        start_warmup_value=0.0,
    )
    wd_schedule = cosine_scheduler(
        base_value=optim_cfg.get("weight_decay", 0.1),
        final_value=optim_cfg.get("weight_decay_end", 0.1),
        total_iters=total_iters,
    )
    momentum_schedule = moco_momentum_scheduler(
        base_momentum=global_args.get("moco", {}).get("base_momentum", 0.99),
        total_iters=total_iters,
    )
    last_layer_lr_schedule = cosine_scheduler(
        base_value=lr,
        final_value=optim_cfg.get("min_lr", 1e-6),
        total_iters=total_iters,
        warmup_iters=optim_cfg.get("warmup_epochs", 10) * epoch_len,
        start_warmup_value=0.0,
        freeze_iters=optim_cfg.get("freeze_last_layer_epochs", 0) * epoch_len,
    )

    return lr_schedule, wd_schedule, momentum_schedule, last_layer_lr_schedule