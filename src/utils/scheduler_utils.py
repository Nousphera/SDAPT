import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models.dinov2.utils.utils import CosineScheduler


def build_dino_optimizer(global_args, model):
    params_groups = model

    if not params_groups:
        return None

    optim_args = global_args["optim"]
    optimizer_name = optim_args.get("name", "adamw").lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params_groups,
            lr=optim_args["lr"],
            weight_decay=optim_args["weight_decay"],
            momentum=optim_args.get("momentum", 0.0),
            nesterov=optim_args.get("nesterov", False),
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params_groups,
            lr=optim_args["lr"],
            weight_decay=optim_args["weight_decay"],
            betas=(optim_args["adamw_beta1"], optim_args["adamw_beta2"]),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}. Expected 'adamw' or 'sgd'.")

def build_dino_schedulers(global_args):
    OFFICIAL_EPOCH_LENGTH = global_args["train"]["OFFICIAL_EPOCH_LENGTH"]
    lr = dict(
        base_value=global_args["optim"]["lr"],
        final_value=global_args["optim"]["min_lr"],
        total_iters=global_args["optim"]["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=global_args["optim"]["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=global_args["optim"]["weight_decay"],
        final_value=global_args["optim"]["weight_decay_end"],
        total_iters=global_args["optim"]["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=global_args["teacher"]["momentum_teacher"],
        final_value=global_args["teacher"]["final_momentum_teacher"],
        total_iters=global_args["optim"]["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=global_args["teacher"]["teacher_temp"],
        final_value=global_args["teacher"]["teacher_temp"],
        total_iters=global_args["teacher"]["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=global_args["teacher"]["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=global_args["teacher"]["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[: global_args["optim"]["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = (
        0  # mimicking the original schedules
    )

    return lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule

def get_u_shaped_split_avg_client_optimizer_and_scheduler(model, global_args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        global_args["nr_of_epochs"] * global_args["OFFICIAL_EPOCH_LENGTH"],
        eta_min=0,
    )
    return optimizer, scheduler

def get_u_shaped_split_avg_optimizer_and_scheduler(
    server_backbone,
    global_args,
):
    optim_cfg = global_args["optim"]

    base_lr = 0.001

    weight_decay = optim_cfg.get("weight_decay", 0.05)
    betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))

    epochs = global_args["nr_of_epochs"]

    warmup_epochs = optim_cfg.get("warmup_epochs", 3)
    warmup_start_factor = optim_cfg.get("warmup_start_factor", 1e-3)

    steps_per_epoch = global_args["OFFICIAL_EPOCH_LENGTH"]
    min_lr = 0.0

    client_params = []

    backbone_params = list(server_backbone.parameters())

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": base_lr,
            },
        ],
        betas=betas,
        weight_decay=weight_decay,
    )

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_lr,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    return optimizer, scheduler
