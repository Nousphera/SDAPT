import json
import os

import torch
from models.dinov2.utils.utils import CosineScheduler
from xformers.ops import fmha

from models.dinov2.loss.dino_clstoken_loss import DINOLoss
from models.dinov2.loss.ibot_patch_loss import iBOTPatchLoss
from models.dinov2.loss.koleo_loss import KoLeoLoss
from models.dinov2.fsdp import get_fsdp_modules
from models.dinov2.utils.misc import deactivate_lora_before_save, reactivate_lora_after_save


class Dinov2FLTrainer:

    AMP_DTYPE = torch.bfloat16

    def __init__(self):
        self.dino_loss = DINOLoss(65536)
        self.ibot_patch_loss = iBOTPatchLoss(65536)
        self.koleo_loss = KoLeoLoss(dist_eps=1e-8)
        self.teacher = None
        self.momentum_schedule = CosineScheduler(**dict(
            base_value=0.90,
            final_value=1.0,
            total_iters=100 * 5,
        ))

    # ------------------------------------------------------------------ utils

    def _autocast(self):
        return torch.autocast("cuda", dtype=self.AMP_DTYPE)

    @staticmethod
    def _freeze_head(head):
        for p in head.parameters():
            p.requires_grad_(False)       # grads still flow *through* the head into the backbone

    @staticmethod
    def _get_ibot_head(module):
        return getattr(module, "ibot_head", None) or module.dino_head

    def _build_mask_metadata(self, batch, masks, device):
        """Prefer the metadata the collator already computed on CPU.

        Recomputing it here forces a device->host sync (via .nonzero() / .shape[0])
        on the critical path of every iteration.
        """
        if masks is None:
            return None, None, 0, torch.tensor([0], device=device, dtype=torch.long)

        if batch.get("mask_indices_list", None) is not None:
            mask_indices_list = batch["mask_indices_list"].cuda(non_blocking=True)
            masks_weight = batch["masks_weight"].cuda(non_blocking=True)
            n_masked = int(batch["n_masked_patches"])
        else:
            mask_indices_list = masks.flatten().nonzero().flatten().to(device)
            masks_weight = (
                (1 / masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(masks)[masks].to(device)
            )
            n_masked = mask_indices_list.shape[0]

        n_masked_tensor = torch.full((1,), n_masked, device=device, dtype=torch.long)
        return mask_indices_list, masks_weight, n_masked, n_masked_tensor

    # ------------------------------------------------------------------ loss

    @torch.no_grad()
    def get_teacher_output(self, backbone, teacher_global_crops, teacher_temp, n_masked_patches,
                           upperbound, mask_indices_list, n_masked_patches_tensor, use_ibot=True):
        teacher_out = backbone(teacher_global_crops, True)

        teacher_cls_tokens = teacher_out["x_norm_clstoken"].chunk(2)
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
        teacher_cls_after_head = backbone.teacher['dino_head'](teacher_cls_tokens)

        # Sinkhorn-Knopp is an iterative normalisation; run it in fp32 for stability.
        with torch.autocast("cuda", enabled=False):
            teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                teacher_cls_after_head.float(), teacher_temp=teacher_temp
            ).view(2, -1, *teacher_cls_after_head.shape[1:])

        masked_teacher_ibot_softmaxed_centered = None
        if use_ibot:
            teacher_patch_tokens = teacher_out["x_norm_patchtokens"]
            buffer_tensor_teacher = teacher_patch_tokens.new_zeros(upperbound, teacher_patch_tokens.shape[-1])
            torch.index_select(
                teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[:n_masked_patches],
            )
            masked_teacher_after_head = self._get_ibot_head(backbone.teacher)(buffer_tensor_teacher)[:n_masked_patches]
            with torch.autocast("cuda", enabled=False):
                masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    masked_teacher_after_head.float(),
                    teacher_temp=teacher_temp,
                    n_masked_patches_tensor=n_masked_patches_tensor,
                )

        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

    def _compute_loss(self, device, backbone, head, batch, teacher_temp, global_args):
        n_global_crops = 2
        n_local_crops = global_args["n_local_crops"]
        use_ibot = global_args.get("ibot_use", True)
        n_local_terms = max(n_local_crops * n_global_crops, 1)
        n_global_terms = (n_global_crops - 1) * n_global_crops
        ibot_loss_scale = 1.0 / n_global_crops
        loss_scales = 2

        global_crops = batch["collated_global_crops"].cuda(non_blocking=True)
        local_crops = batch["collated_local_crops"].cuda(non_blocking=True)
        masks = batch["collated_masks"].cuda(non_blocking=True) if use_ibot else None
        upperbound = batch["upperbound"] if use_ibot else 0

        with self._autocast():
            # The head is frozen and this branch only feeds the no-grad teacher,
            # so there is no reason to build a graph for it.
            with torch.no_grad():
                teacher_global = head(global_crops)

            student_global = head(global_crops, masks=masks)
            student_local = head(local_crops, masks=None)

            mask_indices_list, masks_weight, n_masked_patches, n_masked_patches_tensor = \
                self._build_mask_metadata(batch, masks, device)

            teacher_dino_list, masked_teacher_ibot = self.get_teacher_output(
                backbone, teacher_global, teacher_temp, n_masked_patches, upperbound,
                mask_indices_list, n_masked_patches_tensor, use_ibot=use_ibot,
            )

            student_global_out = backbone(student_global, False)
            student_local_out = backbone(student_local, False)
            student_global_cls = student_global_out["x_norm_clstoken"]

            inputs_for_student_head_list = [
                student_local_out["x_norm_clstoken"].unsqueeze(0),
                student_global_cls.unsqueeze(0),
            ]

            student_ibot_head = None
            if use_ibot:
                student_patch_tokens = student_global_out["x_norm_patchtokens"]
                buffer_patch_tokens = student_patch_tokens.new_zeros(upperbound, student_global_cls.shape[-1])
                buffer_patch_tokens[:n_masked_patches].copy_(
                    torch.index_select(student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
                )
                student_ibot_head = self._get_ibot_head(backbone.student)
                if student_ibot_head is backbone.student.dino_head:
                    inputs_for_student_head_list.append(buffer_patch_tokens.unsqueeze(0))
                else:
                    student_masked_patch_after_head = student_ibot_head(buffer_patch_tokens)[:n_masked_patches]

            attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
            outputs_list = attn_bias.split(backbone.student.dino_head(cat_inputs))

            student_local_cls_after_head = outputs_list.pop(0).squeeze(0)
            student_global_cls_after_head = outputs_list.pop(0).squeeze(0)
            if use_ibot and student_ibot_head is backbone.student.dino_head:
                student_masked_patch_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        # Losses in fp32: cross-entropy / logsumexp over a 65536-way head is the
        # one place where bf16's 8 mantissa bits actually bite.
        with torch.autocast("cuda", enabled=False):
            loss_dict = {}
            total = 0

            if n_local_crops > 0:
                dino_local = self.dino_loss(
                    student_output_list=student_local_cls_after_head.float().chunk(n_local_crops),
                    teacher_out_softmaxed_centered_list=teacher_dino_list,
                ) / (n_global_terms + n_local_terms)
                loss_dict["dino_local_crops_loss"] = dino_local
                total = total + dino_local

            dino_global = (
                self.dino_loss(
                    student_output_list=[student_global_cls_after_head.float()],
                    teacher_out_softmaxed_centered_list=[teacher_dino_list.flatten(0, 1)],
                )
                * loss_scales
                / (n_global_terms + n_local_terms)
            )
            loss_dict["dino_global_crops_loss"] = dino_global
            total = total + dino_global

            koleo = 0.1 * sum(self.koleo_loss(p.float()) for p in student_global_cls.chunk(2))
            loss_dict["koleo_loss"] = koleo / loss_scales
            total = total + koleo

            if use_ibot:
                ibot_patch = (
                    self.ibot_patch_loss.forward_masked(
                        student_masked_patch_after_head.float(),
                        masked_teacher_ibot,
                        student_masks_flat=masks,
                        n_masked_patches=n_masked_patches,
                        masks_weight=masks_weight,
                    )
                    * loss_scales
                    * ibot_loss_scale
                )
                loss_dict["ibot_loss"] = ibot_patch / 2
                total = total + ibot_patch

            # Keep this on-device. Calling .item() here would block the CPU on the
            # whole forward pass before backward is even queued.
            loss_dict["total_loss"] = torch.stack(
                [v.detach() for k, v in loss_dict.items() if k != "total_loss"]
            ).sum()

        batch_size = float(student_global.shape[0])   # from .shape, not a sync
        return total, loss_dict, batch_size

    # ------------------------------------------------------------------ local training

    def apply_optim_scheduler(self, optimizer, lr, wd, last_layer_lr):
        for group in optimizer.param_groups:
            group["weight_decay"] = wd * group["wd_multiplier"]
            group["lr"] = (last_layer_lr if group["is_last_layer"] else lr) * group["lr_multiplier"]

    def update_teacher(self, student, teacher, momentum):
        if student is None or teacher is None:
            return
        with torch.no_grad():
            for k in student.keys():
                student_modules = list(get_fsdp_modules(student[k])) or [student[k]]
                teacher_modules = list(get_fsdp_modules(teacher[k])) or [teacher[k]]
                for sm, tm in zip(student_modules, teacher_modules):
                    for sp, tp in zip(sm.parameters(), tm.parameters()):
                        tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)

    def _optimizer_step(self, backbone, optimizer, global_args):
        clip_grad = global_args["optim"]["clip_grad"]
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def _local_round(self, device, client_id, backbone, head, optimizer, schedules,
                     data_iterator, start_step, n_iters, global_args):
        """Train one client on its own data for `n_iters` iterations."""
        accum_iter = global_args["optim"]["accum_iter"]
        lr_s, wd_s, mom_s, temp_s, last_lr_s = schedules

        optimizer.zero_grad(set_to_none=True)
        samples, n_done, losses = 0.0, 0, {}

        for it in range(n_iters):
            step = start_step + it
            lr, wd, teacher_temp, last_layer_lr = lr_s[step], wd_s[step], temp_s[step], last_lr_s[step]
            self.apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

            try:
                batch = next(data_iterator)
            except StopIteration:
                break

            loss, loss_dict, batch_size = self._compute_loss(
                device, backbone, head, batch, teacher_temp, global_args
            )
            (loss / accum_iter).backward()
            del loss

            if (it + 1) % accum_iter == 0:
                self._optimizer_step(backbone, optimizer, global_args)
                log_dict = {"client_nr": client_id, "iteration": it, "lr": lr, "loss": loss_dict["total_loss"]}
                # print(log_dict, flush=True)
                # self.update_teacher(head, mom)

            samples += batch_size
            # Accumulate on-device; a single sync happens at the end of the round.
            for name, value in loss_dict.items():
                v = value.detach()
                losses[name] = losses[name] + v if name in losses else v
            n_done += 1

        # flush leftover gradients so nothing crosses the aggregation boundary
        if n_done % accum_iter != 0:
            self._optimizer_step(backbone, optimizer, global_args)
            # self.update_teacher(head, mom)

        if losses:
            names = list(losses.keys())
            values = (torch.stack([losses[n] for n in names]).float() / max(n_done, 1)).cpu().tolist()
            mean_losses = dict(zip(names, values))
        else:
            mean_losses = {}

        return {
            "samples": samples,
            "losses": mean_losses,
            "iters": n_done,
            "lr": lr_s[start_step],
        }

    # ------------------------------------------------------------------ aggregation

    @torch.no_grad()
    def federated_average(self, client_backbones, sample_counts=None):
        """FedAvg: weighted average of client backbones, broadcast back to every client."""
        client_ids = list(client_backbones.keys())
        if len(client_ids) < 2:
            return

        total = float(sum(max(sample_counts.get(c, 0.0), 0.0) for c in client_ids)) if sample_counts else 0.0
        if total > 0:
            weights = {c: max(sample_counts.get(c, 0.0), 0.0) / total for c in client_ids}
        else:
            weights = {c: 1.0 / len(client_ids) for c in client_ids}

        # state_dict() returns references, not copies. Average and broadcast one
        # key at a time so we never hold a full extra model on the device.
        states = {c: client_backbones[c].state_dict() for c in client_ids}

        for key, ref in states[client_ids[0]].items():
            if not torch.is_floating_point(ref):
                continue

            acc = torch.zeros_like(ref, dtype=torch.float32)
            for c in client_ids:
                acc.add_(states[c][key].detach().to(acc.dtype), alpha=weights[c])

            averaged = acc.to(ref.dtype)
            for c in client_ids:
                states[c][key].detach().copy_(averaged)
            del acc, averaged

    # ------------------------------------------------------------------ io

    def _write_metrics_json_row(self, metrics, eval_store_dir):
        logs_dir = os.path.join(eval_store_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        row = {k: (v.detach().float().item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
        with open(os.path.join(logs_dir, "training_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def do_test(self, backbone, head, iteration, eval_store_dir, global_args):
        eval_dir = os.path.join(eval_store_dir, "eval", str(iteration))
        os.makedirs(eval_dir, exist_ok=True)

        torch.cuda.synchronize()
        lora_config = global_args.get("lora")
        deactivate_lora_before_save(backbone, lora_config)
        checkpoint = {"backbone": backbone.state_dict()}
        if getattr(head, "teacher", None) is not None:
            checkpoint["teacher"] = head.teacher.state_dict()
        torch.save(checkpoint, os.path.join(eval_dir, "teacher_checkpoint.pth"))
        reactivate_lora_after_save(backbone, lora_config)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ epoch loop

    def train_epoch(self, device, client_backbones, client_backbones_optimizers, client_backbones_schedulers,
                    client_data_iterators, client_heads, epoch_nr, global_args, **_):
        epoch_length = global_args["OFFICIAL_EPOCH_LENGTH"]
        train_epoch_length = global_args.get("train", {}).get("OFFICIAL_EPOCH_LENGTH", epoch_length)
        local_iters = max(int(global_args.get("fed", {}).get("local_iters", epoch_length)), 1)
        client_ids = list(client_backbones.keys())

        for cid in client_ids:
            client_backbones[cid].train()
            client_heads[cid].train()
            self._freeze_head(client_heads[cid])

        epoch_start = (epoch_nr - 1) * train_epoch_length
        done, round_nr = 0, 0

        while done < epoch_length:
            n_iters = min(local_iters, epoch_length - done)
            samples, round_losses, lr = {}, {}, None

            for cid in client_ids:
                stats = self._local_round(
                    device, cid,
                    client_backbones[cid], client_heads[cid],
                    client_backbones_optimizers[cid], client_backbones_schedulers[cid],
                    client_data_iterators[cid], epoch_start + done, n_iters, global_args,
                )
                samples[cid] = stats["samples"]
                lr = stats["lr"]
                for name, value in stats["losses"].items():
                    round_losses[name] = round_losses.get(name, 0.0) + value / len(client_ids)

            mom = self.momentum_schedule[epoch_nr]
            self.federated_average(client_backbones, samples)
            for cid in client_ids:
                client_backbones_optimizers[cid].state.clear()
            self.update_teacher(client_backbones[client_ids[0]].student, self.teacher, momentum=mom)

            done += n_iters
            round_nr += 1
            log_dict = {"epoch": epoch_nr, "round": round_nr, "iteration": done, "lr": lr,
                        "samples": sum(samples.values()), **round_losses}
            print(log_dict, flush=True)
            self._write_metrics_json_row(log_dict, global_args["eval_store_dir"])

        if epoch_nr > 0 and epoch_nr % global_args["eval_epoch"] == 0:
            first = client_ids[0]
            self.do_test(
                client_backbones[first],
                client_heads[first],
                f"training_{epoch_nr}",
                global_args["eval_store_dir"],
                global_args,
            )

    def train(self, device, client_backbones, client_backbones_optimizers, client_backbones_schedulers,
              client_data_iterators, client_heads, global_args):
        client_ids = list(client_backbones.keys())
        self.teacher = client_backbones[client_ids[0]].teacher
        for client_id in client_ids:
            client_backbones[client_id].teacher = self.teacher
        for epoch_nr in range(global_args["nr_of_epochs"]):
            print(f"Starting epoch {epoch_nr + 1}", flush=True)
            self.train_epoch(
                device=device,
                client_backbones=client_backbones,
                client_backbones_optimizers=client_backbones_optimizers,
                client_backbones_schedulers=client_backbones_schedulers,
                client_data_iterators=client_data_iterators,
                client_heads=client_heads,
                epoch_nr=epoch_nr + 1,
                global_args=global_args,
            )