# Similar to MPSL Trainer, but for unimodal models
#
# Single-GPU (H100) version:
#   - no FSDP / sharding (removed FSDP wrap, ShardingStrategy, MixedPrecision,
#     ModuleWrapPolicy, _reshard, get_fsdp_modules)
#   - bf16 via torch.autocast instead of fp16 + ShardedGradScaler
#     (bf16 doesn't need loss scaling: same exponent range as fp32)

import contextlib
import json
import os

import torch
from models.dinov2.loss.dino_clstoken_loss import DINOLoss
from models.dinov2.loss.ibot_patch_loss import iBOTPatchLoss
from models.dinov2.loss.koleo_loss import KoLeoLoss
from xformers.ops import fmha
from models.dinov2.utils.misc import deactivate_lora_before_save, reactivate_lora_after_save
from torch.utils.flop_counter import FlopCounterMode


class Dinov2SLTrainer():

    def __init__(self, autocast_dtype=torch.bfloat16):
        super().__init__()
        self.dino_loss = DINOLoss(65536)
        self.ibot_patch_loss = iBOTPatchLoss(65536)
        self.koleo_loss = KoLeoLoss(dist_eps=1e-8)
        self.autocast_dtype = autocast_dtype  # bf16 on H100
        self.client_fp = None
        self.client_bp = None
        self.server_fp = None
        self.server_bp = None

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _build_mask_metadata(self, combined_masks, device):
        if combined_masks is None:
            return None, None, 0, torch.tensor([0], device=device, dtype=torch.long)

        mask_indices_list = combined_masks.flatten().nonzero().flatten().to(device)
        masks_weight = (
            (1 / combined_masks.sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(combined_masks)[combined_masks]
            .to(device)
        )
        n_masked_patches = mask_indices_list.shape[0]
        n_masked_patches_tensor = torch.tensor([n_masked_patches], device=device, dtype=torch.long)

        return mask_indices_list, masks_weight, n_masked_patches, n_masked_patches_tensor

    def _get_ibot_head(self, module):
        ibot_head = getattr(module, "ibot_head", None)
        if ibot_head is None:
            return module.dino_head
        return ibot_head

    @staticmethod
    def _cat_crop_major(tensors, n_crops):
        """Each tensor is [n_crops * B_i, ...] and crop-major within its client.
        Returns one tensor that is crop-major across all clients:
        [crop1_client0, crop1_client1, ..., crop2_client0, crop2_client1, ...]
        """
        per_client_chunks = [t.chunk(n_crops, dim=0) for t in tensors]
        return torch.cat(
            [chunks[crop] for crop in range(n_crops) for chunks in per_client_chunks],
            dim=0,
        )

    def _single_iteration(self, device, server_model, server_optimizer, client_data_iterators: dict, client_models: dict, teacher_temp, global_args):
        n_global_crops = 2
        n_local_crops = global_args['n_local_crops']
        use_ibot = global_args.get('ibot_use', True)   # master switch for iBOT
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        ibot_loss_scale = 1.0 / n_global_crops
        intermediate_activations_per_client = {}
        combined_activations_entire_batch = {}
        mask_patch_tokens_upperbound = 0

        for client_id in range(global_args['nr_of_clients']):
            try:
                X = next(client_data_iterators[client_id])
            except StopIteration as exception:
                print(exception)
                print("problems in paradise")
                continue

            global_crops = X["collated_global_crops"].cuda(non_blocking=True)
            local_crops = X["collated_local_crops"].cuda(non_blocking=True)
            masks = X["collated_masks"].cuda(non_blocking=True) if use_ibot else None
            if use_ibot:
                mask_patch_tokens_upperbound += X["upperbound"]

            with self._autocast(), self.client_fp:
                client_teacher_global_crops = client_models[client_id](global_crops)
                client_student_global_crops = client_models[client_id](global_crops, masks=masks)   # masks=None when iBOT off
                client_student_local_crops = client_models[client_id](local_crops, masks=None)

            intermediate_activations_per_client[client_id] = {
                "teacher_global_crops": client_teacher_global_crops,
                "student_global_crops": client_student_global_crops,
                "student_local_crops": client_student_local_crops,
                "masks": masks,                     # None when iBOT off
                "upperbound": X["upperbound"],
            }

        if len(intermediate_activations_per_client) == 0:
            return {}

        n_crops_per_type = {
            "teacher_global_crops": n_global_crops,
            "student_global_crops": n_global_crops,
            "masks": n_global_crops,
            "student_local_crops": n_local_crops,
        }

        for activation_type in ("teacher_global_crops", "student_global_crops", "student_local_crops", "masks"):
            activations_per_type = [
                client_outputs[activation_type]
                for client_outputs in intermediate_activations_per_client.values()
                if activation_type in client_outputs and client_outputs[activation_type] is not None
            ]

            if len(activations_per_type) == 0:
                combined_activations_entire_batch[activation_type] = None
            elif len(activations_per_type) == 1:
                combined_activations_entire_batch[activation_type] = activations_per_type[0]
            else:
                combined_activations_entire_batch[activation_type] = self._cat_crop_major(
                    activations_per_type, n_crops_per_type[activation_type]
                )

        combined_masks = combined_activations_entire_batch["masks"]   # None when iBOT off
        mask_indices_list, masks_weight, n_masked_patches, n_masked_patches_tensor = self._build_mask_metadata(
            combined_masks,
            device,
        )

        with self._autocast(), self.server_fp:
            teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = self.get_teacher_output(
                server_model,
                combined_activations_entire_batch,
                teacher_temp,
                n_masked_patches,
                mask_patch_tokens_upperbound,
                mask_indices_list,
                n_masked_patches_tensor,
                use_ibot=use_ibot,
            )

            inputs_for_student_head_list = []
            student_global_backbone_output_dict = server_model(combined_activations_entire_batch['student_global_crops'], False)
            student_local_backbone_output_dict = server_model(combined_activations_entire_batch['student_local_crops'], False)

            student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
            inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

            student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
            inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

            # ---- iBOT student patch-token head (skipped entirely when iBOT off) ----
            if use_ibot:
                _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
                ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
                buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                    mask_patch_tokens_upperbound, _dim
                )
                buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                    torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
                )
                student_ibot_head = self._get_ibot_head(server_model.student)
                if student_ibot_head is server_model.student.dino_head:
                    inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
                else:
                    student_global_masked_patch_tokens_after_head = student_ibot_head(buffer_tensor_patch_tokens)[
                        :n_masked_patches
                    ]

            _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
            outputs_list = _attn_bias.split(server_model.student.dino_head(cat_inputs))

            student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
            student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
            if use_ibot and student_ibot_head is server_model.student.dino_head:
                student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

            loss_dict = {}
            loss_accumulator = 0

            if n_local_crops > 0:
                dino_local_crops_loss = self.dino_loss(
                    student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                    teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
                ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
                loss_dict["dino_local_crops_loss"] = dino_local_crops_loss
                loss_accumulator += dino_local_crops_loss

            loss_scales = 2

            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            loss_accumulator += dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens
            koleo_loss = 0.1 * sum(self.koleo_loss(p) for p in student_cls_tokens.chunk(2))
            loss_accumulator += koleo_loss
            loss_dict["koleo_loss"] = koleo_loss / loss_scales

            # ---- iBOT patch loss (skipped entirely when iBOT off) ----
            if use_ibot:
                ibot_patch_loss = (
                    self.ibot_patch_loss.forward_masked(
                        student_global_masked_patch_tokens_after_head,
                        masked_teacher_ibot_softmaxed_centered,
                        student_masks_flat=combined_masks,
                        n_masked_patches=n_masked_patches,
                        masks_weight=masks_weight,
                    )
                    * loss_scales
                    * ibot_loss_scale
                )
                loss_dict["ibot_loss"] = ibot_patch_loss / 2
                loss_accumulator += ibot_patch_loss

            loss_dict["total_loss"] = sum(
                loss_value.detach().float().item() if torch.is_tensor(loss_value) else float(loss_value)
                for loss_name, loss_value in loss_dict.items()
                if loss_name != "total_loss"
            )
            loss_dict["current_batch_size"] = float(combined_activations_entire_batch["student_global_crops"].shape[0])

        with self.server_bp:
            # bf16 needs no loss scaling, so we just call backward() directly
            (loss_accumulator / global_args["optim"]["accum_iter"]).backward()

            if hasattr(loss_accumulator, 'detach'):
                loss_accumulator = loss_accumulator.detach()
            del loss_accumulator

            del combined_activations_entire_batch
            if 'student_global_backbone_output_dict' in locals():
                del student_global_backbone_output_dict
            if 'student_local_backbone_output_dict' in locals():
                del student_local_backbone_output_dict
            if 'intermediate_activations_per_client' in locals():
                del intermediate_activations_per_client

            return loss_dict

    @torch.no_grad()
    def get_teacher_output(self, server_model, combined_activations_entire_batch, teacher_temp, n_masked_patches, upperbound, mask_indices_list, n_masked_patches_tensor, use_ibot=True):
        teacher_backbone_output_dict = server_model(combined_activations_entire_batch['teacher_global_crops'], True)
        teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
        teacher_cls_tokens = teacher_cls_tokens.chunk(2)
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))

        teacher_cls_tokens_after_head = server_model.teacher.dino_head(teacher_cls_tokens)
        teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
            teacher_cls_tokens_after_head, teacher_temp=teacher_temp
        ).view(2, -1, *teacher_cls_tokens_after_head.shape[1:])

        # ---- iBOT teacher branch (skipped when iBOT off) ----
        masked_teacher_ibot_softmaxed_centered = None
        if use_ibot:
            teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = teacher_patch_tokens.shape[-1]
            buffer_tensor_teacher = teacher_patch_tokens.new_zeros(upperbound, _dim)
            torch.index_select(
                teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[:n_masked_patches],
            )
            teacher_ibot_head = self._get_ibot_head(server_model.teacher)
            masked_teacher_patch_tokens_after_head = teacher_ibot_head(buffer_tensor_teacher)[:n_masked_patches]
            masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                masked_teacher_patch_tokens_after_head,
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
            )

        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

    def apply_optim_scheduler(self, optimizer, lr, wd, last_layer_lr):
        for param_group in optimizer.param_groups:
            is_last_layer = param_group["is_last_layer"]
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            param_group["weight_decay"] = wd * wd_multiplier
            param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

    def update_teacher(self, server_model, momentum):
        # Single-GPU: parameters aren't sharded, so a plain foreach EMA update
        # works directly (no need to walk FSDP-wrapped submodules).
        with torch.no_grad():
            for k in server_model.student.keys():
                student_params = list(server_model.student[k].parameters())
                teacher_params = list(server_model.teacher[k].parameters())
                torch._foreach_mul_(teacher_params, momentum)
                torch._foreach_add_(teacher_params, student_params, alpha=(1.0 - momentum))

    def do_test(self, server_model, iteration, eval_store_dir):
        new_state_dict = server_model.teacher.state_dict()

        iterstring = str(iteration)
        eval_dir = os.path.join(eval_store_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)

    def _write_metrics_json_row(self, metrics, eval_store_dir):
        logs_dir = os.path.join(eval_store_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        metrics_path = os.path.join(logs_dir, "training_metrics.jsonl")

        serializable_metrics = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                serializable_metrics[key] = value.detach().float().item()
            else:
                serializable_metrics[key] = value

        with open(metrics_path, "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(serializable_metrics) + "\n")

    def train_epoch(self, device, server_model, server_optimizer, server_scheduler, client_data_iterators, client_models, client_model_requires_any_grad, client_optimizers, client_schedulers, experiment_results, epoch_nr, global_args):
        """
        We are using activations combining; We combine all (modality-specific) client-side activations into a single large batch, requiring only a single FP on the server-model. This speeds up training.
        """
        print("yeehaa, we're training with DINO Trainer! Note that this trainer is designed for unimodal models, so make sure to use it with the appropriate model and data.")

        server_model.train()
        (
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            teacher_temp_schedule,
            last_layer_lr_schedule,
        ) = server_scheduler
        accum_iter = global_args["optim"]["accum_iter"]
        # use the TRAIN-level OFFICIAL_EPOCH_LENGTH (same unit used when building schedulers)
        train_official_epoch_length = global_args.get("train", {}).get("OFFICIAL_EPOCH_LENGTH", global_args.get("OFFICIAL_EPOCH_LENGTH", 1))

        # track optimizer-update steps for logging only
        update_step = train_official_epoch_length * epoch_nr

        self.client_fp = FlopCounterMode(display=False) if epoch_nr == 1 else contextlib.nullcontext()
        self.client_bp = FlopCounterMode(display=False) if epoch_nr == 1 else contextlib.nullcontext()
        self.server_fp = FlopCounterMode(display=False) if epoch_nr == 1 else contextlib.nullcontext()
        self.server_bp = FlopCounterMode(display=False) if epoch_nr == 1 else contextlib.nullcontext()

        prev_update_iter = (epoch_nr - 1) * train_official_epoch_length

        for iteration in range(global_args['OFFICIAL_EPOCH_LENGTH']):
            # increment update_step only at accumulation boundaries so schedules are indexed by optimizer updates
            if iteration % accum_iter == 0:
                prev_update_iter = iteration + ((epoch_nr - 1) * train_official_epoch_length)

            lr = lr_schedule[prev_update_iter]
            wd = wd_schedule[prev_update_iter]
            mom = momentum_schedule[prev_update_iter]
            teacher_temp = teacher_temp_schedule[prev_update_iter]
            last_layer_lr = last_layer_lr_schedule[prev_update_iter]
            self.apply_optim_scheduler(server_optimizer, lr, wd, last_layer_lr)

            if iteration % accum_iter == 0:
                server_optimizer.zero_grad(set_to_none=True)

            loss_dict = self._single_iteration(
                device,
                server_model,
                server_optimizer,
                client_data_iterators,
                client_models,
                teacher_temp,
                global_args)

            # clip gradients + step (no scaler needed for bf16)
            if (iteration + 1) % accum_iter == 0:
                with self.server_bp:
                    if global_args['optim']['clip_grad']:
                        for v in server_model.student.values():
                            torch.nn.utils.clip_grad_norm_(v.parameters(), global_args['optim']['clip_grad'])
                    server_optimizer.step()

                    self.update_teacher(server_model, mom)

                    if epoch_nr == 1 and iteration + 1 == accum_iter:

                        print("FLOPS\n")
                        print(f"Client FP: {self.client_fp.get_total_flops() / 1e9:.3f} GFLOPs")
                        print(f"Client BP: {self.client_bp.get_total_flops() / 1e9:.3f} GFLOPs")
                        print(f"Server FP: {self.server_fp.get_total_flops() / 1e9:.3f} GFLOPs")
                        print(f"Server BP: {self.server_bp.get_total_flops() / 1e9:.3f} GFLOPs")
                        self.client_fp = contextlib.nullcontext()
                        self.client_bp = contextlib.nullcontext()
                        self.server_fp = contextlib.nullcontext()
                        self.server_bp = contextlib.nullcontext()

            if (iteration + 1) % 10 == 0 or (iteration + 1) == global_args['OFFICIAL_EPOCH_LENGTH']:
                log_dict = {
                    "epoch": epoch_nr,
                    "iteration": iteration + 1,
                    "lr": lr,
                    "wd": wd,
                    "mom": mom,
                    "last_layer_lr": last_layer_lr,
                    "total_loss": loss_dict["total_loss"],
                }

                for loss_name, loss_value in loss_dict.items():
                    if loss_name in {"total_loss", "current_batch_size"}:
                        continue
                    if torch.is_tensor(loss_value):
                        loss_value = loss_value.detach().float().item()
                    log_dict[loss_name] = loss_value

                print(log_dict)
                self._write_metrics_json_row(log_dict, global_args['eval_store_dir'])

            if epoch_nr > 0 and epoch_nr % global_args['eval_epoch'] == 0:
                # Merge lora weights before saving the teacher model
                torch.cuda.synchronize()
                lora_config = dict(global_args.get("lora") or {})
                split_depth = int(getattr(server_model, "split_depth", 0))
                if split_depth and lora_config.get("unfreeze_blocks") is not None:
                    model_depth = len(server_model.student_backbone.blocks) + split_depth
                    lora_config["unfreeze_blocks"] = [
                        block_index - split_depth
                        for block_index in lora_config["unfreeze_blocks"]
                        if split_depth <= block_index < model_depth
                    ]
                deactivate_lora_before_save(server_model.student_backbone, lora_config)
                deactivate_lora_before_save(server_model.teacher_backbone, lora_config)

                self.do_test(server_model, f"training_{epoch_nr}", global_args['eval_store_dir'])
                torch.cuda.synchronize()

                reactivate_lora_after_save(server_model.student_backbone, lora_config)
                reactivate_lora_after_save(server_model.teacher_backbone, lora_config)
                torch.cuda.synchronize()

    def train(self, device, server_model, server_optimizer, server_scheduler, client_data_iterators, client_models, global_args):
        for epoch_nr in range(global_args['nr_of_epochs']):
            print(f'Starting epoch {epoch_nr + 1}')

            self.train_epoch(
                device=device,
                server_model=server_model,
                server_optimizer=server_optimizer,
                server_scheduler=server_scheduler,
                client_data_iterators=client_data_iterators,
                client_models=client_models,
                client_model_requires_any_grad=[],
                client_optimizers=None,
                client_schedulers=None,
                experiment_results=None,
                epoch_nr=epoch_nr + 1,
                global_args=global_args
            )