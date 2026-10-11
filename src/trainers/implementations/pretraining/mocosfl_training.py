import contextlib
import json
import os

import torch
from tqdm import tqdm
from torch.utils.flop_counter import FlopCounterMode

from models.dinov2.utils.misc import deactivate_lora_before_save, reactivate_lora_after_save


class MoCoSLTrainer:

    def __init__(self, autocast_dtype=torch.bfloat16):
        super().__init__()
        self.autocast_dtype = autocast_dtype  # bf16 on H100
        self.client_fp = None
        self.client_bp = None
        self.server_fp = None
        self.server_bp = None

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    # ------------------------------------------------------------- utilities

    @staticmethod
    def _cat_clients(tensors):
        """Concatenate per-client activations along the batch dim, preserving
        client order so that view1[i] and view2[i] stay the same image."""
        if len(tensors) == 0:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)

    # -------------------------------------------------------------- one step

    def _single_iteration(
        self,
        device,
        server_model,
        client_data_iterators: dict,
        client_models: dict,
        client_key_models: dict,
        global_args,
    ):
        per_client = {}

        for client_id in range(global_args["nr_of_clients"]):
            try:
                X = next(client_data_iterators[client_id])
            except StopIteration as exception:
                print(exception)
                print(f"client {client_id} iterator exhausted, skipping")
                continue

            view1 = X["collated_view1"].to(device, non_blocking=True)
            view2 = X["collated_view2"].to(device, non_blocking=True)

            with self._autocast(), self.client_fp:
                # query path: gradients flow back into the client head
                q_tokens_v1 = client_models[client_id](view1)
                q_tokens_v2 = client_models[client_id](view2)

                # key path: momentum client head, never backpropped
                key_head = (
                    client_key_models[client_id]
                    if client_key_models is not None
                    else client_models[client_id]
                )
                with torch.no_grad():
                    k_tokens_v1 = key_head(view1).detach()
                    k_tokens_v2 = key_head(view2).detach()

            per_client[client_id] = {
                "q_tokens_v1": q_tokens_v1,
                "q_tokens_v2": q_tokens_v2,
                "k_tokens_v1": k_tokens_v1,
                "k_tokens_v2": k_tokens_v2,
            }

        if len(per_client) == 0:
            return {}, None

        combined = {
            name: self._cat_clients([c[name] for c in per_client.values()])
            for name in ("q_tokens_v1", "q_tokens_v2", "k_tokens_v1", "k_tokens_v2")
        }

        with self._autocast(), self.server_fp:
            # ---- key branch first (no grad) -------------------------------
            with torch.no_grad():
                k1 = server_model.forward_key(combined["k_tokens_v1"])
                k2 = server_model.forward_key(combined["k_tokens_v2"])

            # ---- query branch ---------------------------------------------
            q1 = server_model.forward_query(combined["q_tokens_v1"])
            q2 = server_model.forward_query(combined["q_tokens_v2"])

            # ---- symmetric InfoNCE against queue + in-batch negatives ------
            loss_accumulator = server_model.compute_loss(q1, q2, k1, k2)

            loss_dict = {
                "infonce_loss": loss_accumulator.detach().float().item(),
                "total_loss": loss_accumulator.detach().float().item(),
                "current_batch_size": float(q1.shape[0]),
                "queue_warm": float(bool(server_model.queue_warm)),
            }

        with self.server_bp:
            # bf16 needs no loss scaling, so we just call backward() directly
            (loss_accumulator / global_args["optim"]["accum_iter"]).backward()

        # enqueue AFTER the loss so positives never sit among their own negatives
        keys_to_enqueue = torch.cat([k1, k2], dim=0)

        del loss_accumulator, combined, per_client, q1, q2
        return loss_dict, keys_to_enqueue

    # --------------------------------------------------------------- updates

    def apply_optim_scheduler(self, optimizer, lr, wd, last_layer_lr):
        for param_group in optimizer.param_groups:
            is_last_layer = param_group["is_last_layer"]
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            param_group["weight_decay"] = wd * wd_multiplier
            param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

    @torch.no_grad()
    def update_key_encoder(self, server_model, client_models, client_key_models, momentum):

        def _momentum_copy(src_module, dst_module):
            src_params = list(src_module.parameters())
            dst_params = list(dst_module.parameters())
            torch._foreach_mul_(dst_params, momentum)
            torch._foreach_add_(dst_params, src_params, alpha=(1.0 - momentum))

        for k in server_model.teacher.keys():
            _momentum_copy(server_model.student[k], server_model.teacher[k])

        if client_key_models is not None:
            for client_id, key_head in client_key_models.items():
                _momentum_copy(client_models[client_id], key_head)

    # ------------------------------------------------------------ checkpoints

    def do_test(self, server_model, iteration, eval_store_dir):
        """Save the momentum (key) branch — that's the encoder you transfer."""
        new_state_dict = server_model.teacher.state_dict()

        eval_dir = os.path.join(eval_store_dir, "eval", str(iteration))
        os.makedirs(eval_dir, exist_ok=True)
        torch.save({"teacher": new_state_dict}, os.path.join(eval_dir, "key_encoder_checkpoint.pth"))

    def _write_metrics_json_row(self, metrics, eval_store_dir):
        logs_dir = os.path.join(eval_store_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        metrics_path = os.path.join(logs_dir, "training_metrics.jsonl")

        serializable_metrics = {}
        for key, value in metrics.items():
            serializable_metrics[key] = (
                value.detach().float().item() if torch.is_tensor(value) else value
            )

        with open(metrics_path, "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(serializable_metrics) + "\n")

    # ------------------------------------------------------------ train loop

    def train_epoch(
        self,
        device,
        server_model,
        server_optimizer,
        server_scheduler,
        client_data_iterators,
        client_models,
        client_key_models,
        epoch_nr,
        global_args,
    ):
        print("Training with the MoCo split-learning trainer (v3 heads + v2 queue, single-GPU bf16).")

        server_model.train()
        lr_schedule, wd_schedule, momentum_schedule, last_layer_lr_schedule = server_scheduler

        accum_iter = global_args["optim"]["accum_iter"]
        train_official_epoch_length = global_args.get("train", {}).get(
            "OFFICIAL_EPOCH_LENGTH", global_args.get("OFFICIAL_EPOCH_LENGTH", 1)
        )

        self.client_fp = FlopCounterMode(display=False) if epoch_nr == 1 else contextlib.nullcontext()
        self.client_bp = FlopCounterMode(display=False) if epoch_nr == 1 else contextlib.nullcontext()
        self.server_fp = FlopCounterMode(display=False) if epoch_nr == 1 else contextlib.nullcontext()
        self.server_bp = FlopCounterMode(display=False) if epoch_nr == 1 else contextlib.nullcontext()

        prev_update_iter = (epoch_nr - 1) * train_official_epoch_length

        for iteration in range(global_args["OFFICIAL_EPOCH_LENGTH"]):
            if iteration % accum_iter == 0:
                prev_update_iter = iteration + ((epoch_nr - 1) * train_official_epoch_length)
                server_optimizer.zero_grad(set_to_none=True)

            lr = lr_schedule[prev_update_iter]
            wd = wd_schedule[prev_update_iter]
            mom = momentum_schedule[prev_update_iter]
            last_layer_lr = last_layer_lr_schedule[prev_update_iter]
            self.apply_optim_scheduler(server_optimizer, lr, wd, last_layer_lr)

            loss_dict, keys_to_enqueue = self._single_iteration(
                device,
                server_model,
                client_data_iterators,
                client_models,
                client_key_models,
                global_args,
            )

            if (iteration + 1) % accum_iter == 0:
                with self.server_bp:
                    # clip gradients + step (no scaler needed for bf16)
                    if global_args["optim"]["clip_grad"]:
                        for v in server_model.student.values():
                            torch.nn.utils.clip_grad_norm_(
                                v.parameters(), global_args["optim"]["clip_grad"]
                            )
                    server_optimizer.step()

                    # momentum update AFTER the optimiser step
                    self.update_key_encoder(server_model, client_models, client_key_models, mom)

                    if epoch_nr == 1 and iteration + 1 == accum_iter:
                        print("FLOPS\n")
                        print(f"Client FP: {self.client_fp.get_total_flops() / 1e9:.3f} GFLOPs")
                        print(f"Server FP: {self.server_fp.get_total_flops() / 1e9:.3f} GFLOPs")
                        print(f"Server BP: {self.server_bp.get_total_flops() / 1e9:.3f} GFLOPs")
                        self.client_fp = contextlib.nullcontext()
                        self.client_bp = contextlib.nullcontext()
                        self.server_fp = contextlib.nullcontext()
                        self.server_bp = contextlib.nullcontext()

            # queue update happens every micro-batch (keys are already detached)
            if keys_to_enqueue is not None:
                server_model.dequeue_and_enqueue(keys_to_enqueue)
                del keys_to_enqueue

            if (iteration + 1) % 10 == 0 or (iteration + 1) == global_args["OFFICIAL_EPOCH_LENGTH"]:
                log_dict = {
                    "epoch": epoch_nr,
                    "iteration": iteration + 1,
                    "lr": lr,
                    "wd": wd,
                    "mom": mom,
                    "last_layer_lr": last_layer_lr,
                }
                log_dict.update(loss_dict)
                print(log_dict)
                self._write_metrics_json_row(log_dict, global_args["eval_store_dir"])

        if epoch_nr > 0 and epoch_nr % global_args["eval_epoch"] == 0:
            torch.cuda.synchronize()
            lora_config = global_args.get("lora")
            deactivate_lora_before_save(server_model.student_backbone, lora_config)
            deactivate_lora_before_save(server_model.teacher_backbone, lora_config)
            self.do_test(server_model, f"training_{epoch_nr}", global_args["eval_store_dir"])
            reactivate_lora_after_save(server_model.student_backbone, lora_config)
            reactivate_lora_after_save(server_model.teacher_backbone, lora_config)
            torch.cuda.synchronize()

    def train(
        self,
        device,
        server_model,
        server_optimizer,
        server_scheduler,
        client_data_iterators,
        client_models,
        client_key_models,
        global_args,
    ):
        for epoch_nr in range(global_args["nr_of_epochs"]):
            print(f"Starting epoch {epoch_nr + 1}")
            self.train_epoch(
                device=device,
                server_model=server_model,
                server_optimizer=server_optimizer,
                server_scheduler=server_scheduler,
                client_data_iterators=client_data_iterators,
                client_models=client_models,
                client_key_models=client_key_models,
                epoch_nr=epoch_nr + 1,
                global_args=global_args,
            )