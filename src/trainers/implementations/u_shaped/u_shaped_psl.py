# Similar to MPSL Trainer, but for unimodal models

import json
import os
import torch
from torch import nn
class UShapedPSL():

    def __init__(self):
        super()

    def _write_metrics_json_row(self, global_args, metrics):
        logs_dir = os.path.join(global_args['logs_dir'], "logs")
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

    def _get_correct_per_classifier(self, client_output, labels):
        correct_per_classifier = {}

        for key, logits in client_output.items():
            predictions = torch.argmax(logits, dim=1)
            correct_per_classifier[key] = (predictions == labels).sum().item()

        return correct_per_classifier

    # def _extract_features(self, server_model, smashed_data, global_args):
    #     """CLS + mean-pooled patch tokens, concatenated over several blocks."""
    #     n = global_args.get("feature_layers", 4)          # int, or list of block indices
    #     use_patch_mean = global_args.get("use_patch_mean_pooling", True)

    #     outputs = server_model.get_intermediate_layers(
    #         smashed_data, n=n, reshape=False, return_class_token=True, norm=True
    #     )

    #     feats = []
    #     for patch_tokens, cls_token in outputs:           # patch_tokens: (B, N, D)
    #         feats.append(cls_token)                       # (B, D)
    #         if use_patch_mean:
    #             feats.append(patch_tokens.mean(dim=1))    # (B, D)
    #     return torch.cat(feats, dim=-1)

    def _extract_features(self, server_model, smashed_data, global_args):
        """CLS + optionally mean-pooled patch tokens from the last N blocks.

        feature_layer_from_end = 1 -> last block
        feature_layer_from_end = 2 -> last 2 blocks
        feature_layer_from_end = 3 -> last 3 blocks
        etc.
        """
        n_layers = global_args.get("feature_layers", 1)
        use_patch_mean = global_args.get("use_patch_mean_pooling", True)

        n_blocks = self._num_blocks(server_model)

        if not 1 <= n_layers <= n_blocks:
            raise ValueError(
                f"feature_layers={n_layers} out of range; "
                f"model has {n_blocks} blocks"
            )

        # get_intermediate_layers expects absolute block indices
        start_idx = n_blocks - n_layers
        block_indices = list(range(start_idx, n_blocks))

        outputs = server_model.get_intermediate_layers(
            smashed_data,
            n=block_indices,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        feats = []

        for patch_tokens, cls_token in outputs:
            feats.append(cls_token)

            if use_patch_mean:
                feats.append(patch_tokens.mean(dim=1))

        return torch.cat(feats, dim=-1)


    @staticmethod
    def _num_blocks(model):
        n = getattr(model, "n_blocks", None)
        if n is not None:
            return n
        blocks = model.blocks
        # chunked ViTs store a list of BlockChunks instead of plain blocks
        last = blocks[-1]
        return len(last) if isinstance(last, (nn.ModuleList, nn.Sequential)) else len(blocks)

    def _single_iteration(
        self,
        device,
        server_model,
        server_optimizer,
        server_scheduler,
        clients,
        global_args
    ):

        loss_dict = {}

        server_model.train()

        update_server_every = global_args["server_batch_size"] // global_args["batch_size"]

        if server_optimizer is not None:
            server_optimizer.zero_grad()

        for client_id in range(global_args['nr_of_clients']):
            client = clients[client_id]
            X, y = next(client.train_iterator)
            X = X.to(device)
            y = y.to(device)

            client_head = client.head
            client_tail = client.tail

            client_tail.train()
            client.optimizer.zero_grad()
            smashed_data = client_head(X)
            smashed_data = smashed_data.detach()
            features = self._extract_features(server_model, smashed_data, global_args)
            predictions = client_tail(features)
            # server_features = server_model.get_intermediate_layers(smashed_data, n=1, reshape=False, return_class_token=True, norm=True)
            # cls_token = server_features[0][1]

            # predictions = client_tail(cls_token)
            loss = client.get_loss(predictions, y=y)
            loss = loss / update_server_every
            loss.backward()

            client.optimizer.step()
            client.scheduler.step()

            if server_optimizer is not None:
                if ((client_id + 1) % update_server_every == 0) or (
                    client_id + 1 == global_args['nr_of_clients']
                ):
                    server_optimizer.step()
                    server_scheduler.step()
                    server_optimizer.zero_grad()
          
            loss_dict[client_id] = loss.item()
        return loss_dict

    def _train_epoch(self, device, server_model, server_optimizer, server_scheduler, clients, global_args, epoch_nr):       
        for client in clients.values():
            client.head.eval()
            client.tail.train()

        if server_optimizer is not None:
            server_model.train()
        

        # use the TRAIN-level OFFICIAL_EPOCH_LENGTH (same unit used when building schedulers)
        train_official_epoch_length =  global_args["OFFICIAL_EPOCH_LENGTH"]

        window_loss_sum = 0.0
        window_iter_count = 0
        for iteration in range(global_args['OFFICIAL_EPOCH_LENGTH']):
            loss = self._single_iteration(
                device,
                server_model,
                server_optimizer,
                server_scheduler,
                clients,
                global_args)

            window_loss_sum += sum(loss.values())  # Average loss across clients for this iteration
            window_iter_count += 1

            total_epoch_length = global_args['OFFICIAL_EPOCH_LENGTH']
            if (iteration + 1) % 20 == 0 or iteration + 1 == total_epoch_length:
                remaining_iterations = total_epoch_length - iteration - 1
                avg_loss = window_loss_sum / window_iter_count
                print(f"Iteration {iteration + 1}/{total_epoch_length} ({remaining_iterations} remaining), average loss over last {window_iter_count} iterations: {avg_loss}")
                self._write_metrics_json_row(global_args, {
                    "epoch": epoch_nr,
                    "iteration": iteration + 1,
                    "remaining_iterations": remaining_iterations,
                    "window_iterations": window_iter_count,
                    "average_loss": avg_loss,
                })
                window_loss_sum = 0.0
                window_iter_count = 0


    def train(self, device, server_model, server_optimizer, server_scheduler, clients, evaluator, global_args):
        for epoch_nr in range(global_args['nr_of_epochs']):
            print(f'Starting epoch {epoch_nr + 1}')

            self._train_epoch(
                device=device,
                server_model=server_model,
                server_optimizer=server_optimizer,
                server_scheduler=server_scheduler,
                clients = clients,
                epoch_nr=epoch_nr + 1,
                global_args=global_args
            )

            if (epoch_nr + 1) % global_args["eval_epoch"] == 0:
                for client in clients.values():
                    client.test_iterator = iter(client.test_dataloader)

                self._test_epoch(
                    device=device,
                    server_model=server_model,
                    clients = clients,
                    epoch_nr=epoch_nr + 1,
                    evaluator = evaluator,
                    global_args=global_args
                )

    def _test_iteration(self, device, server_model, clients, global_args):
        client_output = {}
        labels_per_client = {}

        server_model.eval()

        for client_id in range(global_args['nr_of_clients']):
            try:
                X, y = next(clients[client_id].test_iterator)
            except StopIteration:
                continue

            X = X.to(device)
            if isinstance(y, torch.Tensor):
                y = y.to(device)

            client_head = clients[client_id].head
            client_tail = clients[client_id].tail

            client_head.eval()
            client_tail.eval()

            with torch.no_grad():

                smashed_data = client_head(X)

                # server_features = server_model.get_intermediate_layers(smashed_data, n=1, reshape=False, return_class_token=True, norm=True)

                # cls_token = server_features[0][1]
                features = self._extract_features(server_model, smashed_data, global_args)
                predictions = client_tail(features)
                # predictions = client_tail(cls_token)

            client_output[client_id] = predictions
            labels_per_client[client_id] = y

        return client_output, labels_per_client

    def _test_epoch(self, device, server_model, clients, epoch_nr, evaluator, global_args):
        server_model.eval()
        for client in clients.values():
            client.head.eval()
            client.tail.eval()

        evaluator.reset()

        max_batches = global_args.get("max_nr_of_batches_in_epoch", float("inf"))
        batch_count = 0
        while batch_count < max_batches:
            client_output, labels_per_client = self._test_iteration(
                device, server_model, clients, global_args
            )
            if not client_output:
                break
            for cid in client_output:
                evaluator.update(
                    client_output[cid], labels_per_client[cid], client_id=cid
                )
            batch_count += 1

        metrics = evaluator.compute()
        if metrics is None:
            print(f"Epoch {epoch_nr}: No test data available")
            return

        print(f"\nEpoch {epoch_nr} Test Results:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                for tk, tv in v.items():
                    print(f"  {k}/{tk}: {tv:.4f}")
            else:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        for client_id, accuracy in metrics.get("client_accuracies", {}).items():
            flat[f"client_{client_id}_accuracy"] = accuracy
        self._write_metrics_json_row(global_args, {"epoch": epoch_nr, "phase": "test", **flat})
