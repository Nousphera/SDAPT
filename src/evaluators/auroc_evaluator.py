import torch
import math                                    # math.nan, math.isnan
import numpy as np                             # np.unique, array indexing/masking
import torch                                   # torch.sigmoid, torch.cat
from sklearn.metrics import roc_auc_score      # the AUROC itself
from evaluators.evaluator import Evaluator


class MultilabelAUROCEvaluator(Evaluator):
    def __init__(self, task_names, mean_over=None, ignore_index=-1):
        # task_names: full ordered list of task columns (e.g. CHEXPERT_PATHOLOGY_TASKS)
        # mean_over:  subset to average for the headline number (e.g. CHEXPERT_COMPETITION_TASKS)
        self.task_names = list(task_names)
        self.mean_over = list(mean_over) if mean_over else self.task_names
        self.ignore_index = ignore_index

    def reset(self):
        self._logits, self._labels = [], []
        self._client_logits, self._client_labels = {}, {}

    def update(self, logits, labels, client_id=None):
        logits = logits.detach().float().cpu()
        labels = labels.detach().float().cpu()
        self._logits.append(logits)
        self._labels.append(labels)

        if client_id is not None:
            self._client_logits.setdefault(client_id, []).append(logits)
            self._client_labels.setdefault(client_id, []).append(labels)

    def compute(self):
        if not self._logits:
            return None
        probs = torch.sigmoid(torch.cat(self._logits)).numpy()
        labels = torch.cat(self._labels).numpy()

        per_task = {}
        for i, name in enumerate(self.task_names):
            y, s = labels[:, i], probs[:, i]
            mask = y != self.ignore_index           # drop uncertain/ignored
            y, s = y[mask], s[mask]
            # roc_auc_score needs both classes present in the eval subset
            per_task[name] = roc_auc_score(y, s) if len(np.unique(y)) == 2 else math.nan

        vals = [per_task[t] for t in self.mean_over if not math.isnan(per_task[t])]
        std_per_task = {}
        client_mean_aurocs = {}
        for i, name in enumerate(self.task_names):
            client_aurocs = []
            for client_id in self._client_logits:
                client_probs = torch.sigmoid(torch.cat(self._client_logits[client_id])).numpy()
                client_labels = torch.cat(self._client_labels[client_id]).numpy()
                y = client_labels[:, i]
                s = client_probs[:, i]
                mask = y != self.ignore_index
                y = y[mask]
                s = s[mask]
                if len(np.unique(y)) == 2:
                    client_aurocs.append(roc_auc_score(y, s))

            std_per_task[name] = (
                torch.tensor(client_aurocs, dtype=torch.float32).std(unbiased=False).item()
                if client_aurocs else math.nan
            )

        for client_id in self._client_logits:
            client_probs = torch.sigmoid(torch.cat(self._client_logits[client_id])).numpy()
            client_labels = torch.cat(self._client_labels[client_id]).numpy()
            client_task_aurocs = []

            for task_name in self.mean_over:
                task_index = self.task_names.index(task_name)
                y = client_labels[:, task_index]
                s = client_probs[:, task_index]
                mask = y != self.ignore_index
                y, s = y[mask], s[mask]
                if len(np.unique(y)) == 2:
                    client_task_aurocs.append(roc_auc_score(y, s))

            if client_task_aurocs:
                client_mean_aurocs[client_id] = sum(client_task_aurocs) / len(client_task_aurocs)

        client_mean_values = torch.tensor(list(client_mean_aurocs.values()), dtype=torch.float32)

        return {
            "mean_auroc": sum(vals) / len(vals) if vals else math.nan,
            "per_task_auroc": per_task,
            "std_per_task_auroc": std_per_task,
            "client_mean_aurocs": client_mean_aurocs,
            "mean_client_auroc": client_mean_values.mean().item() if len(client_mean_values) else math.nan,
            "std_client_auroc": client_mean_values.std(unbiased=False).item() if len(client_mean_values) else math.nan,
        }