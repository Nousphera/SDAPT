import torch

from evaluators.evaluator import Evaluator


class AccuracyEvaluator(Evaluator):
    def reset(self):
        self._preds, self._labels = [], []
        self._client_correct = {}
        self._client_total = {}

    def update(self, logits, labels, client_id=None):
        preds = torch.argmax(logits, dim=1).cpu()
        labels = labels.cpu()
        self._preds.append(preds)
        self._labels.append(labels)

        if client_id is not None:
            self._client_correct[client_id] = (
                self._client_correct.get(client_id, 0) + (preds == labels).sum().item()
            )
            self._client_total[client_id] = (
                self._client_total.get(client_id, 0) + labels.numel()
            )

    def compute(self):
        if not self._preds:
            return None
        preds = torch.cat(self._preds)
        labels = torch.cat(self._labels)

        overall = (preds == labels).float().mean().item()
        num_classes = int(labels.max().item()) + 1
        per_class = {}
        for c in range(num_classes):
            m = labels == c
            per_class[c] = (preds[m] == labels[m]).float().mean().item() if m.any() else 0.0

        worst = min(per_class, key=per_class.get)
        metrics = {
            "overall_accuracy": overall,
            "avg_class_accuracy": sum(per_class.values()) / len(per_class),
            "worst_class": worst,
            "worst_class_accuracy": per_class[worst],
        }

        if self._client_total:
            client_accuracies = {
                client_id: self._client_correct[client_id] / self._client_total[client_id]
                for client_id in self._client_total
            }
            client_accuracy_tensor = torch.tensor(list(client_accuracies.values()))
            metrics.update({
                "client_accuracies": client_accuracies,
                "mean_client_accuracy": client_accuracy_tensor.mean().item(),
                "std_client_accuracy": client_accuracy_tensor.std(unbiased=False).item(),
            })

        return metrics