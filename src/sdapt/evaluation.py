"""Private linear probes on frozen DINO CLS features.

The caller supplies deterministic, non-augmented image transforms and the same
IID training partitions for every method. No test label selects a checkpoint or
hyperparameter. CheXpert tasks are independent binary problems: a k-shot task
uses up to k positives AND k negatives from its client's partition. Uncertain (-1)
and missing (NaN) labels provide no supervision and do not enter metrics.

Features are extracted once into temporary disk-backed arrays, including when
``run_evaluations`` evaluates several label budgets. Arrays are never reused
between invocations, so a cache cannot silently refer to another encoder.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class EvaluationConfig:
    """Explicit implementation defaults, not manuscript-reported hyperparameters.

    ``shots=None`` means full-data linear probing. ``feature_cache_dir`` is an
    optional parent directory for a fresh temporary cache, removed on return.
    A full run needs about 4 * feature_dim * (train_rows + test_rows) disk bytes.
    """

    shots: int | None = None
    epochs: int = 25
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 256
    seed: int = 2024
    num_workers: int = 0
    feature_cache_dir: str | None = None

    def __post_init__(self):
        if self.shots is not None and (type(self.shots) is not int or self.shots < 1):
            raise ValueError("shots must be a positive integer or None for full-data LP")
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _seed(seed: int, client_id: int, task_id: int = 0) -> int:
    return int(np.random.SeedSequence([seed, client_id, task_id]).generate_state(1)[0])


def _known_binary(targets):
    return np.isfinite(targets) & ((targets == 0) | (targets == 1))


def sample_client_indices(
    targets, partition, shots, seed, client_id, num_classes, multilabel=False,
):
    """Select dataset indices using only a client's training partition.

    Single-label output is one index array; multilabel output maps class index
    to its independent binary task's index array. Requested budgets are capped
    by available labels; returned metrics report actual counts for every class.
    Sorting candidates makes a partition's presentation order irrelevant. The
    seed excludes method and budget, so k-shot sets are nested across budgets.
    """
    targets = np.asarray(targets)
    partition = np.sort(np.asarray(partition, dtype=np.int64))
    if shots is not None and (shots < 1 or int(shots) != shots):
        raise ValueError("shots must be a positive integer or None")

    def select(candidates, task_id):
        if shots is None:
            return candidates
        rng = np.random.default_rng(_seed(seed, client_id, task_id))
        return rng.permutation(candidates)[:shots]

    if multilabel:
        result = {}
        for class_id in range(num_classes):
            labels = targets[partition, class_id]
            if shots is None:
                result[class_id] = partition[_known_binary(labels)]
            else:
                result[class_id] = np.concatenate([
                    select(partition[labels == value], 2 * class_id + value)
                    for value in (0, 1)
                ])
        return result
    if shots is None:
        values = targets[partition]
        known = np.isfinite(values) & (values >= 0) & (values < num_classes)
        return partition[known]
    return np.concatenate([
        select(partition[targets[partition] == class_id], class_id)
        for class_id in range(num_classes)
    ])


def binary_metrics(targets, scores):
    """Binary metrics with JSON null for undefined quantities and explicit counts."""
    targets = np.asarray(targets)
    scores = np.asarray(scores)
    if targets.ndim != 1 or targets.shape != scores.shape:
        raise ValueError("binary targets and scores must be matching one-dimensional arrays")
    known = _known_binary(targets)
    labels = targets[known].astype(np.int64)
    values = scores[known]
    if not np.all(np.isfinite(values)):
        raise ValueError("known test labels must have finite prediction scores")
    predictions = values >= 0.5
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    true_positives = int(np.sum(predictions & (labels == 1)))
    false_positives = int(np.sum(predictions & (labels == 0)))
    false_negatives = int(np.sum(~predictions & (labels == 1)))
    correct = int(np.sum(predictions == labels))
    denominator = 2 * true_positives + false_positives + false_negatives
    auc_reason = None if positives and negatives else (
        "no_known_test_labels" if len(labels) == 0 else "test_has_one_class"
    )
    return {
        "auroc": float(roc_auc_score(labels, values)) if auc_reason is None else None,
        "auroc_undefined_reason": auc_reason,
        "accuracy": correct / len(labels) if len(labels) else None,
        "f1": 2 * true_positives / denominator if denominator else None,
        "known_count": int(len(labels)),
        "ignored_count": int(len(targets) - len(labels)),
        "positive_count": positives,
        "negative_count": negatives,
        "correct_count": correct,
    }


def _multiclass_metrics(targets, predictions, classes):
    known = np.isfinite(targets) & (targets >= 0) & (targets < len(classes))
    labels = targets[known].astype(np.int64)
    predicted = predictions[known]
    correct = int(np.sum(labels == predicted))
    per_class = {}
    for class_id, name in enumerate(classes):
        support = int(np.sum(labels == class_id))
        predicted_count = int(np.sum(predicted == class_id))
        true_positives = int(np.sum((labels == class_id) & (predicted == class_id)))
        per_class[name] = {
            "test_support": support,
            "predicted_count": predicted_count,
            "correct_count": true_positives,
            "recall": true_positives / support if support else None,
            "precision": true_positives / predicted_count if predicted_count else None,
            "f1": 2 * true_positives / (support + predicted_count) if support + predicted_count else None,
        }
    return {
        "accuracy": correct / len(labels) if len(labels) else None,
        "correct_count": correct,
        "known_count": int(len(labels)),
        "ignored_count": int(len(targets) - len(labels)),
        "per_class": per_class,
    }


class _ImagesOnly(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])][0]


def _extract_features(encoder, dataset, indices, path, device, config):
    loader = DataLoader(
        _ImagesOnly(dataset, indices), batch_size=config.batch_size,
        shuffle=False, drop_last=False, num_workers=config.num_workers,
    )
    output = None
    offset = 0
    with torch.no_grad():
        for images in loader:
            features = encoder(images.to(device))
            if not isinstance(features, torch.Tensor) or features.ndim != 2 or len(features) != len(images):
                raise ValueError("encoder must return a [batch, feature_dim] CLS feature tensor")
            values = features.detach().float().cpu().numpy()
            if not np.all(np.isfinite(values)):
                raise ValueError("encoder returned nonfinite features")
            if output is None:
                output = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(len(indices), values.shape[1]),
                )
            if values.shape[1] != output.shape[1]:
                raise ValueError("encoder feature dimension changed between batches")
            output[offset:offset + len(values)] = values
            offset += len(values)
    if output is None:
        raise ValueError("cannot extract features from an empty dataset selection")
    output.flush()
    return output


def _fit_head(features, row_indices, targets, output_dim, binary, device, config, seed):
    # Isolate initialization from other methods and restore the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = nn.Linear(features.shape[1], output_dim)
        nn.init.normal_(head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(head.bias)
    head = head.to(device).train()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    final_loss = None
    for _ in range(config.epochs):
        permutation = rng.permutation(len(row_indices))
        loss_sum = 0.0
        for start in range(0, len(permutation), config.batch_size):
            positions = permutation[start:start + config.batch_size]
            images = torch.from_numpy(np.asarray(features[row_indices[positions]])).to(device)
            labels = torch.as_tensor(
                targets[positions], dtype=torch.float32 if binary else torch.long, device=device,
            )
            logits = head(images)
            loss = loss_fn(logits[:, 0] if binary else logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("linear-probe training produced a nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(positions)
        final_loss = loss_sum / len(row_indices)
    return head.eval(), final_loss


def _predict(head, features, device, batch_size, binary):
    predictions = np.empty(len(features), dtype=np.float32 if binary else np.int64)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            # Copy only this batch, avoiding writable-tensor warnings for read-only views.
            batch = torch.from_numpy(np.array(features[start:start + batch_size], copy=True)).to(device)
            logits = head(batch)
            values = torch.sigmoid(logits[:, 0]) if binary else logits.argmax(dim=1)
            predictions[start:start + len(values)] = values.cpu().numpy()
    return predictions


def _mean_defined(values):
    defined = [value for value in values if value is not None]
    return (float(np.mean(defined)) if defined else None), len(defined)


def _validate_inputs(train_dataset, test_dataset, partitions):
    classes = [str(name) for name in train_dataset.classes]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("dataset classes must be nonempty and uniquely named")
    if classes != [str(name) for name in test_dataset.classes]:
        raise ValueError("train and test class order must match")
    multilabel = bool(train_dataset.multilabel)
    if multilabel != bool(test_dataset.multilabel):
        raise ValueError("train and test must use the same label representation")
    for name, dataset in (("train", train_dataset), ("test", test_dataset)):
        targets = np.asarray(dataset.targets)
        expected = (len(dataset), len(classes)) if multilabel else (len(dataset),)
        if len(dataset) == 0 or targets.shape != expected:
            raise ValueError(f"{name} targets must have shape {expected} and contain samples")
        finite = targets[np.isfinite(targets)]
        if multilabel:
            if not np.all(np.isin(finite, [-1, 0, 1])):
                raise ValueError(f"{name} multilabel targets must be 0, 1, -1, or NaN")
        elif not np.all((finite == np.floor(finite)) & (finite >= -1) & (finite < len(classes))):
            raise ValueError(f"{name} class labels must be integers in range, -1, or NaN")
        if np.any(np.isinf(targets)):
            raise ValueError(f"{name} labels cannot be infinite")
    normalized = []
    for partition in partitions:
        values = np.asarray(partition)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("client partitions must be nonempty one-dimensional index arrays")
        if not np.all((values == np.floor(values)) & (values >= 0) & (values < len(train_dataset))):
            raise ValueError("client partition contains invalid training indices")
        normalized.append(values.astype(np.int64))
    if not normalized:
        raise ValueError("at least one client partition is required")
    all_indices = np.concatenate(normalized)
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("client partitions overlap or contain duplicate indices")
    if len(all_indices) != len(train_dataset):
        raise ValueError("client partitions must cover the training dataset exactly once")
    return classes, multilabel, normalized


def _evaluate_cached(train_features, test_features, row_lookup, train_targets,
                     test_targets, selections, classes, multilabel, device, config,
                     client_ids=None):
    client_metrics, heads, sampled_indices = [], [], []
    if client_ids is None:
        client_ids = range(len(selections))
    for client_id, selection in zip(client_ids, selections):
        client_seed = _seed(config.seed, client_id)
        if multilabel:
            per_class, class_heads, class_indices = {}, {}, {}
            for class_id, name in enumerate(classes):
                indices = selection[class_id]
                labels = train_targets[indices, class_id].astype(np.int64)
                positives = int(np.sum(labels == 1))
                negatives = int(np.sum(labels == 0))
                class_indices[name] = indices.tolist()
                if positives and negatives:
                    head, loss = _fit_head(
                        train_features, row_lookup[indices], labels, 1, True, device,
                        config, _seed(config.seed, client_id, class_id),
                    )
                    scores = _predict(head, test_features, device, config.batch_size, True)
                    metrics = binary_metrics(test_targets[:, class_id], scores)
                    class_heads[name] = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                    metrics.update(trained=True, final_training_loss=loss)
                    del head
                else:
                    known = _known_binary(test_targets[:, class_id])
                    metrics = {
                        "trained": False, "final_training_loss": None,
                        "auroc": None, "auroc_undefined_reason": "training_has_fewer_than_two_classes",
                        "accuracy": None, "f1": None, "correct_count": None,
                        "known_count": int(known.sum()), "ignored_count": int((~known).sum()),
                        "positive_count": int(np.sum(test_targets[:, class_id] == 1)),
                        "negative_count": int(np.sum(test_targets[:, class_id] == 0)),
                    }
                    class_heads[name] = None
                metrics.update(training_positive_count=positives, training_negative_count=negatives,
                               training_sample_count=int(len(indices)))
                per_class[name] = metrics
            mean_auc, count_auc = _mean_defined([value["auroc"] for value in per_class.values()])
            client_metrics.append({
                "client_id": client_id, "seed": client_seed, "auroc": mean_auc,
                "defined_auroc_count": count_auc, "total_class_count": len(classes),
                "training_sample_count": len(np.unique(np.concatenate(list(selection.values())))),
                "per_class": per_class,
            })
            heads.append({"client_id": client_id, "type": "independent_binary_linear", "classes": class_heads})
            sampled_indices.append({"client_id": client_id, "per_class": class_indices})
        else:
            if not len(selection):
                raise ValueError(f"client {client_id} has no known training labels")
            labels = train_targets[selection].astype(np.int64)
            head, loss = _fit_head(
                train_features, row_lookup[selection], labels, len(classes), False,
                device, config, client_seed,
            )
            predictions = _predict(head, test_features, device, config.batch_size, False)
            metrics = _multiclass_metrics(test_targets, predictions, classes)
            for class_id, name in enumerate(classes):
                metrics["per_class"][name]["training_sample_count"] = int(np.sum(labels == class_id))
            metrics.update(client_id=client_id, seed=client_seed,
                           training_sample_count=int(len(selection)), final_training_loss=loss)
            client_metrics.append(metrics)
            heads.append({"client_id": client_id, "type": "multiclass_linear",
                          "state_dict": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}})
            sampled_indices.append({"client_id": client_id, "indices": selection.tolist()})
            del head

    return _assemble_result(client_metrics, heads, sampled_indices, classes, multilabel,
                            config, len(test_targets), train_features.shape[1])


def _assemble_result(client_metrics, heads, sampled_indices, classes, multilabel,
                     config, test_sample_count, feature_dim):
    if multilabel:
        all_tasks = [client["per_class"][name] for client in client_metrics for name in classes]
        mean_auc, count_auc = _mean_defined([task["auroc"] for task in all_tasks])
        scored = [task for task in all_tasks if task["trained"]]
        known_count = sum(task["known_count"] for task in scored)
        correct_count = sum(task["correct_count"] for task in scored)
        global_metrics = {
            "auroc": mean_auc, "defined_auroc_count": count_auc,
            "total_client_class_count": len(all_tasks),
            "accuracy": correct_count / known_count if known_count else None,
            "prediction_count": known_count, "correct_count": correct_count,
            "aggregation": "AUROC is the unweighted mean of defined client-class AUROCs; accuracy pools known binary predictions",
        }
        aggregate_classes = {}
        for name in classes:
            values = [client["per_class"][name]["auroc"] for client in client_metrics]
            mean, count = _mean_defined(values)
            aggregate_classes[name] = {"auroc": mean, "defined_auroc_count": count,
                                       "total_client_count": len(client_metrics)}
    else:
        count = sum(client["known_count"] for client in client_metrics)
        correct = sum(client["correct_count"] for client in client_metrics)
        global_metrics = {
            "accuracy": correct / count if count else None,
            "prediction_count": count, "correct_count": correct,
            "aggregation": "accuracy pools predictions of private client heads on the same held-out test set",
        }
        aggregate_classes = {}
        for name in classes:
            values = [client["per_class"][name] for client in client_metrics]
            support = sum(value["test_support"] for value in values)
            correct = sum(value["correct_count"] for value in values)
            aggregate_classes[name] = {"recall": correct / support if support else None,
                                       "test_prediction_count": support, "correct_count": correct}
    metrics = {
        "task_type": "multilabel_binary" if multilabel else "multiclass",
        "protocol": "full_data_linear_probe" if config.shots is None else f"{config.shots}_shot_linear_probe",
        "requested_shots": config.shots,
        "sampling_policy": "all_known_labels" if config.shots is None else "per_class_budget_capped_by_available_labels",
        "classes": classes, "client_count": len(client_metrics),
        "test_sample_count": test_sample_count, "feature_dim": int(feature_dim),
        "encoder_frozen": True, "global": global_metrics,
        "per_class": aggregate_classes, "clients": client_metrics,
    }
    serialized_config = asdict(config)
    if serialized_config["feature_cache_dir"] is not None:
        serialized_config["feature_cache_dir"] = str(serialized_config["feature_cache_dir"])
    return {"metrics": metrics, "heads": heads, "config": serialized_config, "sampled_indices": sampled_indices}


def _freeze_encoder(encoder, device):
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _evaluate_feature_cache(encoder, train_dataset, test_dataset, all_selections,
                            classes, multilabel, device, configs, client_ids=None):
    """One cache lifetime for either the shared encoder or one private encoder."""
    train_targets, test_targets = np.asarray(train_dataset.targets), np.asarray(test_dataset.targets)
    selected_arrays = [indices for selections in all_selections for selection in selections
                       for indices in (selection.values() if multilabel else [selection])]
    selected_indices = np.unique(np.concatenate(selected_arrays))
    row_lookup = np.full(len(train_dataset), -1, dtype=np.int64)
    row_lookup[selected_indices] = np.arange(len(selected_indices))
    _freeze_encoder(encoder, device)
    if not len(selected_indices):
        if not multilabel:
            raise ValueError("no known training labels are available")
        # No binary task can be trained, so neither image split needs encoding.
        # Zero rows supply only the interface's feature dimension to the report.
        empty_features = np.empty((0, encoder.num_features), dtype=np.float32)
        return [
            _evaluate_cached(empty_features, empty_features, row_lookup, train_targets,
                             test_targets, selections, classes, multilabel, device, config,
                             client_ids=client_ids)
            for config, selections in zip(configs, all_selections)
        ]
    cache_parent = configs[0].feature_cache_dir
    if cache_parent is not None:
        Path(cache_parent).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sdapt-evaluation-", dir=cache_parent) as directory:
        train_features = test_features = None
        try:
            train_features = _extract_features(
                encoder, train_dataset, selected_indices, Path(directory) / "train.npy", device, configs[0],
            )
            test_features = _extract_features(
                encoder, test_dataset, np.arange(len(test_dataset)), Path(directory) / "test.npy", device, configs[0],
            )
            if train_features.shape[1] != test_features.shape[1]:
                raise ValueError("train and test encoder feature dimensions differ")
            return [
                _evaluate_cached(train_features, test_features, row_lookup, train_targets,
                                 test_targets, selections, classes, multilabel, device, config,
                                 client_ids=client_ids)
                for config, selections in zip(configs, all_selections)
            ]
        finally:
            for features in (train_features, test_features):
                if features is not None:
                    features._mmap.close()


def run_evaluations(encoder, train_dataset, test_dataset, partitions, device,
                    configs: Sequence[EvaluationConfig],
                    encoder_for_client: Callable[[int], nn.Module] | None = None):
    """Evaluate several label budgets with one frozen feature extraction pass.

    Return one result per config, containing JSON-safe metrics, exact sampled
    training indices, and CPU head state dictionaries suitable for torch.save.
    The encoder is deliberately left frozen and in evaluation mode. Its caller
    must provide the adapted EMA teacher or the official no-DAPT backbone.
    For a method with private client frontends, ``encoder_for_client(client_id)``
    provides each client's complete encoder. Calls are sequential and may mutate
    and return the same module; each client's features are cached independently
    across budgets and released before requesting the next client encoder.
    """
    configs = list(configs)
    if not configs:
        raise ValueError("at least one evaluation config is required")
    classes, multilabel, partitions = _validate_inputs(train_dataset, test_dataset, partitions)
    train_targets = np.asarray(train_dataset.targets)
    all_selections = [
        [sample_client_indices(train_targets, partition, config.shots, config.seed,
                               client_id, len(classes), multilabel)
         for client_id, partition in enumerate(partitions)]
        for config in configs
    ]
    device = torch.device(device)
    if encoder_for_client is None:
        return _evaluate_feature_cache(encoder, train_dataset, test_dataset, all_selections,
                                       classes, multilabel, device, configs)
    per_budget = [[] for _ in configs]
    for client_id in range(len(partitions)):
        client_encoder = encoder_for_client(client_id)
        client_selections = [[selections[client_id]] for selections in all_selections]
        results = _evaluate_feature_cache(
            client_encoder, train_dataset, test_dataset, client_selections,
            classes, multilabel, device, configs, client_ids=[client_id],
        )
        for partials, result in zip(per_budget, results):
            partials.append(result)
    combined = []
    for config, partials in zip(configs, per_budget):
        feature_dims = {partial["metrics"]["feature_dim"] for partial in partials}
        if len(feature_dims) != 1:
            raise ValueError("client encoders must expose the same feature dimension")
        combined.append(_assemble_result(
            [client for partial in partials for client in partial["metrics"]["clients"]],
            [head for partial in partials for head in partial["heads"]],
            [indices for partial in partials for indices in partial["sampled_indices"]],
            classes, multilabel, config, len(test_dataset), feature_dims.pop(),
        ))
    return combined


def run_evaluation(encoder, train_dataset, test_dataset, partitions, device,
                   config: EvaluationConfig | None = None,
                   encoder_for_client: Callable[[int], nn.Module] | None = None):
    """Single-budget convenience wrapper; prefer run_evaluations for a sweep."""
    return run_evaluations(
        encoder, train_dataset, test_dataset, partitions, device,
        [EvaluationConfig() if config is None else config],
        encoder_for_client=encoder_for_client,
    )[0]
