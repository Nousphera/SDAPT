"""Downstream protocol tests use tiny local features, never pretrained downloads."""

import json
import io

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from sdapt.evaluation import (
    EvaluationConfig,
    binary_metrics,
    run_evaluation,
    run_evaluations,
    sample_client_indices,
)


def test_shots_are_per_class_local_reproducible_and_nested():
    targets = np.repeat(np.arange(3), 30)
    partition = np.concatenate([np.arange(5, 25) + 30 * c for c in range(3)])
    selected = sample_client_indices(targets, partition, 4, 2024, 2, 3)
    repeated = sample_client_indices(targets, partition, 4, 2024, 2, 3)
    other_client = sample_client_indices(targets, partition, 4, 2024, 3, 3)
    eight = sample_client_indices(targets, partition, 8, 2024, 2, 3)
    assert np.bincount(targets[selected]).tolist() == [4, 4, 4]
    assert set(selected) <= set(partition)
    assert np.array_equal(selected, repeated)
    assert not np.array_equal(selected, other_client)
    assert set(selected) <= set(eight)


def test_multilabel_shots_balance_each_task_and_ignore_uncertainty():
    targets = np.array([[0, 1], [1, -1], [0, 0], [1, 0], [-1, 1], [np.nan, 1]])
    selected = sample_client_indices(targets, np.arange(6), 1, 17, 0, 2, multilabel=True)
    assert set(selected) == {0, 1}
    assert sorted(targets[selected[0], 0].tolist()) == [0, 1]
    assert sorted(targets[selected[1], 1].tolist()) == [0, 1]
    full = sample_client_indices(targets, np.arange(6), None, 17, 0, 2, multilabel=True)
    assert full[0].tolist() == [0, 1, 2, 3]
    assert full[1].tolist() == [0, 2, 3, 4, 5]


def test_shot_budget_is_capped_by_each_clients_available_class_labels():
    targets = np.array([0, 0, 1])
    selected = sample_client_indices(targets, np.arange(3), 2, 1, 2, 3)
    assert sorted(selected.tolist()) == [0, 1, 2]
    assert np.bincount(targets[selected], minlength=3).tolist() == [2, 1, 0]


def test_missing_binary_class_is_untrained_and_counts_remain_visible(tmp_path):
    train = FeatureImages([[-2, 0], [2, 0]], [[0, -1], [1, -1]], ["a", "b"], True)
    test = FeatureImages([[-2, 0], [2, 0]], [[0, 0], [1, 1]], train.classes, True)
    result = run_evaluation(
        ObservedEncoder(), train, test, [[0, 1]], "cpu",
        EvaluationConfig(shots=16, epochs=1, feature_cache_dir=str(tmp_path)),
    )
    metrics = result["metrics"]
    assert metrics["requested_shots"] == 16
    assert metrics["clients"][0]["per_class"]["a"]["training_positive_count"] == 1
    assert metrics["clients"][0]["per_class"]["a"]["training_negative_count"] == 1
    missing = metrics["clients"][0]["per_class"]["b"]
    assert missing["training_sample_count"] == 0
    assert not missing["trained"]
    assert missing["auroc"] is None


def test_binary_metrics_mask_uncertainty_and_report_undefined_auc():
    metrics = binary_metrics(np.array([0, 1, -1, np.nan]), np.array([0.1, 0.9, 1.0, 0.0]))
    assert metrics["auroc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["known_count"] == 2
    assert metrics["ignored_count"] == 2
    undefined = binary_metrics(np.array([1, 1, -1]), np.array([0.4, 0.8, 0.0]))
    assert undefined["auroc"] is None
    assert undefined["auroc_undefined_reason"] == "test_has_one_class"
    assert undefined["positive_count"] == 2
    assert undefined["negative_count"] == 0
    empty = binary_metrics(np.array([-1, np.nan]), np.array([0.4, 0.8]))
    assert empty["accuracy"] is None
    assert empty["auroc"] is None
    json.dumps([metrics, undefined, empty], allow_nan=False)


class FeatureImages(Dataset):
    def __init__(self, images, labels, classes, multilabel=False):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.targets = np.asarray(labels)
        self.classes = classes
        self.multilabel = multilabel

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class ObservedEncoder(nn.Module):
    """Fixture encoder whose batch-norm state exposes accidental train mode."""

    num_features = 2

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(2)
        self.observations = []

    def forward(self, images):
        self.observations.append((self.training, torch.is_grad_enabled(), len(images)))
        return self.norm(images)


def test_frozen_encoder_private_heads_and_disk_cache_end_to_end(tmp_path):
    train = FeatureImages(
        [[-2, 0], [-3, 0], [2, 0], [3, 0]] * 2,
        [0, 0, 1, 1] * 2,
        ["left", "right"],
    )
    test = FeatureImages([[-2, 0], [2, 0]], [0, 1], train.classes)
    encoder = ObservedEncoder().train()
    original = {key: value.clone() for key, value in encoder.state_dict().items()}
    result = run_evaluation(
        encoder, train, test, [np.arange(4), np.arange(4, 8)], "cpu",
        EvaluationConfig(shots=1, epochs=20, learning_rate=0.1, batch_size=2,
                         feature_cache_dir=str(tmp_path)),
    )
    assert result["metrics"]["global"]["accuracy"] == 1.0
    assert result["metrics"]["global"]["prediction_count"] == 4
    assert result["metrics"]["test_sample_count"] == 2
    assert len(result["heads"]) == 2
    assert all(not training and not grad for training, grad, _ in encoder.observations)
    assert sum(batch for _, _, batch in encoder.observations) == 6
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    assert all(torch.equal(original[key], value) for key, value in encoder.state_dict().items())
    assert list(tmp_path.iterdir()) == []
    assert all(item["training_sample_count"] == 2 for item in result["metrics"]["clients"])
    assert result["metrics"]["clients"][0]["per_class"]["left"]["recall"] == 1.0
    json.dumps(result["metrics"], allow_nan=False)


def test_multilabel_end_to_end_averages_only_defined_client_task_auc(tmp_path):
    train = FeatureImages(
        [[-2, -1], [-3, 1], [2, -1], [3, 1], [0, 0]],
        [[0, 0], [0, 1], [1, 0], [1, 1], [-1, -1]],
        ["pathology_a", "pathology_b"], True,
    )
    test = FeatureImages(
        [[-3, -1], [3, -1], [0, 0]],
        [[0, 1], [1, 1], [-1, -1]], train.classes, True,
    )
    result = run_evaluation(
        ObservedEncoder(), train, test, [np.arange(5)], "cpu",
        EvaluationConfig(shots=None, epochs=20, learning_rate=0.1, batch_size=4,
                         feature_cache_dir=str(tmp_path)),
    )
    metrics = result["metrics"]
    assert metrics["global"]["auroc"] == 1.0
    assert metrics["global"]["defined_auroc_count"] == 1
    assert metrics["global"]["total_client_class_count"] == 2
    assert metrics["per_class"]["pathology_b"]["auroc"] is None
    assert metrics["clients"][0]["per_class"]["pathology_a"]["training_positive_count"] == 2
    assert metrics["clients"][0]["per_class"]["pathology_a"]["training_negative_count"] == 2
    assert metrics["clients"][0]["per_class"]["pathology_b"]["ignored_count"] == 1
    json.dumps(metrics, allow_nan=False)


def test_overlapping_client_partitions_are_rejected_before_training():
    data = FeatureImages([[-1, 0], [1, 0]], [0, 1], ["a", "b"])
    encoder = ObservedEncoder()
    with pytest.raises(ValueError, match="overlap"):
        run_evaluation(encoder, data, data, [[0, 1], [1]], "cpu", EvaluationConfig(epochs=1))
    assert encoder.observations == []


def test_multiple_label_budgets_share_one_encoder_extraction(tmp_path):
    train = FeatureImages([[-2, 0], [-3, 0], [2, 0], [3, 0]], [0, 0, 1, 1], ["a", "b"])
    test = FeatureImages([[-2, 0], [2, 0]], [0, 1], train.classes)
    encoder = ObservedEncoder()
    configs = [EvaluationConfig(shots=shots, epochs=1, batch_size=2, feature_cache_dir=str(tmp_path))
               for shots in (1, None)]
    results = run_evaluations(encoder, train, test, [np.arange(4)], "cpu", configs)
    assert sum(count for _, _, count in encoder.observations) == 6
    assert [result["metrics"]["clients"][0]["training_sample_count"] for result in results] == [2, 4]
    assert list(tmp_path.iterdir()) == []


def test_client_specific_encoder_callback_uses_each_clients_features(tmp_path):
    train = FeatureImages([[-2, 0], [2, 0]] * 2, [0, 1] * 2, ["a", "b"])
    test = FeatureImages([[-2, 0], [2, 0]], [0, 1], train.classes)
    encoder = ObservedEncoder()
    selected_clients = []

    def encoder_for_client(client_id):
        selected_clients.append(client_id)
        with torch.no_grad():
            encoder.norm.weight.fill_(1.0 if client_id == 0 else -1.0)
        return encoder

    result = run_evaluation(
        encoder, train, test, [[0, 1], [2, 3]], "cpu",
        EvaluationConfig(shots=1, epochs=8, learning_rate=0.1, batch_size=2,
                         feature_cache_dir=str(tmp_path)),
        encoder_for_client=encoder_for_client,
    )
    assert selected_clients == [0, 1]
    assert sum(count for _, _, count in encoder.observations) == 8
    assert result["heads"][0]["state_dict"]["weight"][1, 0] > 0
    assert result["heads"][1]["state_dict"]["weight"][1, 0] < 0
    assert result["metrics"]["global"]["accuracy"] == 1.0
    assert result["metrics"]["global"]["prediction_count"] == 4
    assert [client["client_id"] for client in result["metrics"]["clients"]] == [0, 1]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ["shots", "epochs", "batch_size", "seed", "num_workers"])
def test_integer_config_fields_reject_float_values_before_execution(field):
    with pytest.raises(ValueError, match=field):
        EvaluationConfig(**{field: 1.0})


def test_full_lp_marks_tasks_without_training_negatives_as_untrained(tmp_path):
    train = FeatureImages([[-2, 0], [2, 0]], [[0, 1], [1, 1]], ["a", "b"], True)
    test = FeatureImages([[-2, 0], [2, 0]], [[0, 0], [1, 1]], train.classes, True)
    result = run_evaluation(
        ObservedEncoder(), train, test, [[0, 1]], "cpu",
        EvaluationConfig(epochs=1, feature_cache_dir=str(tmp_path)),
    )
    metrics = result["metrics"]["clients"][0]["per_class"]["b"]
    assert result["heads"][0]["classes"]["b"] is None
    assert not metrics["trained"]
    assert metrics["auroc"] is None
    assert metrics["auroc_undefined_reason"] == "training_has_fewer_than_two_classes"
    assert result["metrics"]["global"]["defined_auroc_count"] == 1


def test_all_unknown_client_remains_untrained_with_private_encoder_and_safe_checkpoint(tmp_path):
    train = FeatureImages([[-2, 0], [2, 0]], [[-1], [1]], ["a"], True)
    test = FeatureImages([[-2, 0], [2, 0]], [[0], [1]], ["a"], True)
    result = run_evaluation(
        ObservedEncoder(), train, test, [[0], [1]], "cpu",
        EvaluationConfig(shots=4, epochs=1, feature_cache_dir=tmp_path),
        encoder_for_client=lambda _: ObservedEncoder(),
    )
    missing = result["metrics"]["clients"][0]["per_class"]["a"]
    assert not missing["trained"]
    assert missing["training_sample_count"] == 0
    assert result["heads"][0]["classes"]["a"] is None
    assert result["metrics"]["global"]["auroc"] is None
    checkpoint = io.BytesIO()
    torch.save({"heads": result["heads"], "config": result["config"]}, checkpoint)
    checkpoint.seek(0)
    loaded = torch.load(checkpoint, weights_only=True)
    assert loaded["config"]["feature_cache_dir"] == str(tmp_path)
