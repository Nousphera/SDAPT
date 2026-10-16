import json
import hashlib

import numpy as np
import pytest
from PIL import Image

from sdapt.data import CHEXPERT_LABELS, iid_partition, load_dataset


def manifest(tmp_path, name, target, classes, multilabel=False, path="image.png"):
    folder = tmp_path / name
    folder.mkdir()
    Image.new("RGB", (12, 10)).save(folder / "image.png")
    (folder / "metadata.json").write_text(json.dumps({"classes": classes, "multilabel": multilabel}))
    (folder / "train.jsonl").write_text(json.dumps({"path": path, "target": target}) + "\n")
    return folder


def test_loads_rgb_and_preserves_unknown_labels(tmp_path):
    manifest(tmp_path, "chexpert", [1, -1, 0, -1, 0, 1, 0, 0], list(CHEXPERT_LABELS), True)
    data = load_dataset("chexpert", tmp_path, "train")
    image, labels = data[0]
    assert image.mode == "RGB"
    assert labels.tolist() == [1, -1, 0, -1, 0, 1, 0, 0]
    assert data.targets.dtype == np.float32
    assert data.multilabel


def test_rejects_path_escape_and_invalid_targets(tmp_path):
    folder = manifest(tmp_path, "fmow", 0, ["airport"], path="../outside.png")
    with pytest.raises(ValueError, match="path"):
        load_dataset("fmow", tmp_path, "train")
    (folder / "train.jsonl").write_text('{"path":"image.png","target":2}\n')
    with pytest.raises(ValueError, match="target"):
        load_dataset("fmow", tmp_path, "train")


def test_missing_images_are_not_silently_skipped(tmp_path):
    manifest(tmp_path, "fmow", 0, ["airport"], path="missing.png")
    with pytest.raises(FileNotFoundError, match="missing.png"):
        load_dataset("fmow", tmp_path, "train")


def test_iid_partition_is_deterministic_exhaustive_and_disjoint():
    parts = iid_partition(23, 4, seed=7)
    assert [len(p) for p in parts] == [6, 6, 6, 5]
    assert sorted(np.concatenate(parts).tolist()) == list(range(23))
    assert all(np.array_equal(a, b) for a, b in zip(parts, iid_partition(23, 4, seed=7)))
    assert not np.array_equal(parts[0], iid_partition(23, 4, seed=8)[0])
    with pytest.raises(ValueError):
        iid_partition(3, 4)


def test_preparation_receipt_detects_changed_manifest(tmp_path):
    folder = manifest(tmp_path, "fmow", 0, ["airport"])
    metadata = json.loads((folder / "metadata.json").read_text())
    metadata["manifest_sha256"] = {"train": hashlib.sha256((folder / "train.jsonl").read_bytes()).hexdigest()}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    (folder / "train.jsonl").write_text('{"path":"image.png","target":0}\n\n')
    with pytest.raises(ValueError, match="checksum"):
        load_dataset("fmow", tmp_path, "train")


def test_manifest_receipt_count_must_match_rows(tmp_path):
    folder = manifest(tmp_path, "fmow", 0, ["airport"])
    metadata = json.loads((folder / "metadata.json").read_text())
    metadata["counts"] = {"train": 2}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="count mismatch"):
        load_dataset("fmow", tmp_path, "train")
