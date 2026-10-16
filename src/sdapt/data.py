"""Validated manifests shared by adaptation and downstream evaluation."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DATASETS = ("fmow", "quickdraw", "chexpert")
CHEXPERT_LABELS = (
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Edema",
    "Consolidation", "Atelectasis", "Pleural Effusion", "Support Devices",
)
FMOW_CLASSES = (
    "airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture",
    "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership",
    "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution",
    "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital",
    "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility",
    "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility",
    "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison",
    "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard",
    "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility",
    "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower",
    "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo",
)


def image_path(root: Path, relative: str) -> Path:
    """Permit only relative image paths inside the explicitly declared image root."""
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"Unsafe image path: {relative!r}")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes its root: {relative!r}")
    return resolved


def validate_target(target, classes, multilabel):
    if multilabel:
        if not isinstance(target, list) or len(target) != len(classes):
            raise ValueError("Multilabel target must have one value per class")
        if any(isinstance(v, bool) or v not in (-1, 0, 1) for v in target):
            raise ValueError("Multilabel target values must be -1 (unknown), 0 or 1")
    elif isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(classes):
        raise ValueError(f"Invalid classification target: {target!r}")


class ManifestDataset(Dataset):
    """One image per line; missing and uncertain radiograph targets remain -1."""

    def __init__(self, name, root, split, transform=None):
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.name, self.split, self.transform = name, split, transform
        self.root = Path(root).expanduser().resolve() / name
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Dataset is not prepared: {metadata_path}. Run python scripts/prepare_data.py --dataset {name}")
        metadata = json.loads(metadata_path.read_text())
        self.classes = metadata["classes"]
        self.multilabel = metadata["multilabel"]
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be a nonempty list of unique names")
        if name == "chexpert" and (self.classes != list(CHEXPERT_LABELS) or not self.multilabel):
            raise ValueError("CheXpert requires the eight paper labels in their documented order")
        declared_root = Path(metadata.get("image_roots", {}).get(split, ".")).expanduser()
        self.image_root = (self.root / declared_root).resolve()
        self.records, self.paths = [], []
        seen = set()
        manifest_path = self.root / f"{split}.jsonl"
        expected_hash = metadata.get("manifest_sha256", {}).get(split)
        if expected_hash:
            digest = hashlib.sha256()
            with manifest_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"Manifest checksum mismatch: {manifest_path}; rerun preparation with its source directory")
        with manifest_path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    path = image_path(self.image_root, record["path"])
                    validate_target(record["target"], self.classes, self.multilabel)
                    if path in seen:
                        raise ValueError(f"duplicate image path: {record['path']}")
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing image: {path}")
                    seen.add(path)
                    self.records.append(record)
                    self.paths.append(path)
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid manifest line {number}: {error}") from error
        if not self.records:
            raise ValueError(f"Empty {split} manifest for {name}")
        expected_count = metadata.get("counts", {}).get(split)
        if expected_count is not None and len(self.records) != expected_count:
            raise ValueError(f"Manifest count mismatch: expected {expected_count}, read {len(self.records)}")
        self.targets = np.asarray([r["target"] for r in self.records], dtype=np.float32 if self.multilabel else np.int64)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as opened:
            image = opened.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[index].copy() if self.multilabel else int(self.targets[index])
        return image, target


def load_dataset(name, root, split, transform=None):
    return ManifestDataset(name, root, split, transform)


def iid_partition(dataset_or_len, num_clients=10, seed=0):
    """Seeded random IID index partitions, reused unchanged by DAPT and FT."""
    count = int(dataset_or_len) if isinstance(dataset_or_len, (int, np.integer)) else len(dataset_or_len)
    if num_clients < 1 or count < num_clients:
        raise ValueError("Every client requires at least one sample")
    indices = np.random.default_rng(seed).permutation(count)
    return list(np.array_split(indices, num_clients))
