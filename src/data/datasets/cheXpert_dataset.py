# The 14 CheXpert diagnostic observations (multilabel head).
# Each is 4-way encoded on HuggingFace: unlabeled / uncertain / absent / present.
import copy
import glob
import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import ClassLabel, Dataset, Features
from datasets import Image as HFImage

from data.datasets.dataset_abstract import DinoDataset

CHEXPERT_PATHOLOGY_TASKS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# The 5 pathologies used for the official CheXpert benchmark AUC.
CHEXPERT_COMPETITION_TASKS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
    "Support Devices",
]

# Extra (non-diagnostic) attributes exposed as their own tasks.
CHEXPERT_METADATA_TASKS = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]

CHEXPERT_HF_CLASSES = ["unlabeled", "uncertain", "absent", "present"]
_CHEXPERT_CODE = {n: i for i, n in enumerate(CHEXPERT_HF_CLASSES)}

# Column names that might hold the relative image path in a labels file.
_PATH_COLUMN_CANDIDATES = (
    "path",
    "Path",
    "path_to_image",
    "image_id",
    "img_path",
    "filename",
)


class CheXpertFromHuggingFace(DinoDataset):
    def __init__(
        self,
        transform,
        csv_path=None,  # local labels file (.csv or .jsonl) for this split.
        image_root=None,  # local image folder, e.g. "dataman/cxr".
        target_transform=None,
        pathology_tasks=None,
        uncertain_policy="ignore",
        train=True,  # kept for call-site compatibility; loading itself is identical either way
        to_rgb=True,
        frontal_only=False,
        subsample_fraction=1.0,
        few_shot_task=None,
        max_shots=None,
        seed=0,
        dataset_paths=None,
    ):
        super().__init__(in_c=3)
        self.transforms = transform
        self.target_transform = target_transform
        self.to_rgb = to_rgb

        dataset_paths = dataset_paths or {}
        if train:
            csv_path = csv_path or dataset_paths.get("chexpert_train_labels")
            image_root = image_root or dataset_paths.get("chexpert_train_images")
        else:
            csv_path = csv_path or dataset_paths.get("chexpert_test_labels")
            image_root = image_root or dataset_paths.get("chexpert_test_images")
        if not csv_path or not image_root:
            raise ValueError("CheXpert requires train/test labels and image paths in dataset_paths")

        assert uncertain_policy in ("ones", "zeros", "ignore")
        self.uncertain_policy = uncertain_policy

        self.pathology_tasks = list(pathology_tasks or CHEXPERT_PATHOLOGY_TASKS)

        self.dataset = self._load_local_split(
            image_root, csv_path, frontal_only=frontal_only
        )

        self.data_len = len(self.dataset)

        # Build name<->code maps per task straight from the HF ClassLabel features,
        # so we never hardcode an encoding that could drift.
        features = self.dataset.features
        self.task_names = {}  # task -> {class_name: code}
        for task in self.pathology_tasks:
            feat = features[task]
            if isinstance(feat, ClassLabel):
                self.task_names[task] = {n: i for i, n in enumerate(feat.names)}
            else:
                self.task_names[task] = {}

        self.label_to_idx = {task: i for i, task in enumerate(self.pathology_tasks)}

        # Scalar single-task mode. None => emit the full multilabel vector.
        # Set via few_shot_task (test: whole split, scalar labels) or via
        # to_few_shot / max_shots (train: balanced few-shot support set).
        self.single_task_idx = (
            self.label_to_idx[few_shot_task] if few_shot_task is not None else None
        )

        # few-shot and subsample are mutually exclusive: subsample_dataset drops
        # all-negative rows, which would wipe out the clean negatives few-shot needs.
        if max_shots is not None:
            assert few_shot_task is not None, "max_shots requires few_shot_task"
            self.to_few_shot(few_shot_task, max_shots, seed=seed)
        elif subsample_fraction < 1.0:
            self.subsample_dataset(subsample_fraction, seed=seed)

    def __len__(self):
        return self.data_len

    def _binarize(self, task, code):
        names = self.task_names.get(task, {})
        present = names.get("present")
        uncertain = names.get("uncertain")

        if present is not None and code == present:
            return 1.0
        if uncertain is not None and code == uncertain:
            return {"ones": 1.0, "zeros": 0.0, "ignore": -1.0}[self.uncertain_policy]
        # absent, unlabeled, or any unmatched code -> negative
        return 0.0

    # ------------------------------------------------------------------ #
    # Local (non-HF) loading: shared by train and test.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_path(p, root):
        """Resolve a (possibly split-prefixed) relative image path against a
        local image root, e.g. "train/patient42142/study5/view1_frontal.jpg"
        -> "<root>/patient42142/study5/view1_frontal.jpg".
        """
        root = Path(root)
        p = str(p).replace("\\", "/")

        for base in (root, root.parent):
            if (base / p).exists():
                return str(base / p)

        # Cheap fix for a leading split folder that isn't actually on disk
        # (e.g. labels say "train/patientX/...", files live at root/patientX/...).
        parts = p.split("/")
        if len(parts) > 1:
            stripped = root / "/".join(parts[1:])
            if stripped.exists():
                return str(stripped)

        # Last resort: recursive glob on the trailing patient/study/view segments.
        tail = "/".join(parts[-3:])  # patientX/studyY/viewZ.jpg
        hits = glob.glob(str(root / "**" / tail), recursive=True)
        if hits:
            return hits[0]

        raise FileNotFoundError(f"cannot locate {p} under {root}")

    @staticmethod
    def _to_code(v):
        """Standard CheXpert value (1.0/0.0/-1.0/null/NaN) -> 4-way HF code."""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return _CHEXPERT_CODE["unlabeled"]
        v = float(v)
        return {
            1.0: _CHEXPERT_CODE["present"],
            0.0: _CHEXPERT_CODE["absent"],
            -1.0: _CHEXPERT_CODE["uncertain"],
        }.get(v, _CHEXPERT_CODE["unlabeled"])

    def _load_local_split(self, image_root, labels_path, frontal_only=False):
        """Build an HF-shaped Dataset from a local image folder + a labels
        file. Supports ``.csv`` (test-style, one row per image with a path
        column) and ``.jsonl``/``.ndjson`` (one JSON object per line, keyed
        by "path_to_image"). See class docstring for the encoding.
        """
        image_root = Path(image_root)
        labels_path = Path(labels_path)
        ext = labels_path.suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(labels_path)
        elif ext in (".jsonl", ".ndjson", ".json"):
            # Robust to either true JSONL or a JSON-array file.
            try:
                df = pd.read_json(labels_path, lines=True)
            except ValueError:
                records = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
                df = pd.DataFrame.from_records(records)
        else:
            raise ValueError(f"unsupported labels file type: {labels_path}")

        path_col = next((c for c in _PATH_COLUMN_CANDIDATES if c in df.columns), None)
        assert path_col is not None, (
            f"no path column found in {labels_path} "
            f"(looked for {_PATH_COLUMN_CANDIDATES})"
        )

        n_requested = len(df)
        resolved, found = [], []
        for p in df[path_col]:
            try:
                resolved.append(self._resolve_path(p, image_root))
                found.append(True)
            except FileNotFoundError:
                resolved.append(None)
                found.append(False)

        df["_abs"] = resolved
        n_missing = n_requested - sum(found)
        if n_missing:
            df = df[found].reset_index(drop=True)

        if frontal_only:
            df = df[df["_abs"].str.lower().str.contains("frontal")].reset_index(drop=True)

        data = {"image": df["_abs"].tolist()}
        missing_cols = []
        for task in self.pathology_tasks:
            if task in df.columns:
                data[task] = [self._to_code(v) for v in df[task]]
            else:
                missing_cols.append(task)
                data[task] = [_CHEXPERT_CODE["unlabeled"]] * len(df)
        if missing_cols:
            print(
                f"[chexpert-local] columns absent from {labels_path.name}, "
                f"filled as unlabeled: {missing_cols}"
            )

        print(
            f"[chexpert-local] loaded {len(df)} samples from {labels_path.name}"
            + (f" ({n_missing} skipped: image not found under {image_root})" if n_missing else "")
        )

        features = Features(
            {
                "image": HFImage(),
                **{t: ClassLabel(names=CHEXPERT_HF_CLASSES) for t in self.pathology_tasks},
            }
        )
        return Dataset.from_dict(data, features=features)

    def few_shot_indices(self, task, k_shot, seed=0):
        """Sample a balanced few-shot support set for ONE pathology `task`.

        Returns (pos_idx, neg_idx): dataset indices, each up to length k_shot.

        Policy:
          * positives = rows encoded 'present'
          * negatives = explicit 'absent' first, then blank/'unlabeled' fallback
          * 'uncertain' rows are NEVER used as anchors (noisiest label)
          * balanced (k_shot vs k_shot) because CheXpert is negative-skewed
        """
        names = self.task_names.get(task, {})
        present = names.get("present")
        absent = names.get("absent")
        uncertain = names.get("uncertain")
        assert present is not None, f"{task} has no 'present' class on HF"

        col = [int(c) for c in self.dataset[task]]  # raw 0..3 codes for the split

        pos_idx = [i for i, c in enumerate(col) if c == present]
        neg_primary = [
            i for i, c in enumerate(col) if absent is not None and c == absent
        ]
        neg_fallback = [
            i for i, c in enumerate(col) if c not in (present, uncertain, absent)
        ]  # blank / unlabeled

        rng = random.Random(seed)

        def take(pool, k):
            return list(pool) if len(pool) <= k else rng.sample(pool, k)

        pos_sel = take(pos_idx, k_shot)
        neg_sel = take(neg_primary, k_shot)
        if len(neg_sel) < k_shot:  # top up from blanks
            neg_sel += take(neg_fallback, k_shot - len(neg_sel))

        return pos_sel, neg_sel

    def to_few_shot(self, task, max_shots, seed=0):
        """Reduce this dataset to a balanced, single-task few-shot support set and
        switch __getitem__ to emit a scalar 0/1 label for `task`.

        ~max_shots total: max_shots//2 positives + max_shots//2 negatives.
        Balanced because CheXpert is heavily negative-skewed. Mutates in place
        (like subsample_dataset) and returns self.
        """
        k_per_class = max_shots
        pos, neg = self.few_shot_indices(task, k_per_class, seed=seed)

        selected = sorted(pos + neg)
        self.dataset = self.dataset.select(selected)
        self.data_len = len(self.dataset)
        self.single_task_idx = self.label_to_idx[task]  # turns on scalar output

        if len(pos) < k_per_class or len(neg) < k_per_class:
            print(
                f"[few-shot] {task}: wanted {k_per_class}/class, "
                f"got {len(pos)} pos / {len(neg)} neg"
            )
        return self

    def _build_target(self, sample):
        return torch.tensor(
            [self._binarize(task, int(sample[task])) for task in self.pathology_tasks],
            dtype=torch.float32,
        )

    def make_few_shot_dataset(self, task, max_shots, seed=0):
        pos, neg = self.few_shot_indices(task, max_shots, seed=seed)
        selected = sorted(pos + neg)

        new_ds = copy.copy(self)  # shallow copy of the wrapper object only
        new_ds.dataset = self.dataset.select(selected)  # fresh HF Dataset, doesn't touch self.dataset
        new_ds.data_len = len(new_ds.dataset)
        new_ds.single_task_idx = self.label_to_idx[task]  # scalar output on

        if len(pos) < max_shots or len(neg) < max_shots:
            print(
                f"[few-shot] {task}: wanted {max_shots}/class, "
                f"got {len(pos)} pos / {len(neg)} neg"
            )
        return new_ds

    def subsample_dataset(self, fraction, seed=0):
        """Keep `fraction` of the positive samples for *every* task, then restrict
        the dataset to the union of the selected indices.

        Unlike a global subsample (take `fraction` of all rows at random), this
        guarantees each pathology retains ~`fraction` of its positive examples,
        so rare classes don't get wiped out. Pass e.g. 0.1 for 10%.

        Note (multilabel): one row can satisfy the quota for several tasks at once,
        so the resulting subset is usually larger than `fraction * len(dataset)`,
        and rows that are negative for *all* tasks are dropped entirely.
        """
        assert 0.0 < fraction <= 1.0
        rng = random.Random(seed)

        selected = set()
        for task in self.pathology_tasks:
            col = self.dataset[task]  # raw 0..3 codes for the whole split
            pos_idx = [
                i for i, code in enumerate(col)
                if self._binarize(task, int(code)) == 1.0
            ]
            n_keep = round(fraction * len(pos_idx))
            if n_keep:
                selected.update(rng.sample(pos_idx, n_keep))

        selected = sorted(selected)
        self.dataset = self.dataset.select(selected)
        self.data_len = len(self.dataset)
        return selected

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        if self.to_rgb and image.mode != "RGB":
            image = image.convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)

        target = self._build_target(sample)
        if self.single_task_idx is not None:
            target = target[self.single_task_idx].long()  # scalar 0/1 for KnnHead
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target