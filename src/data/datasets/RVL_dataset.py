import os

import numpy as np
from data.datasets.dataset_abstract import DinoDataset
from datasets import concatenate_datasets, load_dataset
import PIL.Image
# RVL-CDIP class names in the order used by `chainyo/rvl-cdip` (alphabetical).
# NOTE: the *integer* labels come straight from the loaded HF dataset (the
# `label` field is already an int in [0, 15]), so training never depends on this
# list -- it's here for readability/logging only, and the class also reads the
# authoritative names from the dataset's ClassLabel feature at runtime. This
# matters because HF mirrors order the 16 classes differently: `chainyo/rvl-cdip`
# is alphabetical (below), while `aharley/rvl_cdip` uses the original Harley
# order (letter=0, form=1, ...). Never compare label integers across mirrors.
CATEGORIES = [
    "advertisement",
    "budget",
    "email",
    "file_folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news_article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific_publication",
    "scientific_report",
    "specification",
]


class RVLCDIPFromHuggingFace(DinoDataset):
    # DINOv2 expects ImageNet-normalized 3-channel input. RVL-CDIP images are
    # grayscale (PIL mode "L") and get replicated to RGB, so all three channels
    # share identical stats. Keeping ImageNet stats keeps the input distribution
    # aligned with what the pretrained backbone saw. If you prefer to normalize
    # to the document domain, compute mean/std over the train split and swap
    # them in (all three channels will be equal).
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        csv_path,
        transform,
        target_transform=None,
        max_shots=0,
        subsample_fraction=1.0,
        train=True,
        custom_targets=None,
        path_prefix="",
        seed=0,
        split=None,
        combine_train_val=True,
    ):
        """
        Creates a Dataset for RVL-CDIP document image classification (16 classes).
        :param csv_path: kept for interface compatibility (unused, data comes from HF).
        :param transform: pytorch transforms for augmentation and tensor conversion.
        :param split: optional explicit HF split ("train"/"validation"/"test").
                      If None, derived from `train` (eval defaults to "test").
        :param combine_train_val: when building the train pool, merge the val
                      split into it (~35k + 40k) since eval uses the held-out
                      test split. Set False to keep the ~35k train split alone.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform

        # Load data from huggingface
        path_prefix = os.environ.get("TMPDIR")

        # `chainyo/rvl-cdip` is stored as Parquet (no loading script), so it
        # loads on `datasets` >= 4.x with no trust_remote_code.
        # Splits/sizes for this mirror: train ~35k, val 40k, test 40k. The test
        # split is the full 40k benchmark set, but the train split is TRUNCATED
        # to ~35k -- fine for few-shot sampling and eval, but light for the DAPT
        # stage. If you need the full 320k train pool for continued pretraining,
        # re-host the complete set yourself (as with FMoW) or pull IIT-CDIP.
        dataset = load_dataset("chainyo/rvl-cdip", cache_dir=path_prefix)

        # Eval defaults to the *test* split (full 40k). This mirror names the
        # validation split "val" (not "validation"), so normalize that alias.
        if not train:
            split = "test"
            self.dataset = dataset['test']
        if train:
            self.dataset = concatenate_datasets([dataset["train"], dataset["val"]])
 

        # Authoritative class names for this specific mirror/version.
        label_feature = self.dataset.features["label"]
        self.class_names = list(getattr(label_feature, "names", CATEGORIES))
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Calculate len (optionally restrict to a few-shot subset first)
        if max_shots > 0:
            self._apply_few_shot(max_shots, seed)

        self.data_len = len(self.dataset)

        self.path_prefix = path_prefix

        # custom targets to replace the dataset labels
        self.custom_targets = custom_targets

    def get_targets(self):
        if self.custom_targets is None:
            # labels are already integer class indices (0-15), no name lookup needed
            return np.array(self.dataset["label"], dtype=np.int64)
        return np.array(self.custom_targets, dtype=np.int64)

    def _apply_few_shot(self, max_shots, seed):
        """
        Restrict the dataset to at most `max_shots` samples per class.
        Selection is deterministic given `seed`. Classes with fewer than
        `max_shots` samples keep all of their samples.
        """
        labels = self.dataset["label"]  # columnar access, fast in HF datasets

        # group example indices by class (labels are ints, fine as dict keys)
        indices_by_class = {}
        for idx, label in enumerate(labels):
            indices_by_class.setdefault(label, []).append(idx)

        rng = np.random.RandomState(seed)

        selected = []
        for label, idxs in indices_by_class.items():
            idxs = np.array(idxs)
            if len(idxs) > max_shots:
                chosen = rng.choice(idxs, size=max_shots, replace=False)
            else:
                chosen = idxs
            selected.extend(chosen.tolist())

        selected.sort()  # keep a stable ordering

        self.dataset = self.dataset.select(selected)
        self.data_len = len(self.dataset)

    def __getitem__(self, index):
        index = int(index)
        n = self.data_len
        for offset in range(n):
            i = (index + offset) % n
            try:
                example = self.dataset[i]
                img = example["image"].convert("RGB")
                break
            except (PIL.UnidentifiedImageError, OSError) as e:
                if offset == 0:
                    print(f"[RVL] skipping undecodable image at index {i}: {e}")
                continue
        else:
            raise RuntimeError("No decodable image found in RVL-CDIP dataset")

        img_tensor = self.transforms(img)
        label = int(example["label"])
        return img_tensor, label

    def __len__(self):
        return self.data_len