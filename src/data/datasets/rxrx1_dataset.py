import ssl

import numpy as np
from numpy.random import random
from data.datasets.dataset_abstract import DinoDataset
from wilds import get_dataset


class RxRx1FromWILDS(DinoDataset):
    def __init__(
        self,
        csv_path,
        transform,
        inc_unlabeled=False,
        target_transform=None,
        custom_targets=None,
        train=True,
        path_prefix="",
        subsample_fraction=1.0,
        max_shots=0,
        seed=0,
        eval_split="id_test",   # NEW: 'id_test' (ID feature-quality eval) or 'test'/'val' (OOD)
        dataset_paths=None,
    ):
        super().__init__(in_c=3)                     # RxRx1-WILDS is 3-channel, keep DINOv2 patch embed
        self.transforms = transform
        self.custom_targets = custom_targets
        root_dir = (dataset_paths or {}).get("wilds_root", "")
        if not root_dir:
            raise ValueError("RxRx1 requires wilds_root in dataset_paths")
        ssl._create_default_https_context = ssl._create_unverified_context
        full_dataset = get_dataset(dataset="rxrx1", download=True, root_dir=root_dir)

        # RxRx1-WILDS 'official' splits: 'train', 'val' (OOD), 'test' (OOD), 'id_test' (ID).
        # NOTE: unlike fMoW there is NO 'id_val' split. 'id_test' = held-out site 2 of the
        # SAME 33 training experiments -> same batch distribution as train (ID eval).
        split = "train" if train else eval_split
        subset = full_dataset.get_subset(split)

        y_array = subset.y_array                     # torch LongTensor, labels 0..1138 (1139 classes)
        self.dataset = subset
        self.y_array = y_array
        self.data_len = len(self.dataset)
        self.path_prefix = root_dir

        # must be last: it re-slices self.dataset / self.y_array / self.custom_targets
        if subsample_fraction < 1.0:
            self.subsample_dataset(subsample_fraction, seed=seed)
        if max_shots > 0:
            self.subsample_shots(max_shots, seed=seed)

    def subsample_shots(self, shots, seed=0):
        """Keep at most `shots` samples per class (few-shot sampling).
        Classes with fewer than `shots` examples keep all of theirs. Sampling is
        stratified per class and deterministic given `seed`. Pass e.g. 5 for 5-shot.
        """
        assert shots > 0
        rng = random.Random(seed)
        targets = self.get_targets()  # int label per current sample, aligned to self.dataset
        selected_pos = []
        for cls in np.unique(targets):
            pos = np.where(targets == cls)[0]
            n_keep = min(shots, len(pos))
            selected_pos.extend(rng.sample(list(pos), n_keep))
        selected_pos = np.array(sorted(selected_pos), dtype=np.int64)
        # Re-slice the WILDS subset via its own base indices so that y_array,
        # metadata_array, and full_idxs all stay aligned.
        current_base_indices = np.asarray(self.dataset.indices)
        new_base_indices = current_base_indices[selected_pos]
        self.dataset = type(self.dataset)(
            self.dataset.dataset, new_base_indices, self.dataset.transform
        )
        # Keep the cached views on `self` in sync.
        self.y_array = self.dataset.y_array
        if self.custom_targets is not None:
            self.custom_targets = [self.custom_targets[i] for i in selected_pos]
        self.data_len = len(self.dataset)
        return selected_pos

    def get_targets(self):
        if self.custom_targets is None:
            return self.y_array.numpy().astype(np.int64)
        return np.array(self.custom_targets, dtype=np.int64)

    def __getitem__(self, index):
        x, y, metadata = self.dataset[int(index)]  # x is a PIL RGB image
        img_tensor = self.transforms(x.convert("RGB"))
        return img_tensor, int(y)

    def __len__(self):
        return self.data_len