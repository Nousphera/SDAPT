import os
from PIL import Image, UnidentifiedImageError
import numpy as np
from data.datasets.dataset_abstract import DinoDataset
import pandas as pd


class DomainNetSketchDataset(DinoDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(
        self,
        transform,
        csv_path,
        train = True,
        target_transform=None,
        subsample_fraction=1.0,
        seed=0, 
        custom_targets=None,
        path_prefix="",
        max_shots=0,
        dataset_paths=None,
    ):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Load data from huggingface
        dataset_paths = dataset_paths or {}
        self.data_dir = dataset_paths.get("sketch_images") or os.environ.get("TMPDIR")
        manifest = dataset_paths.get("sketch_test_manifest" if not train else "sketch_train_manifest")
        if not self.data_dir or not manifest:
            raise ValueError("Sketch requires sketch_images and train/test manifest paths in dataset_paths")
        if not train:
            self.dataset = pd.read_csv(manifest, sep=' ', header=None, names=['file', 'target'])
        else:
            self.dataset = pd.read_csv(manifest, sep=' ', header=None, names=['file', 'target'])

        min_label = self.dataset['target'].min()
        print(f"min_label: {min_label}")
        num_classes = self.dataset['target'].nunique()
        print(f"Number of classes: {num_classes}")
        # Calculate len
        if max_shots > 0:
            self._apply_few_shot(max_shots, seed)
        self.data_len = len(self.dataset)

        self.path_prefix = path_prefix
        # custom targets to replace CATEGORIES
        self.custom_targets = custom_targets
    
    def get_targets(self):
        return np.array(self.dataset['target'].values, dtype=np.int64)

    def _apply_few_shot(self, max_shots, seed):
        """
        Restrict the dataset to at most `max_shots` samples per class.
        Selection is deterministic given `seed`. Classes with fewer than
        `max_shots` samples keep all of their samples.
        """
        self.dataset = (
            self.dataset
            .groupby("target", group_keys=False)
            .apply(lambda g: g.sample(n=min(max_shots, len(g)), random_state=seed))
            .reset_index(drop=True)
        )

    def __getitem__(self, index):
        sample_path, target = self.dataset.iloc[int(index)][['file', 'target']]

        img = Image.open(os.path.join(self.data_dir, sample_path)).convert("RGB")
        img_tensor = self.transforms(img)

        return img_tensor, int(target)

    def __len__(self):
        return self.data_len