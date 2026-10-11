import csv
import os
from PIL import Image
import random
import numpy as np
import torch
from data.datasets.dataset_abstract import Dataset, DinoDataset
import torch

PROTEIN_COLS = [
    "nucleoplasm", "nuclear membrane", "nucleoli", "nucleoli fibrillar center",
    "nuclear speckles", "nuclear bodies", "endoplasmic reticulum",
    "golgi apparatus", "peroxisomes", "endosomes", "lysosomes",
    "intermediate filaments", "actin filaments", "focal adhesion sites",
    "microtubules", "microtubule ends", "cytokinetic bridge",
    "mitotic spindle", "microtubule organizing center", "centrosome",
    "lipid droplets", "plasma membrane", "cell junctions", "mitochondria",
    "aggresome", "cytosol", "cytoplasmic bodies", "rods & rings",
]
NUM_CLASSES = len(PROTEIN_COLS)

class SelfNormalize:
    """Per-image, per-channel standardization for fluorescence. Expects (C,H,W) float.
    div_std=False reproduces FINO's 'SelfNormalizeNoDiv' (center only)."""
    def __init__(self, div_std=True, eps=1e-6):
        self.div_std, self.eps = div_std, eps
    def __call__(self, x):
        x = x - x.mean(dim=(1, 2), keepdim=True)
        if self.div_std:
            x = x / (x.std(dim=(1, 2), keepdim=True) + self.eps)
        return x

class PerChannelIntensityJitter:
    """Per-channel brightness/contrast for (C,H,W) float tensors in ~[0,1].
    Channel-safe replacement for ColorJitter (which assumes 1/3-channel RGB)."""
    def __init__(self, brightness=0.4, contrast=0.4):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x):                       # x: (C, H, W), float in ~[0,1]
        C = x.shape[0]
        b = 1.0 + (torch.rand(C, 1, 1) * 2 - 1) * self.brightness
        c = 1.0 + (torch.rand(C, 1, 1) * 2 - 1) * self.contrast
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = (x - mean) * c + mean * b
        return x.clamp_(0.0, 1.0)

def _load_packed_4ch(path):
    """Load a packed [red|green|blue|yellow] grayscale JPG -> (4, H, W) float in [0,1].
 
    pack_channels wrote each image as one grayscale JPG of shape (H, W*4) with the
    four channels concatenated horizontally, so we recover them by slicing width
    into four equal blocks (channel order: red, green, blue, yellow).
    """
    arr = np.asarray(Image.open(path).convert("L"))  # (H, W*4), uint8
    w = arr.shape[1] // 4
    channels = [arr[:, i * w:(i + 1) * w] for i in range(4)]
    img = np.stack(channels, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(img)

class HPADataset(DinoDataset):
    def __init__(
        self,
        img_root=None,
        manifest_root=None,
        target_transform=None,
        transform=None,
        csv_path=None,
        train=True,
        is_test=False,
        test_dir=None,
        seed=0,
        subsample_fraction=1.0,
        image_col="hr_image",
        dataset_paths=None,
    ):

        super().__init__(in_c=4)  # 4 channels for the packed JPGs
        self.transform = transform
        self.is_test = is_test
        self._warned_missing = set()   # so we warn once per bad path, not per epoch
        self.max_skip_attempts = 100   # give up if too many consecutive bad files
        self.train = train
        dataset_paths = dataset_paths or {}
        img_root = img_root or dataset_paths.get("hpa_images")
        manifest_root = manifest_root or dataset_paths.get("hpa_manifest")
        if not img_root or not manifest_root:
            raise ValueError("HPA requires hpa_images and hpa_manifest in dataset_paths")
 
        self.samples = []  
        self.ids = []      
 
        if not self.train:
            test_csv = os.path.join(manifest_root, "sample_submission.csv")
            test_img = os.path.join(img_root, "HPAImageKaggle", "jpg_768_test")
            with open(test_csv, newline="") as f:
                for row in csv.DictReader(f):
                    fname = row["Id"]
                    fname_jpg = fname if fname.lower().endswith(".jpg") else fname + ".jpg"
                    self.samples.append((os.path.join(test_img, fname_jpg), None))
                    self.ids.append(os.path.splitext(os.path.basename(fname))[0])
        else:
            # Each training manifest is paired with its own packed-JPG directory.
            pairs = [
                (os.path.join(manifest_root, "hpa_whole_external_hr.csv"),
                 os.path.join(img_root, "HPAImageKaggle", "jpg_768x768_4channels")),
                (os.path.join(manifest_root, "hpa_whole_kaggle.csv"),
                 os.path.join(img_root, "HPAImageKaggle", "jpg_768_train")),
            ]
            for csv_path, img_dir in pairs:
                with open(csv_path, newline="") as f:
                    for row in csv.DictReader(f):
                        hr = row.get(image_col, "")
                        if not hr:
                            continue
                        target = np.array(
                            [1.0 if row[c] == "True" else 0.0 for c in PROTEIN_COLS],
                            dtype=np.float32,
                        )
                        self.samples.append((os.path.join(img_dir, hr), target))
                        self.ids.append(row.get("ID", os.path.splitext(hr)[0]))

        self.data_len = len(self.samples)
        if subsample_fraction < 1.0:
            self.subsample_dataset(subsample_fraction, seed=seed)   # reuse existing seed param
 
    def get_targets(self):
        """(N, 28) multi-hot matrix for train; None for the unlabeled test set."""
        if self.is_test:
            return None
        return np.stack([t for _, t in self.samples]).astype(np.float32)

    def subsample_dataset(self, fraction, seed=0):
        """Keep `fraction` of the positive samples for *every* protein class, then
        restrict the dataset to the union of the selected samples.

        HPA is multilabel (28-way multi-hot), so "samples of a task" means the
        samples positive for that class. Doing it per class guarantees rare
        proteins keep ~`fraction` of their positives instead of being thinned out.
        Pass e.g. 0.1 for 10%.

        Multilabel caveat: one image can satisfy several classes' quotas at once,
        so the union is usually larger than `fraction * len(dataset)`, and images
        negative for every class are dropped. On the unlabeled test set there are
        no targets, so this degrades to a plain random subsample.
        """
        assert 0.0 < fraction <= 1.0
        rng = random.Random(seed)

        # Unlabeled test set: no per-class labels, so just subsample rows at random.
        if self.is_test or not self.train:
            n_keep = round(fraction * self.data_len)
            selected = sorted(rng.sample(range(self.data_len), n_keep))
        else:
            targets = self.get_targets()  # (N, 28) multi-hot
            selected = set()
            for cls in range(NUM_CLASSES):
                pos_idx = np.where(targets[:, cls] == 1.0)[0]
                n_keep = round(fraction * len(pos_idx))
                if n_keep:
                    selected.update(rng.sample(list(pos_idx), n_keep))
            selected = sorted(selected)

        self.samples = [self.samples[i] for i in selected]
        self.ids = [self.ids[i] for i in selected]
        self.data_len = len(self.samples)
        return selected
    
    def __getitem__(self, index):
        index = int(index)
        attempts = 0
        while True:
            path, target = self.samples[index]
            try:
                img = _load_packed_4ch(path)            # (4, H, W) in [0, 1]
                break
            except OSError as e:                        # FileNotFoundError + corrupt JPGs
                if self.is_test:
                    raise
                if path not in self._warned_missing:
                    self._warned_missing.add(path)
                    print(f"[HPADataset] skipping unreadable image: {path} ({e})")
                attempts += 1
                if attempts >= self.max_skip_attempts:
                    raise RuntimeError(
                        f"[HPADataset] {attempts} consecutive unreadable images; "
                        f"last was {path}. Check the image directory."
                    ) from e
                index = random.randrange(self.data_len)

        img = self.transform(img) if self.transform else self.normalize(img)

        if not self.train:
            return img, self.ids[index]             # no label available
        return img, torch.from_numpy(target)        # (28,) multi-hot float
 
    def __len__(self):
        return self.data_len