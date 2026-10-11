# --------------------------------------------------------
# References:
# SatMAE: https://github.com/sustainlab-group/SatMAE
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import os
import csv
import warnings
import logging
import random
import numpy as np
import pandas as pd
from enum import Enum
from typing import Any, Callable, Optional, List, Tuple, Union
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from wilds import get_dataset
import ssl
import torch
from huggingface_hub import login
from datasets import ClassLabel, load_dataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset, ImageFolder
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

try:
    # import rasterio
    # from rasterio import logging
    from tifffile import imread

    # log = logging.getLogger()
    # log.setLevel(logging.ERROR)
except ModuleNotFoundError as err:
    # Error handling
    print(err)
# import rasterio
# from rasterio import logging

# log = logging.getLogger()
# log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
use_antialias_key = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR > 12)


logger = logging.getLogger("linprobe")


CATEGORIES = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "amusement_park",
    "aquaculture",
    "archaeological_site",
    "barn",
    "border_checkpoint",
    "burial_site",
    "car_dealership",
    "construction_site",
    "crop_field",
    "dam",
    "debris_or_rubble",
    "educational_institution",
    "electric_substation",
    "factory_or_powerplant",
    "fire_station",
    "flooded_road",
    "fountain",
    "gas_station",
    "golf_course",
    "ground_transportation_station",
    "helipad",
    "hospital",
    "impoverished_settlement",
    "interchange",
    "lake_or_pond",
    "lighthouse",
    "military_facility",
    "multi-unit_residential",
    "nuclear_powerplant",
    "office_building",
    "oil_or_gas_facility",
    "park",
    "parking_lot_or_garage",
    "place_of_worship",
    "police_station",
    "port",
    "prison",
    "race_track",
    "railway_bridge",
    "recreational_facility",
    "road_bridge",
    "runway",
    "shipyard",
    "shopping_mall",
    "single-unit_residential",
    "smokestack",
    "solar_farm",
    "space_facility",
    "stadium",
    "storage_tank",
    "surface_mine",
    "swimming_pool",
    "toll_booth",
    "tower",
    "tunnel_opening",
    "waste_disposal",
    "water_treatment_facility",
    "wind_farm",
    "zoo",
]


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """

    def __init__(self, in_c):
        self.in_c = in_c

    def get_targets(self):
        return np.array([i for i, _ in enumerate(CATEGORIES)])

    @property
    def targets(self):
        return self.get_targets()

    @property
    def num_rows(self):
        return len(self)

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                (
                    transforms.RandomResizedCrop(
                        input_size, scale=(0.2, 1.0), interpolation=interpol_mode, antialias=None
                    )
                    if use_antialias_key
                    else transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode)
                ),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(
                size, interpolation=interpol_mode, antialias=None
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)

class FMoWFromHuggingFace(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(
        self,
        csv_path,
        transform,
        target_transform=None,
        custom_targets=None,
        path_prefix="",
        max_shots=0,
        seed=0,
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
        path_prefix = os.environ.get("TMPDIR")
        self.dataset = load_dataset("Staneman/DAPTSL_fMoW", cache_dir=path_prefix)
        if("test" in csv_path):
            self.dataset = self.dataset["validation"]
        else:
            self.dataset = self.dataset["train"]

        is_train_split = "test" not in csv_path
        self.dataset = self.subsample_train_split(
            self.dataset,
            is_train_split=is_train_split,
            max_samples=max_shots,
            seed=seed,
        )

        # Calculate len
        self.data_len = len(self.dataset)

        self.path_prefix = path_prefix

        self.label_to_idx = {label: i for i, label in enumerate(CATEGORIES)}

        # custom targets to replace CATEGORIES
        self.custom_targets = custom_targets

    def subsample_train_split(self, dataset_split, is_train_split, max_samples, seed):
        # Only downsample training data; validation/test stay unchanged.
        if not is_train_split:
            return dataset_split

        if max_samples is None or int(max_samples) <= 0:
            logger.info("HF few-shot disabled: using full training split")
            return dataset_split

        shots_per_class = int(max_samples)
        labels = dataset_split["label"]
        total_samples = len(labels)

        label_to_indices = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(idx)

        rng = random.Random(seed)
        selected_indices = []
        per_class_kept = {}
        for label, indices in label_to_indices.items():
            rng.shuffle(indices)
            keep = min(shots_per_class, len(indices))
            per_class_kept[label] = keep
            selected_indices.extend(indices[:keep])

        rng.shuffle(selected_indices)

        underfull_classes = sum(
            1 for label, indices in label_to_indices.items() if len(indices) < shots_per_class
        )
        logger.info(
            "HF few-shot applied: kept %d samples across %d classes (shots_per_class=%d, underfull_classes=%d, total_before=%d)",
            len(selected_indices),
            len(label_to_indices),
            shots_per_class,
            underfull_classes,
            total_samples,
        )
        return dataset_split.select(selected_indices)

    def get_targets(self):
        if self.custom_targets is None:
            return np.array(
                [self.label_to_idx[label] for label in self.dataset["label"]],
                dtype=np.int64,
            )
        return np.array(self.custom_targets, dtype=np.int64)

    def __getitem__(self, index):
        example = self.dataset[int(index)]

        img = example["image"].convert("RGB")
        img_tensor = self.transforms(img)

        label = self.label_to_idx[example["label"]]
        return img_tensor, label

    def __len__(self):
        return self.data_len

# The 14 CheXpert diagnostic observations (multilabel head).
# Each is 4-way encoded on HuggingFace: unlabeled / uncertain / absent / present.
CHEXPERT_PATHOLOGY_TASKS = [
    "No Finding",
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
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]
 
# Extra (non-diagnostic) attributes exposed as their own tasks.
CHEXPERT_METADATA_TASKS = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]

class CheXpertFromHuggingFace(Dataset):
    """CheXpert chest X-ray dataset loaded from HuggingFace, structured as a
    multi-task / multilabel dataset.
 
    Each item is ``(image, target)`` where ``target`` is a dict:
 
        {
            "labels":     FloatTensor[num_pathology_tasks]   # binarized multilabel
            "labels_raw": LongTensor[num_pathology_tasks]     # raw 0..3 codes
            "metadata":   {"sex", "age", "frontal_lateral", "ap_pa"}
        }
 
    The 14 diagnostic observations are 4-way encoded on HuggingFace
    (unlabeled / uncertain / absent / present). ``labels`` collapses them to a
    binary multilabel vector using ``uncertain_policy``:
        - "ones"   : uncertain -> 1
        - "zeros"  : uncertain -> 0
        - "ignore" : uncertain -> -1  (mask these out in your loss / metric)
    In all policies: present -> 1, absent -> 0, unlabeled -> 0.
    """
 
    # Placeholder normalization stats (grayscale replicated to 3 channels).
    # Recompute on your own split if you need exact values.
    mean = [0.5029, 0.5029, 0.5029]
    std = [0.2899, 0.2899, 0.2899]
 
    def __init__(
        self,
        csv_path,
        transform,
        target_transform=None,
        pathology_tasks=None,
        uncertain_policy="ignore",
        to_rgb=True,
        hf_repo="danjacobellis/chexpert",
    ):
        super().__init__()
        self.transforms = transform
        self.target_transform = target_transform
        self.to_rgb = to_rgb
 
        assert uncertain_policy in ("ones", "zeros", "ignore")
        self.uncertain_policy = uncertain_policy
 
        self.pathology_tasks = list(pathology_tasks or CHEXPERT_PATHOLOGY_TASKS)
 
        # Load from HuggingFace and pick the split.
        path_prefix = os.environ.get("TMPDIR")
        dataset = load_dataset(hf_repo, cache_dir=path_prefix)
        split = "validation" if "test" in csv_path else "train"
        self.dataset = dataset[split]
        self.path_prefix = path_prefix
 
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
 
        logger.info(
            "Loaded CheXpert split=%s (%d samples, %d pathology tasks, uncertain_policy=%s)",
            split,
            self.data_len,
            len(self.pathology_tasks),
            self.uncertain_policy,
        )
 
    def __len__(self):
        return self.data_len
 
    def _binarize(self, task, code):
        names = self.task_names.get(task, {})
        present = names.get("present")
        absent = names.get("absent")
        uncertain = names.get("uncertain")
 
        if present is not None and code == present:
            return 1.0
        if uncertain is not None and code == uncertain:
            return {"ones": 1.0, "zeros": 0.0, "ignore": -1.0}[self.uncertain_policy]
        # absent, unlabeled, or any unmatched code -> negative
        return 0.0
 
    def _build_target(self, sample):
        raw = torch.tensor(
            [int(sample[task]) for task in self.pathology_tasks], dtype=torch.long
        )
        labels = torch.tensor(
            [self._binarize(task, int(sample[task])) for task in self.pathology_tasks],
            dtype=torch.float32,
        )
        metadata = {
            "sex": sample.get("Sex"),
            "age": sample.get("Age"),
            "frontal_lateral": sample.get("Frontal/Lateral"),
            "ap_pa": sample.get("AP/PA"),
        }
        return {"labels": labels, "labels_raw": raw, "metadata": metadata}
 
    def __getitem__(self, idx):
        sample = self.dataset[idx]
 
        image = sample["image"]
        if self.to_rgb and image.mode != "RGB":
            image = image.convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
 
        target = self._build_target(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
 
        return image, target

class FMoWFromWILDS(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(
        self,
        csv_path,
        transform,
        inc_unlabeled=False,
        target_transform=None,
        custom_targets=None,
        path_prefix="",
        max_shots=0,
        seed=0,
    ):
        super().__init__(in_c=3)
        self.transforms = transform
        self.custom_targets = custom_targets

        root_dir = os.environ.get("TMPDIR")
        ssl._create_default_https_context = ssl._create_unverified_context
        full_dataset = get_dataset(dataset="fmow", download=True, root_dir=root_dir)

        # WILDS fMoW splits: 'train', 'id_val', 'id_test', 'val' (OOD), 'test' (OOD)
        # Your old code keyed off "test" in csv_path, so mirror that:
        split = "test" if "test" in csv_path else "train"   # use "id_test" for in-distribution
        is_train_split = "test" not in csv_path

        subset = full_dataset.get_subset(split)
        y_array = subset.y_array  # torch tensor of int labels (0..61)

        self.dataset = subset
        self.y_array = y_array

        self.data_len = len(self.dataset)
        self.path_prefix = root_dir

    def _csv_rows_for_subset(self):
        """CSV row indices for the current WILDS subset, in y_array order."""
        base = getattr(self.dataset, "dataset", self.dataset)
        indices = getattr(self.dataset, "indices", None)

        base2csv = getattr(base, "full_idxs", None)
        if base2csv is None:  # fallback: rebuild the seq-filter mapping ourselves
            base2csv = np.where(base.metadata["split"].to_numpy() != "seq")[0]

        if indices is None:
            csv_rows = np.asarray(base2csv)
        else:
            csv_rows = np.asarray(base2csv)[np.asarray(indices)]
        return base, csv_rows

    def get_country_codes(self):
        """One country_code per image, aligned to self.y_array (NaNs kept)."""
        base, csv_rows = self._csv_rows_for_subset()
        return base.metadata["country_code"].to_numpy()[csv_rows]

    def get_unique_microdomains(self, micro_domain):
        base = getattr(self.dataset, "dataset", self.dataset)
        fields = list(base.metadata_fields)  

        if micro_domain in fields:
            arr = self.dataset.metadata_array[:, fields.index(micro_domain)]
            return np.unique(arr.cpu().numpy())

        _, csv_rows = self._csv_rows_for_subset()
        col = "country_code" 
        values = base.metadata[col].to_numpy()[csv_rows]
        if values.dtype == object:
            values = values[~pd.isna(values)]
        return np.unique(values)

    def get_rows_by_microdomain_value(self, micro_domain, value):
        base = getattr(self.dataset, "dataset", self.dataset)
        fields = list(base.metadata_fields)   

        # region / year: already in the subset-aligned metadata_array
        if micro_domain in fields:
            col = fields.index(micro_domain)
            arr = self.dataset.metadata_array[:, col].cpu().numpy()
            value = self._resolve_metadata_value(base, micro_domain, value)
            return np.where(arr == value)[0]

        _, csv_rows = self._csv_rows_for_subset()
        col_name = "country_code"
        values = base.metadata[col_name].to_numpy()[csv_rows]
        return np.where(values == value)[0]      

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

# The 28 protein-location classes, in the fixed order used by the manifests.
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
 
 
class HPADataset(Dataset):
    def __init__(
        self,
        img_root=None,
        manifest_root=None,
        transform=None,
        is_test=False,
        test_dir=None,
        image_col="hr_image",
        dataset_paths=None,
    ):
        """Multi-label HPA dataset.
 
        Train (is_test=False): reads the two manifests built by hpa_whole_hr_build_csv
        (external + kaggle) under `root`, and their packed-JPG directories. Targets
        are (28,) multi-hot float tensors.
 
        Test (is_test=True): reads your local label-free CSV (`test_csv`) whose image
        filenames live in `test_dir`. __getitem__ returns (image, image_id) since
        there are no labels; align predictions by that id.
        """
        super().__init__()
        self.transform = transform
        self.is_test = is_test
        dataset_paths = dataset_paths or {}
        img_root = img_root or dataset_paths.get("hpa_images") or os.environ.get("HPA_ROOT")
        manifest_root = manifest_root or dataset_paths.get("hpa_manifest")
        if not img_root or not manifest_root:
            raise ValueError("HPA requires hpa_images and hpa_manifest in dataset_paths")
 
        self.samples = []  # list of (abs_image_path, target_or_None)
        self.ids = []      # aligned image ids (useful for the test submission)
 
        if is_test:
            test_csv = os.path.join(manifest_root, "sample_submission.csv")
            test_img = os.path.join(manifest_root, "jpg_768_test")
            with open(test_csv, newline="") as f:
                for row in csv.DictReader(f):
                    fname = row[image_col]
                    self.samples.append((os.path.join(test_img, fname), None))
                    self.ids.append(os.path.splitext(os.path.basename(fname))[0])
        else:
            # Each training manifest is paired with its own packed-JPG directory.
            pairs = [
                (os.path.join(manifest_root, "hpa_whole_external_hr.csv"),
                 os.path.join(img_root, "HPAexternal", "jpg_768x768_4channels")),
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
 
    def get_targets(self):
        """(N, 28) multi-hot matrix for train; None for the unlabeled test set."""
        if self.is_test:
            return None
        return np.stack([t for _, t in self.samples]).astype(np.float32)
 
    def __getitem__(self, index):
        path, target = self.samples[int(index)]
        img = _load_packed_4ch(path)                # (4, H, W) in [0, 1]
        img = self.transform(img) if self.transform else self.normalize(img)
 
        if self.is_test:
            return img, self.ids[int(index)]        # no label available
        return img, torch.from_numpy(target)        # (28,) multi-hot float
 
    def __len__(self):
        return self.data_len


# DIOR class names (index == category id in the HF dataset)
DIOR_CATEGORIES = [
    "Airplane", "Airport", "Baseball field", "Basketball court", "Bridge",
    "Chimney", "Dam", "Expressway service area", "Expressway toll station",
    "Golf course", "Ground track field", "Harbor", "Overpass", "Ship",
    "Stadium", "Storage tank", "Tennis court", "Train station", "Vehicle",
    "Wind mill",
]


class DiorFromHuggingFace(SatelliteDataset):
    # NOTE: these are the fMoW stats from your original class.
    # Recompute on DIOR if you want correct normalization for pretraining.
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(
        self,
        csv_path,
        transform,
        target_transform=None,
        custom_targets=None,
        path_prefix="",
        max_shots=0,
        seed=0,
        label_mode="majority",   # "majority" (most present object) or "constant"
        constant_label=0,        # used for "constant" mode and empty images
    ):
        """
        Domain-adaptive-pretraining dataset wrapping the DIOR object-detection
        dataset (HichTala/dior). Targets are not meaningful for DAPT, so each
        image gets either the majority object category or a constant dummy label.
        """
        super().__init__(in_c=3)
        self.transforms = transform
        self.target_transform = target_transform
        self.label_mode = label_mode
        self.constant_label = constant_label

        # Load data from huggingface
        path_prefix = os.environ.get("TMPDIR")
        self.dataset = load_dataset("HichTala/dior", cache_dir=path_prefix)
        if "test" in csv_path:
            # keeping your convention: "test" -> validation split.
            # DIOR also has a real "test" split if you'd rather use that.
            self.dataset = self.dataset["validation"]
        else:
            self.dataset = self.dataset["train"]

        is_train_split = "test" not in csv_path
        self.data_len = len(self.dataset)
        self.path_prefix = path_prefix
        self.custom_targets = custom_targets

        # category id <-> name (already integer-encoded in the dataset)
        self.label_to_idx = {name: i for i, name in enumerate(DIOR_CATEGORIES)}

        # Precompute image-level targets WITHOUT decoding images.
        # Accessing only the "objects" column avoids loading pixels.
        if custom_targets is not None:
            self._targets = np.array(custom_targets, dtype=np.int64)
        else:
            objects_col = self.dataset["objects"]
            self._targets = np.array(
                [self._image_label(obj["category"]) for obj in objects_col],
                dtype=np.int64,
            )

    def _image_label(self, categories):
        """Reduce per-object categories to a single image-level label."""
        if self.label_mode == "constant" or len(categories) == 0:
            return self.constant_label
        # majority vote (categories are ints in [0, 19])
        counts = np.bincount(categories, minlength=len(DIOR_CATEGORIES))
        return int(counts.argmax())

    def get_targets(self):
        return self._targets

    def __getitem__(self, index):
        example = self.dataset[int(index)]

        img = example["image"].convert("RGB")
        img_tensor = self.transforms(img)

        label = int(self._targets[index])   # <-- use _targets here too
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img_tensor, label

    def __len__(self):
        return self.data_len
    
class DomainNetSketchDataset(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(
        self,
        transform,
        csv_path,
        target_transform=None,
        custom_targets=None,
        path_prefix="",
        max_shots=0,
        seed=0,
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
        path_prefix = path_prefix or os.environ.get("TMPDIR", "")
        if not path_prefix:
            raise ValueError("Sketch requires path_prefix or TMPDIR")
        self.data_dir = os.path.join(path_prefix, "datasets/domain_sketch")
        if("test" in csv_path):
            self.dataset = pd.read_csv(os.path.join(self.data_dir, "sketch_test.txt"), sep=' ', header=None, names=['file', 'target'])
        else:
            self.dataset = pd.read_csv(os.path.join(self.data_dir, "sketch_train.txt"), sep=' ', header=None, names=['file', 'target'])

        is_train_split = "test" not in csv_path
        min_label = self.dataset['target'].min()
        print(f"min_label: {min_label}")
        num_classes = self.dataset['target'].nunique()
        print(f"Number of classes: {num_classes}")
        self.dataset = self.subsample_train_split(
            self.dataset,
            is_train_split=is_train_split,
            max_samples=max_shots,
            seed=seed,
        )

        # Calculate len
        self.data_len = len(self.dataset)

        self.path_prefix = path_prefix
        # custom targets to replace CATEGORIES
        self.custom_targets = custom_targets

    def subsample_train_split(self, dataset_split, is_train_split, max_samples, seed):
        # Only downsample training data; validation/test stay unchanged.
        if not is_train_split:
            return dataset_split

        if max_samples is None or int(max_samples) <= 0:
            logger.info("few-shot disabled: using full training split")
            return dataset_split

        shots_per_class = int(max_samples)
        label_column = "label" if "label" in dataset_split else "target"
        labels = dataset_split[label_column]
        total_samples = len(labels)

        label_to_indices = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(idx)

        rng = random.Random(seed)
        selected_indices = []
        per_class_kept = {}
        for label, indices in label_to_indices.items():
            rng.shuffle(indices)
            keep = min(shots_per_class, len(indices))
            per_class_kept[label] = keep
            selected_indices.extend(indices[:keep])

        rng.shuffle(selected_indices)

        underfull_classes = sum(
            1 for label, indices in label_to_indices.items() if len(indices) < shots_per_class
        )
        logger.info(
            "HF few-shot applied: kept %d samples across %d classes (shots_per_class=%d, underfull_classes=%d, total_before=%d)",
            len(selected_indices),
            len(label_to_indices),
            shots_per_class,
            underfull_classes,
            total_samples,
        )
        if hasattr(dataset_split, "select"):
            return dataset_split.select(selected_indices)

        if isinstance(dataset_split, pd.DataFrame):
            return dataset_split.iloc[selected_indices].reset_index(drop=True)

        raise TypeError(f"Unsupported dataset split type for few-shot subsampling: {type(dataset_split)!r}")
    
    def get_targets(self):
        return np.array(self.dataset['target'].values, dtype=np.int64)

    def __getitem__(self, index):
        sample_path, target = self.dataset.iloc[int(index)][['file', 'target']]

        img = Image.open(os.path.join(self.data_dir, sample_path)).convert("RGB")
        img_tensor = self.transforms(img)

        return img_tensor, int(target)

    def __len__(self):
        return self.data_len

class ViperWeatherDataset(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(
        self,
        transform,
        csv_path,
        target_transform=None,
        custom_targets=None,
        path_prefix="",
        max_shots=0,
        seed=0,
    ):
     
        super().__init__(in_c=3)
        self.transforms = transform

        path_prefix = os.environ.get("TMPDIR")

        if("train" in csv_path):
            self.data_dir = os.path.join(path_prefix, "viper/train")
        else:
            self.data_dir = os.path.join(path_prefix, "viper/val")

        self.image_folder = os.path.join(self.data_dir, "img")
        self.weather_csv = os.path.join(self.data_dir, "weather/weather.txt")
        self.weather_df = pd.read_csv(
            self.weather_csv, sep=',', header=None,
            names=['scene', 'weather'], dtype={'scene': str}
        )
        self.weather_df["weather"] = self.weather_df["weather"].astype("category").cat.codes

        self.viper_df = pd.DataFrame(columns=["image_location", 'weather'])
        rows = []
        for root, dirs, files in os.walk(self.image_folder):
            dirs.sort()
            files.sort()
            for filename in files:
                if not filename.lower().endswith((".png", ".jpg", ".bmp")):
                    continue
                scene = filename[1:3]
                rows.append({
                    "image_location": os.path.join(root, filename),
                    "weather": self.weather_df[self.weather_df.scene == scene].weather.iloc[0],
                })
        self.viper_df = pd.DataFrame(rows, columns=["image_location", "weather"])

        num_classes = self.viper_df['weather'].nunique()
        print(f"Number of classes: {num_classes}")
        self.data_len = len(self.viper_df)

    def get_targets(self):
        return np.array(self.viper_df['weather'].values, dtype=np.int64)

    def __getitem__(self, index):
        sample_path, target = self.viper_df.iloc[int(index)][['image_location', 'weather']]

        img = Image.open(os.path.join(self.data_dir, sample_path)).convert("RGB")
        img_tensor = self.transforms(img)

        return img_tensor, int(target)

    def __len__(self):
        return self.data_len


class BDDDataset(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(
        self,
        transform,
        csv_path,
        target_transform=None,
        custom_targets=None,
        path_prefix="",
        max_shots=0,
        seed=0,
        dataset_paths=None,
    ):
     
        super().__init__(in_c=3)
        self.transforms = transform

        path_prefix = path_prefix or (dataset_paths or {}).get("bdd_root", "")
        if not path_prefix:
            raise ValueError("BDD requires path_prefix")

        if("train" in csv_path):
            self.data_dir = os.path.join(path_prefix, "100k/train")
        else:
            self.data_dir = os.path.join(path_prefix, "100k/val")

        self.image_folder = self.data_dir 

        self.rows = []
        for root, dirs, files in os.walk(self.image_folder):
            dirs.sort()
            files.sort()
            for filename in files:
                if not filename.lower().endswith((".png", ".jpg", ".bmp")):
                    continue
                self.rows.append(os.path.join(root, filename))

        self.data_len = len(self.rows)

    def get_targets(self):
        return np.array([1])

    def __getitem__(self, index):
        sample_path = self.rows[index]
        try:
            img = Image.open(sample_path).convert("RGB")
            img_tensor = self.transforms(img)
            return img_tensor, 1
        except (UnidentifiedImageError, OSError) as e:
            # OSError also covers truncated/partial files
            print(f"Skipping corrupt image {sample_path}: {e}")
            return self.__getitem__(random.randint(0, self.data_len - 1))

    def __len__(self):
        return self.data_len

class ViperSegmentationDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        classes_csv,
        crop_size=224,
        augment=False,
        transform=None,
        target_transform=None,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.crop_size = crop_size

        # ------------------------------------------------------------------
        # RGB -> class mapping
        # ------------------------------------------------------------------
        df = pd.read_csv(classes_csv)

        self.rgb_to_class = {
            (row["red"], row["green"], row["blue"]): row["id"]
            for _, row in df.iterrows()
        }

       
        self.imgs_folder_paths  = sorted(os.listdir("train/img"))
        self.images = []
        for img_folder_path in self.imgs_folder_paths:
            folder_path = os.path.join(
                "train_cls_00-77_0/train/cls",
                img_folder_path
            )
            image_paths_in_dir = sorted([
                os.path.join(folder_path, img_name)
                for img_name in os.listdir(folder_path)
            ])
            self.images.extend(image_paths_in_dir)

        self.mask_folder_paths  = sorted(os.listdir("train_cls_00-77_0/train/cls"))
        self.masks = []
        for mask_folder_path in self.mask_folder_paths:
            folder_path = os.path.join(
                "train_cls_00-77_0/train/cls",
                mask_folder_path
            )
            mask_paths_in_dir = sorted([
                os.path.join(folder_path, mask_name)
                for mask_name in os.listdir(folder_path)
            ])
            self.masks.extend(mask_paths_in_dir)

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def rgb_mask_to_class_mask(self, mask_rgb):
        """
        mask_rgb: numpy array (H, W, 3)
        returns: numpy array (H, W)
        """

        h, w = mask_rgb.shape[:2]

        mask_class = np.zeros((h, w), dtype=np.uint8)

        for color, class_idx in self.rgb_to_class.items():
            matches = np.all(mask_rgb == color, axis=-1)
            mask_class[matches] = class_idx

        return mask_class

    

    def __getitem__(self, idx):

        img_path = self.images[idx]
        mask_path = self.masks[idx]
        print(img_path)
        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        if self.augment:
            image, mask_rgb = self.transform(image, mask_rgb)

        mask_rgb_np = np.array(mask_rgb)
        mask_class = self.rgb_mask_to_class_mask(mask_rgb_np)

        image = TF.to_tensor(image)

        # Normalize image
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask = torch.as_tensor(mask_class, dtype=torch.long)

        return image, mask

class CustomDatasetFromImages(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform, target_transform=None, custom_targets=None, path_prefix=""):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.target_transform = target_transform

        self.path_prefix = path_prefix

        # custom targets to replace CATEGORIES
        self.custom_targets = custom_targets

    def get_targets(self):
        if self.custom_targets is None:
            return super().get_targets()
        return np.array([i for i, _ in enumerate(self.custom_targets)])

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        single_image_name = self.path_prefix + single_image_name
        single_image_name = single_image_name.replace("//", "/")
        img_as_img = Image.open(single_image_name)
        # img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len



#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ["value", "one-hot"]
    mean = [
        1370.19151926,
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
        14.77112979,
        1732.16362238,
        1247.91870117,
    ]
    std = [
        633.15169573,
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        472.37967789,
        14.3114637,
        1310.36996126,
        1087.6020813,
    ]

    def __init__(
        self,
        csv_path: str,
        transform: Any,
        target_transform=None,
        years: Optional[List[int]] = [*range(2000, 2021)],
        categories: Optional[List[str]] = None,
        label_type: str = "value",
        masked_bands: Optional[List[int]] = None,
        dropped_bands: Optional[List[int]] = None,
        dataset_mean: Optional[List[int]] = None,
        dataset_std: Optional[List[int]] = None,
    ):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path).sort_values(["category", "location_id", "timestamp"])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df["year"] = [int(timestamp.split("-")[0]) for timestamp in self.df["timestamp"]]
            self.df = self.df[self.df["year"].isin(years)]

        # self.indices = self.df.index.unique().to_numpy()

        self.transform = transform
        self.target_transform = target_transform

        if label_type not in self.label_types:
            raise ValueError(
                f"FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:",
                ", ".join(self.label_types),
            )
        self.label_type = label_type

        self.masked_bands = np.array(masked_bands, dtype=np.int64) if masked_bands is not None else None
        self.dropped_bands = np.array(dropped_bands, dtype=np.int64) if dropped_bands is not None else None
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

        # Do this do avoid memory leak?
        self.mean = np.array(dataset_mean if dataset_mean else self.mean)
        self.std = np.array(dataset_std if dataset_std else self.std)

        # # Not sure why, but new version of pytorch is running into memory leak
        # self.image_paths = np.asarray(self.df['image_path'].values)
        # self.labels = np.asarray(self.df['category'].values)
        # del self.df

    @staticmethod
    def update_mean_std_with_dropped_bands(dropped_bands):
        mean = np.array(SentinelIndividualImageDataset.mean)
        std = np.array(SentinelIndividualImageDataset.std)
        keep_idxs = [i for i in range(mean.shape[0]) if i not in dropped_bands]
        return mean[keep_idxs].tolist(), std[keep_idxs].tolist()

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):

        # with rasterio.open(img_path) as data:
        #     # img = data.read(
        #     #     out_shape=(data.count, self.resize, self.resize),
        #     #     resampling=Resampling.bilinear
        #     # )
        #     img = data.read()  # (c, h, w)
        # img = img.transpose(1, 2, 0)  # (h, w, c)

        img = imread(img_path)  # (h, w, c)

        return img.astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]
        # selection = {'image_path': self.image_paths[idx], 'category': self.labels[idx]}

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection["image_path"])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = self.mean[self.masked_bands]

        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(images.shape[-1]) if i not in self.dropped_bands]
            images = images[:, :, keep_idxs]

        labels = self.categories.index(selection["category"])

        img_as_tensor = self.transform(images)  # (c, h, w)

        # sample = {
        #     'images': images,
        #     'labels': labels,
        #     'image_ids': selection['image_id'],
        #     'timestamps': selection['timestamp']
        # }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                (
                    transforms.RandomResizedCrop(
                        input_size, scale=(0.2, 1.0), interpolation=interpol_mode, antialias=None
                    )
                    if use_antialias_key
                    else transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode)
                ),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(
                size, interpolation=interpol_mode, antialias=None
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


#########################################################
# VISDA DEFINITIONS
#########################################################


class Visda2017(Dataset):
    def __init__(self, file_path, path_prefix="", transform=None, target_transform=None, split="train"):
        df = pd.read_csv(file_path, delimiter=" ")
        self.image_paths = [f"{path_prefix}/{path}" for path in df.iloc[:, 0] if "train" in path]
        if split == "train+val":
            print("Pre-training on train+val")
            self.image_paths += [f"{path_prefix}/{path}" for path in df.iloc[:, 0] if "validation" in path]
        elif split == "val":
            print("Validation dataset")
            self.image_paths = [f"{path_prefix}/{path}" for path in df.iloc[:, 0] if "validation" in path]
        self.labels = df.iloc[:, 1]
        self.transform = transform
        self.n_classes = 12

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


#########################################################
# IMAGENET DEFINITIONS
#########################################################


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError


_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class ImageNet(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "ImageNet.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "ImageNet.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._split == _Split.TEST:
            assert False, "Class IDs are not available in TEST split"
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._split == _Split.TEST:
            assert False, "Class names are not available in TEST split"
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        actual_index = entries[index]["actual_index"]

        class_id = self.get_class_id(index)

        image_relpath = self.split.get_image_relpath(actual_index, class_id)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return None if self.split == _Split.TEST else entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return None if self.split == _Split.TEST else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return None if self.split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)

    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        labels_full_path = os.path.join(self.root, labels_path)
        labels = []

        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

        return labels

    def _dump_entries(self) -> None:
        split = self.split
        if split == ImageNet.Split.TEST:
            dataset = None
            sample_count = split.length
            max_class_id_length, max_class_name_length = 0, 0
        else:
            labels_path = "labels.txt"
            logger.info(f'loading labels from "{labels_path}"')
            labels = self._load_labels(labels_path)

            # NOTE: Using torchvision ImageFolder for consistency
            from torchvision.datasets import ImageFolder

            dataset_root = os.path.join(self.root, split.get_dirname())
            dataset = ImageFolder(dataset_root)
            sample_count = len(dataset)
            max_class_id_length, max_class_name_length = -1, -1
            for sample in dataset.samples:
                _, class_index = sample
                class_id, class_name = labels[class_index]
                max_class_id_length = max(len(class_id), max_class_id_length)
                max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        if split == ImageNet.Split.TEST:
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                actual_index = index + 1
                class_index = np.uint32(-1)
                class_id, class_name = "", ""
                entries_array[index] = (actual_index, class_index, class_id, class_name)
        else:
            class_names = {class_id: class_name for class_id, class_name in labels}

            assert dataset
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                image_full_path, class_index = dataset.samples[index]
                image_relpath = os.path.relpath(image_full_path, self.root)
                class_id, actual_index = split.parse_image_relpath(image_relpath)
                class_name = class_names[class_id]
                entries_array[index] = (actual_index, class_index, class_id, class_name)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
        split = self.split
        if split == ImageNet.Split.TEST:
            return

        entries_array = self._load_extra(self._entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()
