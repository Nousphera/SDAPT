import os
import ssl
import random
import numpy as np
import pandas as pd
import torch
from data.datasets.HPA_dataset import SelfNormalize
from data.datasets.dataset_abstract import DinoDataset
from models.dinov2.utils.augmentations import DataAugmentationDINO, DataAugmentationDINOCheXPert, DataAugmentationDINOQuickdraw, DataAugmentationDINOSketch
from models.dinov2.utils.data_transforms import GaussianBlur
from models.dinov2.utils.masking import MaskingGenerator
from wilds import get_dataset
from models.dinov2.utils.transforms import get_transform
from torchvision import transforms

def get_dinov2_pretrain_vit_transforms_original(train: bool):
    """
    Returns standard transforms for ViT models.
    Train: Resize -> RandomFlip -> ToTensor -> Normalize
    Val:   Resize -> ToTensor -> Normalize
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ## original is 0.05
    global_crops_scale = [0.32, 1.0]
    local_crops_scale  = [0.05, 0.32]
    local_crops_number = 8
    # global_crops_scale = [0.4, 1.0]
    # local_crops_scale  = [0.15, 0.4]
    # local_crops_number = 8


    if train:
        data_transform = DataAugmentationDINO(
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=98,
    )
        return data_transform
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

def get_dinov2_pretrain_vit_transforms_sketch(train: bool):
    """
    Returns standard transforms for ViT models.
    Train: Resize -> RandomFlip -> ToTensor -> Normalize
    Val:   Resize -> ToTensor -> Normalize
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ## original is 0.05
    global_crops_scale = [0.5, 1.0]
    local_crops_scale  = [0.2, 0.5]
    local_crops_number = 8
    # global_crops_scale = [0.4, 1.0]
    # local_crops_scale  = [0.15, 0.4]
    # local_crops_number = 8


    if train:
        data_transform = DataAugmentationDINOSketch(
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=98,
    )
        return data_transform
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

def get_dinov2_pretrain_vit_transforms(train: bool, dataset: str):
    if dataset == "domain_quickdraw":
        return get_dinov2_pretrain_vit_transforms_quickdraw(train)
    elif dataset == "domain_sketch":
        return get_dinov2_pretrain_vit_transforms_sketch(train)
    elif dataset == "chexpert":
        return get_dinov2_pretrain_vit_transforms_chexpert(train)
    else:
        return get_dinov2_pretrain_vit_transforms_original(train)

def get_mocov2_pretrain_vit_transforms(train: bool):
    """
    Returns standard transforms for ViT models.
    Train: Resize -> RandomFlip -> ToTensor -> Normalize
    Val:   Resize -> ToTensor -> Normalize
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
def get_dinov2_pretrain_vit_transforms_quickdraw(train: bool):
    """
    Returns standard transforms for ViT models.
    Train: Resize -> RandomFlip -> ToTensor -> Normalize
    Val:   Resize -> ToTensor -> Normalize
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ## original is 0.05
    local_crops_scale = [0.25, 0.6]
    global_crops_scale = [0.5, 1.0]
    local_crops_number = 8
    # global_crops_scale = [0.4, 1.0]
    # local_crops_scale  = [0.15, 0.4]
    # local_crops_number = 8


    if train:
        data_transform = DataAugmentationDINOQuickdraw(
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=98,
    )
        return data_transform
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

def get_dinov2_pretrain_vit_transforms_chexpert(train: bool):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    global_crops_scale = [0.50, 1.0]
    local_crops_scale  = [0.20, 0.50]
    local_crops_number = 8

    if train:
        return DataAugmentationDINOCheXPert(
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=518,
            local_crops_size=196,
        )
    else:
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

def dino_collate_fn(
    samples_list,
    mask_probability=0.5,
    mask_ratio_tuple=(0.1, 0.5),
    mask_generator=None,
    n_tokens=None,
    img_size=224,
    patch_size=14,
    dtype=torch.float32,
):
        n_global_crops = len(samples_list[0][0]["global_crops"])
        n_local_crops = len(samples_list[0][0]["local_crops"])
        grid_size = img_size // patch_size
        if n_tokens is None:
            n_tokens = grid_size * grid_size
        if mask_generator is None:
            mask_generator = MaskingGenerator(
                input_size=(grid_size, grid_size),
                max_num_patches=int(mask_ratio_tuple[-1] * n_tokens),
            )
        collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

        collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

        B = len(collated_global_crops)
        N = n_tokens
        n_samples_masked = int(B * mask_probability)
        probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(mask_generator(0)))

        random.shuffle(masks_list)

        collated_masks = torch.stack(masks_list).flatten(1)

        return {
            "collated_global_crops": collated_global_crops.to(dtype),
            "collated_local_crops": collated_local_crops.to(dtype),
            "collated_masks": collated_masks,
            "upperbound": upperbound,
        }

class FMoWFromWILDS(DinoDataset):


    def __init__(
        self,
        csv_path,
        transform,
        inc_unlabeled=False,
        target_transform=None,
        custom_targets=None,
        train = True,
        path_prefix="",
        subsample_fraction=1.0,
        max_shots=0,
        seed=0,
        dataset_paths=None,
    ):
        super().__init__(in_c=3)
        self.transforms = transform
        self.custom_targets = custom_targets

        root_dir = (dataset_paths or {}).get("wilds_root", "")
        if not root_dir:
            raise ValueError("fMoW WILDS requires wilds_root in dataset_paths")
        ssl._create_default_https_context = ssl._create_unverified_context
        full_dataset = get_dataset(dataset="fmow", download=True, root_dir=root_dir)

        # WILDS fMoW splits: 'train', 'id_val', 'id_test', 'val' (OOD), 'test' (OOD)
        # Your old code keyed off "test" in csv_path, so mirror that:
        split = "train" if train else "test"   # use "id_test" for in-distribution
        is_train_split = "test" not in csv_path

        subset = full_dataset.get_subset(split)
        y_array = subset.y_array  # torch tensor of int labels (0..61)

        self.dataset = subset
        self.y_array = y_array

        self.data_len = len(self.dataset)
        self.path_prefix = root_dir

        # must be last: it re-slices self.dataset / self.y_array / self.custom_targets
        if subsample_fraction < 1.0:
            self.subsample_dataset(subsample_fraction, seed=seed)   # reuse existing seed param
        if max_shots > 0:
            self.subsample_shots(max_shots, seed=seed)


    def _csv_samples_for_subset(self):
        """CSV row indices for the current WILDS subset, in y_array order."""
        base = getattr(self.dataset, "dataset", self.dataset)
        indices = getattr(self.dataset, "indices", None)

        base2csv = getattr(base, "full_idxs", None)
        if base2csv is None:  # fallback: rebuild the seq-filter mapping ourselves
            base2csv = np.where(base.metadata["split"].to_numpy() != "seq")[0]

        if indices is None:
            csv_samples = np.asarray(base2csv)
        else:
            csv_samples = np.asarray(base2csv)[np.asarray(indices)]
        return base, csv_samples

    def subsample_dataset(self, fraction, seed=0):
        """Keep `fraction` of the samples of *every* class, then restrict the
        dataset to the selected indices.

        fMoW is single-label, so each class's samples are disjoint: this is
        ordinary stratified subsampling. Doing it per class (rather than one
        global random draw) guarantees rare classes keep ~`fraction` of their
        examples instead of being thinned out or dropped. Pass e.g. 0.1 for 10%.
        """
        assert 0.0 < fraction <= 1.0
        rng = random.Random(seed)

        targets = self.get_targets()  # int label per current sample, aligned to self.dataset

        selected_pos = []
        for cls in np.unique(targets):
            pos = np.where(targets == cls)[0]
            n_keep = round(fraction * len(pos))
            if n_keep:
                selected_pos.extend(rng.sample(list(pos), n_keep))

        selected_pos = np.array(sorted(selected_pos), dtype=np.int64)

        # Re-slice the WILDS subset via its own base indices so that y_array,
        # metadata_array, and full_idxs all stay aligned. Reusing the subset's own
        # class avoids depending on a WILDSSubset import path.
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

    def get_country_codes(self):
        """One country_code per image, aligned to self.y_array (NaNs kept)."""
        base, csv_samples = self._csv_samples_for_subset()
        return base.metadata["country_code"].to_numpy()[csv_samples]

    def get_unique_microdomains(self, micro_domain):
        base = getattr(self.dataset, "dataset", self.dataset)
        fields = list(base.metadata_fields)  

        if micro_domain in fields:
            arr = self.dataset.metadata_array[:, fields.index(micro_domain)]
            return np.unique(arr.cpu().numpy())

        _, csv_samples = self._csv_samples_for_subset()
        col = "country_code" 
        values = base.metadata[col].to_numpy()[csv_samples]
        if values.dtype == object:
            values = values[~pd.isna(values)]
        return np.unique(values)

    def get_samples_by_microdomain_value(self, micro_domain, value):
        base = getattr(self.dataset, "dataset", self.dataset)
        fields = list(base.metadata_fields)   

        # region / year: already in the subset-aligned metadata_array
        if micro_domain in fields:
            col = fields.index(micro_domain)
            arr = self.dataset.metadata_array[:, col].cpu().numpy()
            value = self._resolve_metadata_value(base, micro_domain, value)
            return np.where(arr == value)[0]

        _, csv_samples = self._csv_samples_for_subset()
        col_name = "country_code"
        values = base.metadata[col_name].to_numpy()[csv_samples]
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
