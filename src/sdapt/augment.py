"""Domain-specific DINO multi-crop transforms from the SDAPT experiment code."""
from __future__ import annotations

import numpy as np
import torch
from PIL import ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as F

from .data import DATASETS

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, image):
        if torch.rand(()).item() < self.probability:
            return image.filter(ImageFilter.GaussianBlur(float(torch.empty(()).uniform_(0.1, 2.0))))
        return image


class ContentAwareCrop(T.RandomResizedCrop):
    """Reject blank QuickDraw crops; retain the full drawing if retries fail."""
    def forward(self, image):
        for _ in range(10):
            top, left, height, width = self.get_params(image, self.scale, self.ratio)
            crop = image.crop((left, top, left + width, top + height)).convert("L")
            if (np.asarray(crop) < 128).mean() >= 0.02:
                return F.resized_crop(image, top, left, height, width, self.size, self.interpolation, antialias=True)
        return F.resize(image, self.size, self.interpolation, antialias=True)


class StrokeWidthJitter:
    def __call__(self, image):
        if torch.rand(()).item() >= 0.5:
            return image
        size = 3 if torch.rand(()).item() < 0.5 else 5
        operation = ImageFilter.MinFilter if torch.rand(()).item() < 0.5 else ImageFilter.MaxFilter
        return image.filter(operation(size))


class MultiCrop:
    """Return two global views followed by local views, all normalized RGB tensors."""
    def __init__(self, dataset, local_crops=8, patch_size=14):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        if local_crops < 0:
            raise ValueError("local_crops must be nonnegative")
        self.local_crops = local_crops
        global_size, local_size = crop_sizes(dataset, patch_size)
        self.patch_size = patch_size
        self.global_size, self.local_size = global_size, local_size
        global_scale = (0.32, 1.0) if dataset == "fmow" else (0.5, 1.0)
        local_scale = {"fmow": (0.05, 0.32), "quickdraw": (0.25, 0.6), "chexpert": (0.2, 0.5)}[dataset]
        normalizer = [T.ToTensor(), T.Normalize(MEAN, STD)]

        def geometry(size, scale):
            crop = ContentAwareCrop if dataset == "quickdraw" else T.RandomResizedCrop
            operations = [crop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip()]
            if dataset == "quickdraw":
                operations += [
                    T.RandomApply([T.RandomAffine(12, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8, fill=255)], p=0.7),
                    T.RandomApply([T.ElasticTransform(alpha=20.0, sigma=4.0, fill=255)], p=0.3),
                ]
            return operations

        def view(size, scale, index):
            operations = geometry(size, scale)
            if dataset == "quickdraw":
                operations += [StrokeWidthJitter(), T.Grayscale(num_output_channels=3)]
            else:
                # Saturation/hue shifts carry no radiographic information.
                jitter = T.ColorJitter(0.4, 0.4, 0.2 if dataset == "fmow" else 0, 0.1 if dataset == "fmow" else 0)
                operations += [T.RandomApply([jitter], p=0.8)]
                if dataset == "fmow":
                    operations += [T.RandomGrayscale(p=0.2)]
                blur = (1.0 if dataset == "fmow" else 0.5) if index == 0 else 0.1 if index == 1 else 0.5
                operations += [GaussianBlur(blur)]
                if dataset == "fmow" and index == 1:
                    operations += [T.RandomSolarize(128, p=0.2)]
            return T.Compose(operations + normalizer)

        self.global_views = [view(global_size, global_scale, i) for i in (0, 1)]
        self.local_view = view(local_size, local_scale, 2)

    def __call__(self, image):
        image = image.convert("RGB")
        return [transform(image) for transform in self.global_views] + [self.local_view(image) for _ in range(self.local_crops)]


def crop_sizes(dataset, patch_size=14):
    """Floor paper crop sizes to whole patches without increasing field resolution.

    Patch-14 keeps the paper's 224/98 and radiographic 518/196 crops.
    Patch-8 and patch-16 use 224/96 and radiographic 512/192 instead.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size not in (8, 14, 16):
        raise ValueError("patch_size must be 8, 14 or 16")
    sizes = (518, 196) if dataset == "chexpert" else (224, 98)
    return tuple(size // patch_size * patch_size for size in sizes)


def eval_transform(dataset, patch_size=14):
    """Original whole-image 224x224 downstream preprocessing, without augmentation.

    The original CheXpert downstream transform also used 224; 518/196 are its
    DAPT multi-crop sizes, not a claim about downstream evaluation resolution.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    crop_sizes(dataset, patch_size)  # Validate the selected model's patch geometry.
    interpolation = T.InterpolationMode.BICUBIC if dataset == "chexpert" else T.InterpolationMode.BILINEAR
    return T.Compose([T.Resize((224, 224), interpolation=interpolation), T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform(dataset, patch_size=14):
    """The released downstream protocol uses the same whole-image preprocessing."""
    return eval_transform(dataset, patch_size=patch_size)
