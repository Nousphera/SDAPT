# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import logging
import random

import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter
from .transforms import (
    GaussianBlur,
    MaybeToTensor,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        dataset_mean=None,
        dataset_std=None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # orginal p=0.5
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # orginal p=0.5
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                (
                    make_normalize_transform(dataset_mean, dataset_std)
                    if dataset_mean is not None and dataset_std is not None
                    else make_normalize_transform()
                ),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class ContentAwareRandomResizedCrop(transforms.RandomResizedCrop):
    """RandomResizedCrop that reject-resamples until the crop actually contains ink.
    Solves the 'small crop lands on empty canvas -> degenerate white==white pair' problem."""
    def __init__(self, *args, min_ink_fraction=0.02, max_attempts=10, ink_is_dark=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_ink_fraction = min_ink_fraction
        self.max_attempts = max_attempts
        self.ink_is_dark = ink_is_dark

    def _ink_fraction(self, img, i, j, h, w):
        crop = img.crop((j, i, j + w, i + h)).convert("L")
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        ink = (arr < 0.5) if self.ink_is_dark else (arr > 0.5)
        return float(ink.mean())

    def forward(self, img):
        i = j = h = w = None
        for _ in range(self.max_attempts):
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            if self._ink_fraction(img, i, j, h, w) >= self.min_ink_fraction:
                break
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=True)


class StrokeWidthJitter:
    """Randomly thicken/thin strokes via morphological min/max filtering. Dependency-free (PIL)."""
    def __init__(self, p=0.5, ink_is_dark=True):
        self.p = p
        self.ink_is_dark = ink_is_dark

    def __call__(self, img):
        if random.random() > self.p:
            return img
        size = random.choice([3, 5])
        thicken = random.random() < 0.5
        # dark ink: MinFilter thickens (dark grows), MaxFilter thins
        if thicken == self.ink_is_dark:
            return img.filter(ImageFilter.MinFilter(size))
        return img.filter(ImageFilter.MaxFilter(size))


class DataAugmentationDINOQuickdraw(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        dataset_mean=None,
        dataset_std=None,
        ink_is_dark=True,
        min_ink_fraction=0.02,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # ---- geometric (content-aware crop + flip + affine + mild elastic) ----
        geo_extra = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),  # ok for most classes; watch chirality-sensitive ones
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=12, translate=(0.08, 0.08),
                                             scale=(0.9, 1.1), shear=8, fill=255)],  # fill=255 = white bg
                    p=0.7,
                ),
                transforms.RandomApply(
                    [transforms.ElasticTransform(alpha=20.0, sigma=4.0, fill=255)],
                    p=0.3,
                ),
            ]
        )

        self.geometric_augmentation_global = transforms.Compose(
            [
                ContentAwareRandomResizedCrop(
                    global_crops_size, scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    ink_is_dark=ink_is_dark, min_ink_fraction=min_ink_fraction,
                ),
                geo_extra,
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                ContentAwareRandomResizedCrop(
                    local_crops_size, scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    ink_is_dark=ink_is_dark, min_ink_fraction=min_ink_fraction,
                ),
                geo_extra,
            ]
        )

        # ---- domain-specific stroke aug replaces color jitter ----
        stroke = StrokeWidthJitter(p=0.5, ink_is_dark=ink_is_dark)

        # ---- normalization (force 3 channels, then normalize) ----
        self.normalize = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),  # guarantees 3ch for DINOv2
                transforms.ToTensor(),
                (
                    make_normalize_transform(dataset_mean, dataset_std)
                    if dataset_mean is not None and dataset_std is not None
                    else make_normalize_transform()
                ),
            ]
        )

        # No color jitter, no grayscale-jitter, no solarize, no heavy blur.
        self.global_transfo1 = transforms.Compose([stroke, self.normalize])
        self.global_transfo2 = transforms.Compose([stroke, self.normalize])
        self.local_transfo = transforms.Compose([stroke, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class DataAugmentationDINOSketch(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        dataset_mean=None,
        dataset_std=None,
        ink_is_dark=True,
        min_ink_fraction=0.02,   # can go lower than Quickdraw: Sketch is denser, fewer blank crops
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters (SKETCH):")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # ---- geometric: content-aware crop + flip + affine + GENTLER elastic ----
        # vs Quickdraw: elastic softened (alpha 20->14, p 0.3->0.2) because fine
        # hatching/shading in Sketch fragments into dashes under strong warps.
        geo_extra = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),  # watch chirality-sensitive classes
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=12, translate=(0.08, 0.08),
                                             scale=(0.9, 1.1), shear=8, fill=255)],
                    p=0.7,
                ),
                transforms.RandomApply(
                    [transforms.ElasticTransform(alpha=14.0, sigma=4.0, fill=255)],
                    p=0.2,
                ),
            ]
        )

        self.geometric_augmentation_global = transforms.Compose(
            [
                ContentAwareRandomResizedCrop(
                    global_crops_size, scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    ink_is_dark=ink_is_dark, min_ink_fraction=min_ink_fraction,
                ),
                geo_extra,
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                ContentAwareRandomResizedCrop(
                    local_crops_size, scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    ink_is_dark=ink_is_dark, min_ink_fraction=min_ink_fraction,
                ),
                geo_extra,
            ]
        )

        # ---- stroke-width jitter: still relevant, keep it ----
        stroke = StrokeWidthJitter(p=0.5, ink_is_dark=ink_is_dark)

        # ---- photometric: BACK from DINOv2, but sketch-appropriate ----
        # brightness/contrast are REAL variation in Sketch (pencil pressure, scan
        # exposure, gray shading). saturation/hue stay 0 (no color signal).
        photometric = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0)],
            p=0.5,
        )

        # ---- normalization (force 3 channels, then normalize) ----
        self.normalize = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),  # guarantees 3ch for DINOv2
                transforms.ToTensor(),
                (
                    make_normalize_transform(dataset_mean, dataset_std)
                    if dataset_mean is not None and dataset_std is not None
                    else make_normalize_transform()
                ),
            ]
        )

        # Restore the DINOv2 two-global-view asymmetry in MILD form.
        # Sketch has enough redundant texture (hatching/shading) that a light blur
        # adds scale-robustness without erasing the signal the way it would on Quickdraw.
        # global1: a bit more blur ; global2: little blur.
        # Solarize left OFF by default (still marginal on mostly-white images), but
        # Sketch's midtones make it less degenerate than Quickdraw — toggle below to test.
        self.global_transfo1 = transforms.Compose(
            [stroke, photometric, GaussianBlur(p=0.5), self.normalize]
        )
        self.global_transfo2 = transforms.Compose(
            [
                stroke,
                photometric,
                GaussianBlur(p=0.1),
                # transforms.RandomSolarize(threshold=128, p=0.1),  # optional, ablate this
                self.normalize,
            ]
        )
        self.local_transfo = transforms.Compose(
            [stroke, photometric, GaussianBlur(p=0.2), self.normalize]
        )

    def __call__(self, image):
        output = {}

        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class RandomGamma(object):
    """
    Gamma jitter, as a stand-in for scanner / post-processing variation in CXR.
 
    Acquisition variability across vendors, exposure settings and vendor post-processing
    behaves more like a non-linear window-level shift than a linear brightness scale, so
    this models it better than ColorJitter(brightness=...) on its own.
    gamma < 1 brightens, gamma > 1 darkens.
    """
 
    def __init__(self, gamma_range=(0.7, 1.4), p=0.8):
        self.gamma_range = gamma_range
        self.p = p
 
    def __call__(self, img):
        if random.random() > self.p:
            return img
        gamma = random.uniform(*self.gamma_range)
        return TF.adjust_gamma(img, gamma, gain=1.0)
 
    def __repr__(self):
        return f"{self.__class__.__name__}(gamma_range={self.gamma_range}, p={self.p})"
 
 
class DataAugmentationDINOCheXPert(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=0.5)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                MaybeToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output