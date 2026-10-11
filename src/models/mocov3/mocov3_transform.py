"""Two-view augmentations + collate function for MoCo split learning.

Replaces the DINOv2 global/local multi-crop pipeline: MoCo needs exactly two
views per image, both at full resolution.

Asymmetric augmentation (MoCo v3): view 1 always gets blur, view 2 gets blur
rarely but gets solarisation instead.

Pipeline order matters:
    PIL geometric -> PIL colour -> ToTensor -> blur/solarise -> Normalize
Solarisation and blur run in *tensor* space so the threshold is in [0, 1]
rather than PIL's 0-255, and so the multi-band path can share the same ops.

fMoW-RGB yields 3-channel PIL images (``in_chans=3``). The ``in_chans > 3``
branch is for 4-band (RGB+NIR) variants, where torchvision's ColorJitter /
RandomGrayscale reject the extra channel and we substitute per-channel ops.
"""

import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from models.dinov2.utils.augmentations import ContentAwareRandomResizedCrop, StrokeWidthJitter

# --------------------------------------------------------------- multi-band ops
import random
 
import numpy as np
import torch
from torchvision import transforms
 
# Warm-starting from an ImageNet MoCo v3 checkpoint means keeping the input
# statistics the initialisation expects. Only swap to CheXpert-computed stats
# if you are training from scratch.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
 
 
class TensorCLAHE:
    """Contrast-limited adaptive histogram equalisation on a [0, 1] float tensor.
 
    Stands in for the vendor post-processing / windowing variation that CheXpert
    genuinely contains, and supplies the view asymmetry that Solarize used to
    provide in the reference MoCo v3 recipe.
    """
 
    def __init__(self, p=0.5, clip_limit=(1.0, 3.0), grid=8):
        self.p = p
        self.clip_limit = clip_limit
        self.grid = grid
 
    def __call__(self, x):
        if random.random() > self.p:
            return x
        try:
            import cv2
        except ImportError as exc:  # skimage.exposure.equalize_adapthist also works
            raise ImportError("TensorCLAHE needs opencv-python") from exc
 
        clip = random.uniform(*self.clip_limit)
        op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(self.grid, self.grid))
 
        arr = (x.clamp(0, 1).numpy() * 255).astype(np.uint8)  # (C, H, W)
        out = np.stack([op.apply(c) for c in arr], axis=0)
        return torch.from_numpy(out).float().div_(255.0)
 
 
class TensorGamma:
    """Random gamma on a [0, 1] float tensor.
 
    Monotonic, so it perturbs contrast without scrambling the density ordering
    the way Solarize does. Solarize is dropped entirely for radiographs:
    intensity is the semantic content, not a nuisance factor.
    """
 
    def __init__(self, p=0.5, gamma_range=(0.7, 1.4)):
        self.p = p
        self.gamma_range = gamma_range
 
    def __call__(self, x):
        if random.random() > self.p:
            return x
        return x.clamp(0, 1).pow(random.uniform(*self.gamma_range))
 
 
def _build_cxr_view(
    img_size,
    in_chans,
    blur_p,
    mean,
    std,
    crop_scale,
    input_is_pil,
    clahe_p=0.0,
    gamma_p=0.0,
    hflip_p=0.0,
    rotation_degrees=10,
    translate=(0.05, 0.05),
    scale_jitter=(0.95, 1.05),
    jitter_strength=0.2,
    blur_sigma=(0.1, 1.0),
):
    stages = [
        # scale=(0.2, 1.0) lets a view be pure shoulder or abdomen while the other
        # view holds the finding, which breaks the shared-content assumption.
        # Aspect ratio is also pinned: anatomy does not stretch 3/4 to 4/3.
        transforms.RandomResizedCrop(
            img_size,
            scale=crop_scale,
            ratio=(0.85, 1.18),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
    ]
 
    # No vertical flip and no right-angle rotation: a chest radiograph has a
    # canonical orientation. Only small positioning jitter is real variation.
    if rotation_degrees > 0:
        stages.append(
            transforms.RandomAffine(
                degrees=rotation_degrees,
                translate=translate,
                scale=scale_jitter,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
 
    # Off by default: hflip produces dextrocardia (~0.01% prevalence) and teaches
    # invariance to laterality, which matters for cardiomegaly, effusion side and
    # line/tube position. MoCo-CXR and REMEDIS both keep it, so ablate rather
    # than assume.
    if hflip_p > 0:
        stages.append(transforms.RandomHorizontalFlip(p=hflip_p))
 
    if in_chans == 3:
        stages.append(transforms.Grayscale(num_output_channels=3))
 
    # Hue and saturation are no-ops on grayscale, so only brightness/contrast
    # survive, and at 0.2 rather than 0.4: intensity encodes tissue density.
    # RandomGrayscale, ChannelJitter and ChannelDrop are all dropped as
    # meaningless on single-band input.
    stages.append(
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=jitter_strength, contrast=jitter_strength)],
            p=0.7,
        )
    )
 
    if input_is_pil:
        stages.append(transforms.ToTensor())
 
    if clahe_p > 0:
        stages.append(TensorCLAHE(p=clahe_p))
    if gamma_p > 0:
        stages.append(TensorGamma(p=gamma_p))
 
    # Interstitial markings, small nodules and pneumothorax lines are all
    # high-frequency; sigma is capped near 1.0 and p reduced from the 1.0 that
    # the reference recipe uses on view 1.
    stages.append(TensorGaussianBlur(p=blur_p, sigma=blur_sigma))  # noqa: F821
 
    if mean is not None and std is not None:
        stages.append(transforms.Normalize(mean=mean, std=std))
 
    return transforms.Compose(stages)
 
 
def get_moco_cxr_transforms(
    img_size=224,
    in_chans=3,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    crop_scale=(0.5, 1.0),
    input_is_pil=True,
    hflip_p=0.0,
):
    """MoCo v3 two-view transform for CheXpert domain-adaptive pretraining.
 
    View asymmetry is preserved (the reference recipe gets it from blur 1.0/0.1
    and solarize 0.0/0.2) but re-sourced: CLAHE on view 1, gamma on view 2.
    Without this the two branches collapse to nearly the same distribution once
    solarize is removed.
    """
    common = dict(
        img_size=img_size,
        in_chans=in_chans,
        mean=mean,
        std=std,
        crop_scale=crop_scale,
        input_is_pil=input_is_pil,
        hflip_p=hflip_p,
    )
    aug1 = _build_cxr_view(blur_p=0.5, clahe_p=0.6, gamma_p=0.0, **common)
    aug2 = _build_cxr_view(blur_p=0.1, clahe_p=0.0, gamma_p=0.6, **common)
    return TwoCropsTransform(aug1, aug2)  # noqa: F821

class ChannelJitter(torch.nn.Module):
    """Per-channel brightness/contrast jitter that works for any channel count."""

    def __init__(self, brightness=0.4, contrast=0.4, p=0.8):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        c = x.shape[0]
        b = 1.0 + (torch.rand(c, 1, 1, device=x.device) * 2 - 1) * self.brightness
        x = x * b
        mean = x.mean(dim=(1, 2), keepdim=True)
        ct = 1.0 + (torch.rand(c, 1, 1, device=x.device) * 2 - 1) * self.contrast
        return ((x - mean) * ct + mean).clamp_(0.0, 1.0)


class ChannelDrop(torch.nn.Module):
    """Stand-in for RandomGrayscale on multi-band imagery: collapse all channels
    to their mean with probability p."""

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return x.mean(dim=0, keepdim=True).expand_as(x).clone()


class TensorSolarize(torch.nn.Module):
    """Solarise a float tensor in [0, 1]. NOTE the threshold is 0.5, not 128 --
    torchvision's RandomSolarize on a PIL image expects 0-255 instead."""

    def __init__(self, threshold=0.5, p=0.2):
        super().__init__()
        self.threshold = threshold
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return torch.where(x >= self.threshold, 1.0 - x, x)


class TensorGaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size=9, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def forward(self, x):
        if self.p <= 0.0 or random.random() > self.p:
            return x
        s = random.uniform(*self.sigma)
        return TF.gaussian_blur(x, kernel_size=self.kernel_size, sigma=s)


# ---------------------------------------------------------------- pipelines


def _build_view(
    img_size,
    in_chans,
    blur_p,
    solarize_p,
    mean,
    std,
    crop_scale,
    input_is_pil,
    rotation_degrees=15,
    use_right_angle_rotation=True,
    grayscale_p=0.1,
    channel_drop_p=0.1,
):
    stages = [
        transforms.RandomResizedCrop(img_size, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  # overhead imagery has no canonical "up"
    ]

    if use_right_angle_rotation:
        # 0/90/180/270 rotation: exact invariance overhead imagery actually has,
        # no interpolation artifacts at the border like continuous rotation would add
        stages.append(transforms.RandomChoice([
            transforms.RandomRotation((angle, angle))
            for angle in (0, 90, 180, 270)
        ]))
    elif rotation_degrees > 0:
        stages.append(transforms.RandomRotation(rotation_degrees))

    if in_chans <= 3:
        # PIL-space colour ops (fast, and what the reference recipes use)
        stages += [
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=grayscale_p),  # lowered: spectral/colour cues carry real signal here
        ]
        if input_is_pil:
            stages.append(transforms.ToTensor())
    else:
        if input_is_pil:
            stages.append(transforms.ToTensor())
        # tensor-space equivalents for >3 bands
        stages += [
            ChannelJitter(0.4, 0.4, p=0.8),
            ChannelDrop(p=channel_drop_p),  # lowered: don't destroy NIR/etc too often
        ]

    # tensor-space from here on
    stages += [
        TensorGaussianBlur(p=blur_p),
        TensorSolarize(threshold=0.5, p=solarize_p),
    ]

    if mean is not None and std is not None:
        stages.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(stages)


class TwoCropsTransform:
    """Returns (view1, view2) -- the positive pair."""

    def __init__(self, aug1, aug2):
        self.aug1 = aug1
        self.aug2 = aug2

    def __call__(self, x):
        return self.aug1(x), self.aug2(x)

QUICKDRAW_MEAN = [0.485, 0.456, 0.406]
QUICKDRAW_STD = [0.229, 0.224, 0.225]


def _build_quickdraw_view(
    img_size,
    in_chans,
    blur_p,
    stroke_p,
    mean,
    std,
    crop_scale,
    input_is_pil,
    ink_is_dark=True,
    min_ink_fraction=0.02,
    affine_p=0.7,
    elastic_p=0.3,
    hflip_p=0.5,
    blur_sigma=(0.1, 0.8),
):
    # Background is whatever ink isn't; affine/elastic must fill with it or the
    # border becomes fake ink.
    fill = 255 if ink_is_dark else 0

    stages = [
        # Plain RandomResizedCrop on a sparse line drawing regularly returns a
        # blank tile. With only two views and no local crops there is no cheap
        # crop to absorb that, so a blank view is a broken positive pair.
        ContentAwareRandomResizedCrop(
            img_size,
            scale=crop_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
            ink_is_dark=ink_is_dark,
            min_ink_fraction=min_ink_fraction,
        ),
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1),
                    shear=8,
                    fill=fill,
                )
            ],
            p=affine_p,
        ),
        transforms.RandomApply(
            [transforms.ElasticTransform(alpha=20.0, sigma=4.0, fill=fill)],
            p=elastic_p,
        ),
        # Replaces ColorJitter entirely: the only photometric variation a doodle
        # actually has is how thick the pen was.
        StrokeWidthJitter(p=stroke_p, ink_is_dark=ink_is_dark),
    ]

    if in_chans == 3:
        stages.append(transforms.Grayscale(num_output_channels=3))

    if input_is_pil:
        stages.append(transforms.ToTensor())

    # Strokes are one or two pixels wide; sigma is capped at 0.8 rather than the
    # 2.0 default, otherwise a blurred view is just a grey smear.
    stages.append(TensorGaussianBlur(p=blur_p, sigma=blur_sigma))

    # No TensorSolarize: on near-binary ink it swaps foreground and background
    # wholesale, which is a distribution shift, not an augmentation.

    if mean is not None and std is not None:
        stages.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(stages)


def get_moco_quickdraw_transforms(
    img_size=224,
    in_chans=3,
    mean=QUICKDRAW_MEAN,
    std=QUICKDRAW_STD,
    crop_scale=(0.5, 1.0),
    input_is_pil=True,
    ink_is_dark=True,
    min_ink_fraction=0.02,
    hflip_p=0.5,
):
    """MoCo v3 two-view transform for Quick, Draw!.

    crop_scale defaults to the DINOv2 *global* range (0.5, 1.0), not the local
    one: MoCo has no local crops, so both views must still contain the object.

    View asymmetry comes from stroke width and blur instead of blur/solarize,
    since solarize is unusable here.
    """
    common = dict(
        img_size=img_size,
        in_chans=in_chans,
        mean=mean,
        std=std,
        crop_scale=crop_scale,
        input_is_pil=input_is_pil,
        ink_is_dark=ink_is_dark,
        min_ink_fraction=min_ink_fraction,
        hflip_p=hflip_p,
    )
    aug1 = _build_quickdraw_view(blur_p=0.5, stroke_p=0.8, **common)
    aug2 = _build_quickdraw_view(blur_p=0.1, stroke_p=0.3, **common)
    return TwoCropsTransform(aug1, aug2)

def get_moco_vit_transforms(
    img_size=224,
    in_chans=3,
    mean=None,
    std=None,
    crop_scale=(0.32, 1.0),
    input_is_pil=True,
    dataset="fmow",
):
    """input_is_pil: set False only if your dataset already yields float tensors
    in [0, 1]. FMoWFromHuggingFace yields PIL, so leave it True."""
    if dataset == "chexpert":
        return get_moco_cxr_transforms(
            img_size=img_size,
            in_chans=in_chans,
            mean=mean,
            std=std,
            crop_scale=crop_scale,
            input_is_pil=input_is_pil,
        )
    if dataset == "domain_quickdraw":
        return get_moco_quickdraw_transforms(
            img_size=img_size,
            in_chans=in_chans,
            crop_scale=(0.5, 1.0),
            input_is_pil=input_is_pil,
        )
    else:
        aug1 = _build_view(img_size, in_chans, 1.0, 0.0, mean, std, crop_scale, input_is_pil)
        aug2 = _build_view(img_size, in_chans, 0.1, 0.2, mean, std, crop_scale, input_is_pil)
        return TwoCropsTransform(aug1, aug2)
# ------------------------------------------------------------------ collate

def moco_collate_fn(batch):
    """Dataset items are ``((view1, view2), target)`` -- TwoCropsTransform makes
    the first element a tuple. Bare ``(view1, view2)`` is also accepted."""
    views1, views2, targets = [], [], []
    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], (tuple, list)):
            (v1, v2), target = item
        else:
            v1, v2 = item
            target = -1

        if not torch.is_tensor(v1) or not torch.is_tensor(v2):
            raise TypeError(
                "moco_collate_fn received non-tensor views "
                f"({type(v1).__name__}). The transform pipeline is missing a "
                "ToTensor() step -- pass input_is_pil=True to "
                "get_moco_vit_transforms()."
            )

        views1.append(v1)
        views2.append(v2)
        targets.append(target)

    out = {
        "collated_view1": torch.stack(views1, dim=0),
        "collated_view2": torch.stack(views2, dim=0),
    }
    # if not all(t == -1 for t in targets):
    #     out["targets"] = torch.as_tensor(targets)
    return out