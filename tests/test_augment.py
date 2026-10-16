import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from sdapt.augment import MultiCrop, eval_transform


@pytest.mark.parametrize("domain,global_size,local_size", [("fmow", 224, 98), ("quickdraw", 224, 98), ("chexpert", 518, 196)])
def test_domain_crop_sizes_and_finite_normalized_outputs(domain, global_size, local_size):
    image = Image.new("RGB", (540, 540), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 500, 500), fill="black")
    crops = MultiCrop(domain, local_crops=2)(image)
    assert [tuple(crop.shape) for crop in crops] == [(3, global_size, global_size)] * 2 + [(3, local_size, local_size)] * 2
    assert all(torch.isfinite(crop).all() for crop in crops)


def test_evaluation_transform_is_deterministic_and_preserves_whole_image():
    image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
    transform = eval_transform("fmow")
    assert torch.equal(transform(image), transform(image))
    assert transform(image).shape == (3, 224, 224)


def test_unknown_domain_does_not_fall_back_silently():
    with pytest.raises(ValueError):
        MultiCrop("other")


@pytest.mark.parametrize("patch,domain,global_size,local_size", [(8, "fmow", 224, 96), (16, "fmow", 224, 96), (8, "chexpert", 512, 192), (16, "chexpert", 512, 192)])
def test_crop_geometry_adapts_to_encoder_patch_grid(patch, domain, global_size, local_size):
    image = Image.new("RGB", (540, 540), "gray")
    crops = MultiCrop(domain, local_crops=1, patch_size=patch)(image)
    assert [tuple(crop.shape[-2:]) for crop in crops] == [(global_size, global_size)] * 2 + [(local_size, local_size)]
    assert eval_transform(domain, patch_size=patch)(image).shape == (3, 224, 224)


def test_invalid_patch_size_is_rejected_before_generating_views():
    with pytest.raises(ValueError, match="patch_size"):
        MultiCrop("fmow", patch_size=0)
