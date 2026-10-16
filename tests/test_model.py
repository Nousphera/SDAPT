import copy

import pytest
import timm
import torch
from timm.models.vision_transformer import VisionTransformer

from sdapt.model import QVLoRALinear, SplitEncoder, build_encoder


def fixture_backbone():
    return VisionTransformer(img_size=28, patch_size=14, embed_dim=24, depth=3,
                             num_heads=3, num_classes=0, init_values=1., dynamic_img_size=True)


def test_split_and_merged_lora_match_unsplit_encoder():
    torch.manual_seed(4)
    original = fixture_backbone().eval()
    split = SplitEncoder(copy.deepcopy(original), method='sdapt', lora_rank=2, lora_alpha=4).eval()
    image = torch.randn(2, 3, 42, 42)
    with torch.no_grad():
        torch.testing.assert_close(split(image), original.forward_features(image)[:, 0])
        for module in split.modules():
            if isinstance(module, QVLoRALinear):
                module.q_b.normal_(); module.v_b.normal_()
        merged = fixture_backbone().eval()
        merged.load_state_dict(split.encoder_state_dict(), strict=True)
        torch.testing.assert_close(split(image), merged.forward_features(image)[:, 0], atol=1e-5, rtol=1e-5)


def test_qv_lora_preserves_keys_and_frozen_client():
    layer = QVLoRALinear(torch.nn.Linear(4, 12), rank=2, alpha=2)
    with torch.no_grad():
        layer.q_b.fill_(1); layer.v_b.fill_(1)
    x = torch.randn(2, 4)
    torch.testing.assert_close(layer(x)[:, 4:8], layer.base(x)[:, 4:8])
    assert not layer.base.weight.requires_grad
    model = SplitEncoder(fixture_backbone(), method='sdapt', lora_rank=2)
    model(torch.randn(2, 3, 28, 28)).square().mean().backward()
    assert all(p.grad is None for p in model.client.parameters())
    assert any(p.grad is not None for p in model.server.blocks[-1].parameters())


def test_server_mask_does_not_modify_smashed_features():
    model = SplitEncoder(fixture_backbone(), method='sdapt', lora_rank=2).eval()
    x = torch.randn(2, 3, 28, 28)
    tokens = model.client(x)
    saved = tokens.clone()
    mask = torch.ones(2, 4, dtype=torch.bool)
    masked = model.server(tokens, mask)
    torch.testing.assert_close(tokens, saved)
    assert not torch.allclose(masked, model.server(tokens))


def test_full_vit_b14_split_equivalence():
    torch.manual_seed(1)
    full = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False,
                             num_classes=0, dynamic_img_size=True).eval()
    images = torch.randn(1, 3, 28, 28)
    with torch.no_grad():
        expected = full.forward_features(images)[:, 0]
        split = SplitEncoder(full, method='sdapt').eval()
        torch.testing.assert_close(split(images), expected)


def test_constructor_never_silently_uses_random_weights(tmp_path):
    with pytest.raises(ValueError):
        build_encoder()
    with pytest.raises(FileNotFoundError):
        build_encoder(pretrained_path=tmp_path / 'missing.safetensors')


@pytest.mark.parametrize('split_depth', [0, 2])
def test_nondefault_split_preserves_full_model_and_state_keys(split_depth):
    model = fixture_backbone().eval()
    image = torch.randn(1, 3, 28, 28)
    expected = model.forward_features(image)[:, 0]
    original_state = set(model.state_dict())
    split = SplitEncoder(model, method='sdapt', split_depth=split_depth, lora_rank=2).eval()
    torch.testing.assert_close(split(image), expected)
    assert set(split.encoder_state_dict()) == original_state


def fixture_eva(qkv_bias=False):
    from timm.models.eva import Eva
    return Eva(img_size=32, patch_size=16, embed_dim=32, depth=3, num_heads=4,
               num_classes=0, init_values=1., num_reg_tokens=4, qkv_bias=qkv_bias,
               use_rot_pos_emb=True, use_abs_pos_emb=False, rope_type='dinov3',
               rope_rotate_half=True, use_fc_norm=False, dynamic_img_size=True)


@pytest.mark.parametrize('qkv_bias', [False, True])
def test_dinov3_rotary_register_split_and_lora_merge(qkv_bias):
    original = fixture_eva(qkv_bias).eval()
    images = torch.randn(2, 3, 48, 32)
    expected = original.forward_features(images)[:, 0]
    split = SplitEncoder(copy.deepcopy(original), method='sdapt', lora_rank=2).eval()
    torch.testing.assert_close(split(images), expected)
    for layer in split.modules():
        if isinstance(layer, QVLoRALinear):
            with torch.no_grad():
                layer.q_b.normal_(); layer.v_b.normal_()
    merged = fixture_eva(qkv_bias).eval()
    merged.load_state_dict(split.encoder_state_dict(), strict=True)
    torch.testing.assert_close(split(images), merged.forward_features(images)[:, 0], atol=1e-5, rtol=1e-5)
    smashed = split.client(images)
    # Six patches, with CLS plus all four registers excluded from the mask.
    result = split.server(smashed, torch.ones(2, 6, dtype=torch.bool))
    assert result.shape == (2, 11, 32)


def test_dinov2_register_tokens_are_not_counted_as_ibot_patches():
    original = VisionTransformer(img_size=28, patch_size=14, embed_dim=24, depth=3,
                                 num_heads=3, num_classes=0, reg_tokens=4,
                                 dynamic_img_size=True, no_embed_class=True)
    split = SplitEncoder(original, method='sdapt', lora_rank=2)
    tokens = split.client(torch.randn(2, 3, 28, 28))
    assert tokens.shape == (2, 9, 24)
    assert split.server(tokens, torch.ones(2, 4, dtype=torch.bool)).shape == tokens.shape


def test_gradient_checkpointing_preserves_forward_and_backward():
    plain = SplitEncoder(fixture_eva(), method='sdapt', lora_rank=2)
    checkpointed = copy.deepcopy(plain)
    checkpointed.server.grad_checkpointing = True
    images = torch.randn(2, 3, 32, 32)
    one, two = plain(images), checkpointed(images)
    torch.testing.assert_close(one, two)
    one[:, 0].sum().backward(); two[:, 0].sum().backward()
    for left, right in zip(plain.parameters(), checkpointed.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad)


def test_sdapt_trainable_norms_stay_on_server():
    split = SplitEncoder(fixture_backbone(), method='sdapt', lora_rank=2)
    assert all(not p.requires_grad for p in split.client.parameters())
    for module in split.server.modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert all(p.requires_grad for p in module.parameters())


def test_bfloat16_frozen_encoder_accepts_normal_float_images():
    from sdapt.model import DINOEncoder
    encoder = DINOEncoder(fixture_eva().to(torch.bfloat16))
    output = encoder(torch.randn(2, 3, 32, 32))
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()


def test_strict_meta_loader_materializes_nonpersistent_rotary_buffers(monkeypatch):
    import sdapt.model as implementation
    source = fixture_eva(qkv_bias=True)
    state = {key: value.clone() for key, value in source.state_dict().items()}
    monkeypatch.setattr(implementation.timm, 'create_model', lambda *args, **kwargs: fixture_eva(qkv_bias=True).to('meta'))
    encoder = implementation.build_encoder(state_dict=state, model='dinov3-vits16', parameter_dtype='bfloat16')
    assert all(not item.is_meta for item in encoder.buffers())
    assert encoder.backbone.rope.periods.dtype == torch.float32
    assert torch.isfinite(encoder(torch.randn(1, 3, 32, 32))).all()


@pytest.mark.parametrize('name', ['dino-vits16', 'dinov2-vits14-reg4', 'dinov3-vits16'])
def test_real_family_architecture_strict_meta_load_and_forward(name):
    from sdapt.registry import get_model_spec
    source = timm.create_model(get_model_spec(name).timm_name, pretrained=False,
                               num_classes=0, dynamic_img_size=True).eval()
    size = get_model_spec(name).patch_size * 2
    images = torch.randn(1, 3, size, size)
    with torch.no_grad():
        expected = source.forward_features(images)[:, 0]
        encoder = build_encoder(state_dict=source.state_dict(), model=name)
        torch.testing.assert_close(encoder(images), expected)


@pytest.mark.parametrize('name', __import__('sdapt.registry', fromlist=['MODEL_SPECS']).MODEL_SPECS)
def test_every_registered_architecture_constructs_sdapt_split_on_meta(name):
    from sdapt.registry import get_model_spec
    spec = get_model_spec(name)
    with torch.device('meta'):
        backbone = timm.create_model(spec.timm_name, pretrained=False, num_classes=0, dynamic_img_size=True)
        split = SplitEncoder(backbone, method='sdapt', split_depth=1, lora_rank=64,
                             lora_alpha=32, gradient_checkpointing=True)
    assert len(split.server.blocks) == spec.depth - 1
    assert split.num_features == spec.embed_dim
    assert split.num_prefix_tokens == spec.num_prefix_tokens
    assert all(not parameter.requires_grad for parameter in split.client.parameters())
    assert split.server.grad_checkpointing


def test_frozen_storage_cast_preserves_original_trainable_weights(tmp_path, monkeypatch):
    import sdapt.model as implementation
    from safetensors.torch import save_file
    source = fixture_backbone()
    with torch.no_grad():
        source.blocks[-1].norm1.weight.fill_(1.003)
    path = tmp_path / 'fixture.safetensors'
    save_file(source.state_dict(), str(path))
    monkeypatch.setattr(implementation.timm, 'create_model', lambda *args, **kwargs: fixture_backbone())
    split = implementation.build_split_encoder(path, parameter_dtype='bfloat16', lora_rank=2)
    assert split.client.patch_embed.proj.weight.dtype == torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in split.parameters() if parameter.requires_grad)
    torch.testing.assert_close(split.server.blocks[-1].norm1.weight, torch.full((24,), 1.003), atol=0, rtol=0)
    assert torch.isfinite(split(torch.randn(2, 3, 28, 28))).all()


def test_mixed_precision_lora_merge_preserves_small_updates():
    base = torch.nn.Linear(4, 12, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        base.weight.fill_(1.)
    layer = QVLoRALinear(base, rank=1, alpha=1)
    for parameter in (layer.q_a, layer.q_b, layer.v_a, layer.v_b):
        parameter.data = parameter.data.float()
    with torch.no_grad():
        layer.q_a.fill_(1.); layer.q_b.fill_(.001)
        layer.v_a.fill_(1.); layer.v_b.fill_(.001)
    merged = layer.merged_weight()
    assert merged.dtype == torch.float32
    assert merged[0, 0].item() == pytest.approx(1.001)
    assert torch.isfinite(layer(torch.randn(2, 4).bfloat16())).all()


def test_mixed_storage_keeps_small_online_adam_updates(tmp_path, monkeypatch):
    import sdapt.model as implementation
    from safetensors.torch import save_file
    source = fixture_backbone()
    path = tmp_path / 'fixture.safetensors'
    save_file(source.state_dict(), str(path))
    monkeypatch.setattr(implementation.timm, 'create_model', lambda *args, **kwargs: fixture_backbone())
    split = implementation.build_split_encoder(path, parameter_dtype='bfloat16', lora_rank=2)
    parameter = split.server.blocks[0].norm1.weight
    optimizer = torch.optim.AdamW([parameter], lr=.001, weight_decay=0)
    for _ in range(1000):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    assert parameter.abs().max().item() < .001
