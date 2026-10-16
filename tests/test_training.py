import copy

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from sdapt.model import SplitEncoder
from sdapt.training import (DINOLocalTrainer, MoCoSFLTrainer, SDAPTTrainer,
                            StateAccumulator, ema_update, weighted_average_states)


def backbone(method):
    vit = VisionTransformer(img_size=28, patch_size=14, embed_dim=24, depth=3,
                            num_heads=3, num_classes=0, init_values=1., dynamic_img_size=True)
    return SplitEncoder(vit, method=method, lora_rank=2)


def crops(batch=2, locals=2):
    return [torch.randn(batch, 3, 28, 28) for _ in range(2)] + [torch.randn(batch, 3, 14, 14) for _ in range(locals)]


def small_trainer(method='sdapt', accumulation=1):
    cls = SDAPTTrainer if method == 'sdapt' else DINOLocalTrainer
    return cls(backbone(method), head_dim=8, hidden_dim=32, bottleneck_dim=8,
               accumulation_steps=accumulation, transport_dtype=torch.float32)


def test_sdapt_step_freezes_client_updates_server_and_ema_only_at_boundary():
    torch.manual_seed(4)
    trainer = small_trainer(accumulation=2)
    optimizer = torch.optim.SGD(trainer.trainable_parameters(), lr=.03)
    client_before = copy.deepcopy(trainer.student.client.state_dict())
    server_before = trainer.student.server.blocks[-1].mlp.fc1.weight.detach().clone()
    teacher_before = trainer.teacher_server.blocks[-1].mlp.fc1.weight.detach().clone()
    metrics = trainer.train_step([crops(), crops()], optimizer=optimizer, ema_momentum=.5)
    assert metrics['optimizer_step'] is False
    torch.testing.assert_close(trainer.teacher_server.blocks[-1].mlp.fc1.weight, teacher_before)
    metrics = trainer.train_step([crops(), crops()], optimizer=optimizer, ema_momentum=.5)
    assert metrics['optimizer_step'] is True
    assert not torch.equal(trainer.student.server.blocks[-1].mlp.fc1.weight, server_before)
    torch.testing.assert_close(trainer.teacher_server.blocks[-1].mlp.fc1.weight,
                               (server_before + trainer.student.server.blocks[-1].mlp.fc1.weight) / 2)
    for key, value in trainer.student.client.state_dict().items():
        torch.testing.assert_close(value, client_before[key])
    assert all(p.grad is None for p in trainer.student.client.parameters())
    assert metrics['uplink_bytes'] == 4 * (2 * 5 + 2 * 2) * 24 * 4
    assert metrics['downlink_bytes'] == 0


def test_partial_accumulation_flush_is_not_underweighted():
    torch.manual_seed(3)
    first = small_trainer(accumulation=1)
    second = copy.deepcopy(first)
    second.accumulation_steps = 7
    opt1 = torch.optim.SGD(first.trainable_parameters(), lr=.01)
    opt2 = torch.optim.SGD(second.trainable_parameters(), lr=.01)
    batch = [crops()]
    mask = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.bool)
    first.train_step(batch, optimizer=opt1, masks=mask, ema_momentum=.5)
    second.train_step(batch, optimizer=opt2, masks=mask, ema_momentum=.5, flush=True)
    for x, y in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(x, y)


def test_fedavg_uses_weighted_parameters_not_optimizer_updates():
    first = {'weight': torch.tensor([1., 3.]), 'counter': torch.tensor(1)}
    second = {'weight': torch.tensor([5., 7.]), 'counter': torch.tensor(1)}
    result = weighted_average_states([first, second], [1, 3])
    torch.testing.assert_close(result['weight'], torch.tensor([4., 6.]))
    assert result['counter'] == 1
    accumulator = StateAccumulator()
    accumulator.add(first, 1); accumulator.add(second, 3)
    torch.testing.assert_close(accumulator.average()['weight'], result['weight'])
    with pytest.raises(ValueError):
        weighted_average_states([first, second], [0, 0])


def test_fedavg_local_step_updates_complete_client_and_teacher():
    trainer = small_trainer('fedavg')
    optimizer = torch.optim.SGD(trainer.trainable_parameters(), lr=.03)
    before = trainer.student.client.patch_embed.proj.weight.detach().clone()
    metrics = trainer.train_step([crops()], optimizer=optimizer, ema_momentum=.5)
    assert metrics['optimizer_step']
    assert metrics['uplink_bytes'] == 0
    assert not torch.equal(trainer.student.client.patch_embed.proj.weight, before)
    assert all(p.grad is None for p in trainer.teacher.parameters())


def test_moco_uses_independent_trainable_clients_then_weighted_sync():
    trainer = MoCoSFLTrainer(backbone('mocosfl'), num_clients=2, project_dim=8,
                             hidden_dim=32, transport_dtype=torch.float32, accumulation_steps=1)
    optimizer = torch.optim.SGD(trainer.trainable_parameters(), lr=.01)
    before = [client.patch_embed.proj.weight.detach().clone() for client in trainer.client_models]
    metrics = trainer.train_step([crops(locals=0), crops(locals=0)], optimizer=optimizer, ema_momentum=.5)
    assert metrics['loss'] > 0 and metrics['downlink_bytes'] > 0
    for client, initial in zip(trainer.client_models, before):
        assert not torch.equal(client.patch_embed.proj.weight, initial)
    one = trainer.client_models[0].patch_embed.proj.weight.detach().clone()
    two = trainer.client_models[1].patch_embed.proj.weight.detach().clone()
    trainer.synchronize_clients([1, 3])
    for client in trainer.client_models:
        torch.testing.assert_close(client.patch_embed.proj.weight, (one + 3 * two) / 4)


def test_teacher_snapshot_is_plain_backbone_and_matches_teacher_output():
    trainer = small_trainer()
    optimizer = torch.optim.SGD(trainer.trainable_parameters(), lr=.01)
    trainer.train_step([crops()], optimizer=optimizer, ema_momentum=.5)
    vit = VisionTransformer(img_size=28, patch_size=14, embed_dim=24, depth=3,
                            num_heads=3, num_classes=0, init_values=1., dynamic_img_size=True)
    vit.load_state_dict(trainer.teacher_state_dict(), strict=True)
    images = torch.randn(2, 3, 28, 28)
    with torch.no_grad():
        expected = trainer.teacher_server(trainer.student.client(images))[:, 0]
        torch.testing.assert_close(vit.forward_features(images)[:, 0], expected)


def test_ema_rejects_wrong_structure_and_copies_integer_buffers():
    teacher = torch.nn.BatchNorm1d(3)
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        student.weight.fill_(3)
        student.num_batches_tracked.fill_(5)
    ema_update(teacher, student, .5)
    torch.testing.assert_close(teacher.weight, torch.full((3,), 2.))
    assert teacher.num_batches_tracked == 5
    with pytest.raises(ValueError):
        ema_update(teacher, torch.nn.Linear(3, 3), .5)


def test_freeze_last_head_does_not_apply_weight_decay():
    trainer = small_trainer()
    optimizer = torch.optim.AdamW(trainer.trainable_parameters(), lr=.01, weight_decay=.1)
    before = trainer.head.prototypes.detach().clone()
    trainer.train_step([crops()], optimizer=optimizer, freeze_head=True)
    torch.testing.assert_close(trainer.head.prototypes, before)
    trainer.train_step([crops()], optimizer=optimizer, freeze_head=False)
    assert not torch.equal(trainer.head.prototypes, before)


def test_sdapt_global_centering_is_invariant_to_client_batch_regrouping():
    torch.manual_seed(33)
    separated = small_trainer()
    pooled = copy.deepcopy(separated)
    opt1 = torch.optim.SGD(separated.trainable_parameters(), lr=.01)
    opt2 = torch.optim.SGD(pooled.trainable_parameters(), lr=.01)
    first, second = crops(batch=1), crops(batch=3)
    together = [torch.cat((x, y)) for x, y in zip(first, second)]
    mask = torch.zeros(8, 4, dtype=torch.bool)
    mask[:, 0] = True
    left = separated.train_step([first, second], optimizer=opt1, masks=mask)
    right = pooled.train_step([together], optimizer=opt2, masks=mask)
    assert left['loss'] == pytest.approx(right['loss'], rel=1e-5)
    for x, y in zip(separated.parameters(), pooled.parameters()):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)


def test_nonfinite_gradient_aborts_optimizer_and_teacher_update():
    trainer = small_trainer()
    optimizer = torch.optim.SGD(trainer.trainable_parameters(), lr=.01)
    before = trainer.student.server.blocks[-1].mlp.fc1.weight.detach().clone()
    handle = trainer.head.prototypes.register_hook(lambda grad: grad * float('nan'))
    with pytest.raises(FloatingPointError):
        trainer.train_step([crops()], optimizer=optimizer)
    handle.remove()
    torch.testing.assert_close(trainer.student.server.blocks[-1].mlp.fc1.weight, before)
    assert trainer.pending_steps == 0
    assert all(p.grad is None for p in trainer.parameters())


@pytest.mark.parametrize('method', ['sdapt', 'fedavg', 'mocosfl'])
def test_rotary_register_models_train_and_export_each_baseline(method):
    from timm.models.eva import Eva
    def eva():
        return Eva(img_size=32, patch_size=16, embed_dim=32, depth=3, num_heads=4,
                   num_classes=0, init_values=1., num_reg_tokens=4, qkv_bias=False,
                   use_rot_pos_emb=True, use_abs_pos_emb=False, rope_type='dinov3',
                   rope_rotate_half=True, use_fc_norm=False, dynamic_img_size=True)
    split = SplitEncoder(eva(), method=method, lora_rank=2, gradient_checkpointing=True)
    if method == 'mocosfl':
        trainer = MoCoSFLTrainer(split, num_clients=2, project_dim=8, hidden_dim=32, accumulation_steps=1)
        batch = [[torch.randn(2, 3, 32, 32) for _ in range(2)] for _ in range(2)]
    else:
        constructor = SDAPTTrainer if method == 'sdapt' else DINOLocalTrainer
        trainer = constructor(split, head_dim=8, hidden_dim=32, bottleneck_dim=8, accumulation_steps=1)
        batch = [[torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.randn(2, 3, 16, 16)]]
    optimizer = torch.optim.SGD(trainer.trainable_parameters(), lr=.01)
    result = trainer.train_step(batch, optimizer=optimizer, ema_momentum=.5)
    assert result['optimizer_step'] and np_isfinite(result['loss'])
    restored = eva()
    restored.load_state_dict(trainer.teacher_state_dict(), strict=True)
    assert torch.isfinite(restored.forward_features(torch.randn(2, 3, 32, 32))).all()


def np_isfinite(value):
    import math
    return math.isfinite(value)


def test_bfloat16_ema_does_not_drift_when_student_equals_teacher():
    student = torch.nn.Linear(16, 16).to(torch.bfloat16)
    with torch.no_grad():
        student.weight.copy_(torch.arange(256).reshape(16, 16) / 511)
    teacher = copy.deepcopy(student)
    original = teacher.weight.detach().clone()
    ema_update(teacher, student, .994)
    torch.testing.assert_close(teacher.weight, original.float(), atol=0, rtol=0)


def test_repeated_sub_bfloat16_ema_updates_accumulate_in_float32():
    student = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        teacher.weight.fill_(1.)
        student.weight.fill_(1.0078125)
    for _ in range(1000):
        ema_update(teacher, student, .994)
    assert teacher.weight.dtype == torch.float32
    expected = 1.0078125 - .0078125 * (.994 ** 1000)
    assert teacher.weight.item() == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize('method', ['sdapt', 'fedavg', 'mocosfl'])
def test_bfloat16_students_use_float32_ema_and_train_without_autocast(method):
    student = backbone(method)
    for parameter in student.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    if method == 'mocosfl':
        trainer = MoCoSFLTrainer(student, num_clients=2, project_dim=8, hidden_dim=32, accumulation_steps=1)
        batch = [crops(locals=0), crops(locals=0)]
    else:
        constructor = SDAPTTrainer if method == 'sdapt' else DINOLocalTrainer
        trainer = constructor(student, head_dim=8, hidden_dim=32, bottleneck_dim=8, accumulation_steps=1)
        batch = [crops()]
    teacher = trainer.teacher_server if method == 'sdapt' else trainer.teacher
    assert all(parameter.dtype == torch.float32 for parameter in teacher.parameters())
    optimizer = torch.optim.AdamW(trainer.trainable_parameters(), lr=.001)
    result = trainer.train_step(batch, optimizer=optimizer, ema_momentum=.994)
    assert result['optimizer_step'] and np_isfinite(result['loss'])
    assert all(parameter.dtype == torch.float32 for parameter in trainer.trainable_parameters())
    if method == 'fedavg':
        accumulator = StateAccumulator()
        accumulator.add(trainer.state_dict(), 1)
        trainer.load_state_dict(accumulator.average(), strict=True)
        assert all(parameter.dtype == torch.float32 for parameter in trainer.teacher.parameters())


@pytest.mark.parametrize('autocast', [False, True])
@pytest.mark.parametrize('qkv_bias', [False, True])
def test_rotary_mixed_storage_lora_trains_and_exports(tmp_path, monkeypatch, autocast, qkv_bias):
    import contextlib
    import sdapt.model as implementation
    from safetensors.torch import save_file
    from timm.models.eva import Eva
    def eva():
        return Eva(img_size=32, patch_size=16, embed_dim=32, depth=3, num_heads=4,
                   num_classes=0, init_values=1., num_reg_tokens=4, qkv_bias=qkv_bias,
                   use_rot_pos_emb=True, use_abs_pos_emb=False, rope_type='dinov3',
                   rope_rotate_half=True, use_fc_norm=False, dynamic_img_size=True)
    source = eva()
    path = tmp_path / 'eva.safetensors'
    save_file(source.state_dict(), str(path))
    monkeypatch.setattr(implementation.timm, 'create_model', lambda *args, **kwargs: eva())
    student = implementation.build_split_encoder(path, parameter_dtype='bfloat16', model='dinov3-vits16',
                                                  lora_rank=2, gradient_checkpointing=True)
    trainer = SDAPTTrainer(student, head_dim=8, hidden_dim=32, bottleneck_dim=8, accumulation_steps=1)
    optimizer = torch.optim.AdamW(trainer.trainable_parameters(), lr=.01)
    batch = [[torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.randn(2, 3, 16, 16)]]
    context = torch.autocast('cpu', dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    with context:
        trainer.train_step(batch, optimizer=optimizer, ema_momentum=.5)
    assert any(module.q_b.count_nonzero() for module in trainer.teacher_server.modules()
               if isinstance(module, implementation.QVLoRALinear))
    exported = trainer.teacher_state_dict()
    merged = eva()
    merged.load_state_dict(exported, strict=True)
    merged_split = SplitEncoder(merged, method='mocosfl')
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        smashed = trainer.student.client(images)
        torch.testing.assert_close(trainer.teacher_server(smashed), merged_split.server(smashed), atol=1e-5, rtol=1e-5)
        restored = implementation.build_encoder(state_dict=exported, model='dinov3-vits16', parameter_dtype='bfloat16')
        expected = eva().to(torch.bfloat16)
        expected.load_state_dict(exported, strict=True)
        expected.rope.periods = expected.rope._compute_periods(device='cpu', dtype=torch.float32)
        torch.testing.assert_close(restored(images), expected.forward_features(images.bfloat16())[:, 0])
