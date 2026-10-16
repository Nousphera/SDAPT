from pathlib import Path
import pytest
import torch
from sdapt.experiment import parse_args, schedule, resolve_device, atomic_json


def test_paper_defaults_and_explicit_overrides():
    args = parse_args(['--dataset', 'quickdraw'])
    assert args.method == 'sdapt'
    assert args.nr_of_clients == 10
    assert args.split_layer == 1
    assert args.shots == [4, 8, 16, 0]
    args = parse_args(['--dataset', 'chexpert', '--nr_of_clients', '3', '--batch_size', '2', '--learning_rate', '.002'])
    assert (args.nr_of_clients, args.batch_size, args.learning_rate) == (3, 2, .002)
    assert args.unfreeze_last == 2


@pytest.mark.parametrize('flags', [
    ['--nr_of_clients', '0'], ['--batch_size', '0'], ['--nr_of_epochs', '0'],
    ['--shots', '-1'], ['--split_layer', '12'], ['--learning_rate', '-1'],
    ['--stage', 'evaluate'], ['--method', 'no-dapt', '--stage', 'adapt'],
    ['--accumulation_steps', '0'], ['--local_steps', '0'],
])
def test_invalid_experiments_fail_before_loading_data(flags):
    with pytest.raises(SystemExit):
        parse_args(['--dataset', 'fmow', *flags])


def test_short_schedule_is_finite_and_preserves_start_end():
    assert schedule(0, 1, 1e-3, 1e-6, warmup=10) == pytest.approx(1e-3)
    assert schedule(0, 10, 1e-3, 1e-6, warmup=0) == pytest.approx(1e-3)
    assert schedule(9, 10, 1e-3, 1e-6, warmup=0) == pytest.approx(1e-6)
    assert 0 < schedule(0, 100, 1e-3, 1e-6, warmup=10) < 1e-3


def test_unavailable_gpu_is_an_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='CUDA'):
        resolve_device('cuda')
    assert resolve_device('cpu') == torch.device('cpu')


def test_json_rejects_nonfinite_values_and_keeps_previous_output(tmp_path):
    target = tmp_path / 'metrics.json'
    atomic_json(target, {'accuracy': .5})
    with pytest.raises(ValueError):
        atomic_json(target, {'accuracy': float('nan')})
    assert '.5' in target.read_text()
    assert not list(tmp_path.glob('*.tmp'))


def test_dino_v3_cli_uses_matching_weights_depth_and_dtype():
    args = parse_args(['--dataset', 'fmow', '--model', 'dinov3-vit7b16', '--split_layer', '20', '--parameter_dtype', 'bfloat16', '--gradient_checkpointing'])
    assert args.pretrained == Path('checkpoints/dinov3-vit7b16.safetensors')
    assert args.split_layer == 20 and args.parameter_dtype == 'bfloat16'
    assert args.gradient_checkpointing
    with pytest.raises(SystemExit):
        parse_args(['--dataset', 'fmow', '--model', 'dinov3-vitl16', '--split_layer', '24'])


def test_reduced_weight_storage_rejects_unsupported_device(monkeypatch):
    from sdapt.experiment import _precision
    args = parse_args(['--dataset', 'quickdraw', '--parameter_dtype', 'bfloat16', '--precision', 'float32'])
    with pytest.raises(ValueError, match='MPS'):
        _precision(args, torch.device('mps'))
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: False)
    with pytest.raises(ValueError, match='bfloat16'):
        _precision(args, torch.device('cuda'))
    assert _precision(args, torch.device('cpu')) == torch.float32
