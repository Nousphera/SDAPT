"""Exercise the real four-method workflow using an injected test-only small ViT.

No production tiny model or synthetic-data mode is exposed. Dataset manifests,
augmentations, training objectives, checkpoint IO and evaluation remain real.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from timm.models.vision_transformer import VisionTransformer

from sdapt import experiment, model, training


def _backbone():
    return VisionTransformer(img_size=28, patch_size=14, embed_dim=24, depth=4,
                             num_heads=3, num_classes=0, init_values=1., dynamic_img_size=True)


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    torch.manual_seed(911)
    initial = _backbone()
    pretrained = tmp_path / 'fixture.safetensors'
    save_file(initial.state_dict(), str(pretrained))
    folder = tmp_path / 'data' / 'fmow'
    folder.mkdir(parents=True)
    (folder / 'metadata.json').write_text(json.dumps({'classes': ['vertical', 'horizontal'], 'multilabel': False}))
    for split, count in [('train', 24), ('test', 4)]:
        rows = []
        for index in range(count):
            label = index % 2
            pixels = np.full((32, 32, 3), 240, np.uint8)
            if label:
                pixels[8 + index % 3:20 + index % 3, :, :] = 15
            else:
                pixels[:, 8 + index % 3:20 + index % 3, :] = 15
            path = f'{split}-{index}.png'
            Image.fromarray(pixels).save(folder / path)
            rows.append(json.dumps({'path': path, 'target': label}))
        (folder / f'{split}.jsonl').write_text('\n'.join(rows) + '\n')
    built = []

    def split_factory(pretrained_path, **kwargs):
        kwargs.pop('model', None)
        parameter_dtype = kwargs.pop('parameter_dtype', torch.float32)
        vit = _backbone()
        vit.load_state_dict(load_file(str(pretrained_path)), strict=True)
        split = model.SplitEncoder(vit, **kwargs)
        for parameter in split.parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.to(parameter_dtype)
        built.append((split, {key: value.detach().clone() for key, value in split.client.state_dict().items()}))
        return split

    def encoder_factory(pretrained_path=None, state_dict=None, **kwargs):
        vit = _backbone()
        vit.load_state_dict(load_file(str(pretrained_path)) if state_dict is None else state_dict, strict=True)
        return model.DINOEncoder(vit)

    monkeypatch.setattr(model, 'build_split_encoder', split_factory)
    monkeypatch.setattr(model, 'build_encoder', encoder_factory)
    for name in ('SDAPTTrainer', 'DINOLocalTrainer'):
        constructor = getattr(training, name)
        def factory(student, _constructor=constructor, **kwargs):
            return _constructor(student, head_dim=32, hidden_dim=32, bottleneck_dim=8, **kwargs)
        monkeypatch.setattr(training, name, factory)
    moco_constructor = training.MoCoSFLTrainer
    monkeypatch.setattr(training, 'MoCoSFLTrainer', lambda student, **kwargs: moco_constructor(
        student, project_dim=8, hidden_dim=32, **kwargs))
    common = ['--dataset', 'fmow', '--data_root', str(tmp_path / 'data'),
              '--pretrained', str(pretrained), '--output', str(tmp_path / 'results'),
              '--device', 'cpu', '--precision', 'float32', '--nr_of_clients', '2',
              '--batch_size', '2', '--nr_of_epochs', '1', '--steps_per_epoch', '2',
              '--local_steps', '1', '--accumulation_steps', '7', '--learning_rate', '.002',
              '--lora_rank', '2', '--lora_alpha', '4', '--split_layer', '2',
              '--shots', '1', '0', '--head_epochs', '2', '--head_batch_size', '4',
              '--freeze_head_epochs', '0', '--random_seed', '24']
    return tmp_path, common, built, initial


def test_all_four_methods_train_reload_and_use_identical_fewshot_samples(workflow):
    root, common, built, initial = workflow
    samples, partitions = [], []
    for method in ('no-dapt', 'sdapt', 'fedavg', 'mocosfl'):
        args = experiment.parse_args([*common, '--method', method, '--run_name', method])
        result = experiment.run(args)
        directory = Path(result['directory'])
        assert set(result['evaluation']) == {'1-shot', 'linear-probe'}
        one_shot = json.loads((directory / '1-shot.json').read_text())
        samples.append(one_shot['sampled_indices'])
        partitions.append(json.loads((directory / 'partitions.json').read_text()))
        assert one_shot['config']['epochs'] == 2
        assert all(client['training_sample_count'] == 2 for client in one_shot['metrics']['clients'])
        head_checkpoint = torch.load(directory / 'heads-1-shot.pt', weights_only=True)
        assert len(head_checkpoint['heads']) == 2
        if method == 'no-dapt':
            assert result['adaptation'] is None
            assert not (directory / 'teacher.pt').exists()
            continue
        checkpoint = torch.load(directory / 'teacher.pt', weights_only=True)
        assert checkpoint['protocol']['method'] == method
        assert checkpoint['adaptation']['optimizer_steps'] == (4 if method == 'fedavg' else 1)
        assert checkpoint['adaptation']['uplink_bytes'] > 0
        restored = _backbone()
        restored.load_state_dict(checkpoint['encoder'], strict=True)
        assert any(not torch.equal(initial.state_dict()[key], value) for key, value in checkpoint['encoder'].items())
        split, before = built[-1]
        assert split.split_depth == 2
        if method == 'sdapt':
            for key, value in split.client.state_dict().items():
                torch.testing.assert_close(value, before[key])
            adapters = [module for module in split.modules() if isinstance(module, model.QVLoRALinear)]
            assert adapters and all(module.rank == 2 and module.alpha == 4 for module in adapters)
            assert checkpoint['adaptation']['downlink_bytes'] == 0
        elif method == 'mocosfl':
            assert len(checkpoint['client_states']) == 2
            assert checkpoint['adaptation']['uplink_bytes'] == 2 * checkpoint['adaptation']['paper_uplink_bytes']
        evaluation_args = experiment.parse_args([*common, '--method', method, '--stage', 'evaluate',
            '--checkpoint', str(directory / 'teacher.pt'), '--run_name', method + '-reload'])
        reloaded = experiment.run(evaluation_args)
        assert reloaded['evaluation'] == result['evaluation']
    assert samples[0] == samples[1] == samples[2] == samples[3]
    assert partitions[0] == partitions[1] == partitions[2] == partitions[3]


def test_checkpoint_protocol_rejects_changed_seed_before_output(workflow):
    root, common, _, _ = workflow
    adapted = experiment.run(experiment.parse_args([*common, '--method', 'sdapt', '--stage', 'adapt', '--run_name', 'adapt']))
    checkpoint = Path(adapted['directory']) / 'teacher.pt'
    args = experiment.parse_args([*common, '--method', 'sdapt', '--stage', 'evaluate', '--checkpoint', str(checkpoint),
                                  '--random_seed', '25', '--run_name', 'bad-seed'])
    with pytest.raises(ValueError, match='protocol mismatch for random_seed'):
        experiment.run(args)
    assert not (root / 'results' / 'bad-seed').exists()
