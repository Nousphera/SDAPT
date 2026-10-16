"""Run domain adaptation and private-head evaluation from one source checkout."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import time
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from .registry import DEFAULT_MODEL, MODEL_SPECS, get_model_spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=('fmow', 'quickdraw', 'chexpert'), required=True)
    parser.add_argument('--model', choices=tuple(MODEL_SPECS), default=DEFAULT_MODEL)
    parser.add_argument('--parameter_dtype', choices=('float32', 'bfloat16'), default='float32', help='Frozen-weight storage dtype; trainable weights and EMA updates remain float32')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Recompute transformer activations during backward to reduce memory')
    parser.add_argument('--method', choices=('sdapt', 'fedavg', 'mocosfl', 'no-dapt'), default='sdapt')
    parser.add_argument('--stage', choices=('all', 'adapt', 'evaluate'), default='all')
    parser.add_argument('--data_root', type=Path, default=Path('data'))
    parser.add_argument('--pretrained', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, help='Adapted checkpoint for --stage evaluate')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--run_name', help='Optional name for this experiment directory')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'float32', 'bfloat16'), default='auto')
    parser.add_argument('--nr_of_clients', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=15, help='Images per client per adaptation micro-batch')
    parser.add_argument('--nr_of_epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=250, help='Micro-batches per client; FedAvg uses local steps per round')
    parser.add_argument('--local_steps', type=int, default=50, help='FedAvg local micro-batches per round')
    parser.add_argument('--accumulation_steps', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_learning_rate', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--teacher_warmup_epochs', type=int, default=30)
    parser.add_argument('--freeze_head_epochs', type=int, default=2)
    parser.add_argument('--split_layer', type=int, default=1, help='Number of frozen client transformer blocks for SDAPT')
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=float, default=32)
    parser.add_argument('--unfreeze_last', type=int, help='Fully trainable server blocks; 2 for CheXpert, 1 otherwise')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--shots', type=int, nargs='+', default=[4, 8, 16, 0], help='Shots per class per client; 0 selects full-data linear probing')
    parser.add_argument('--head_epochs', type=int, default=25)
    parser.add_argument('--head_learning_rate', type=float, default=1e-3)
    parser.add_argument('--head_batch_size', type=int, default=64)
    parser.add_argument('--feature_cache_dir', type=Path, help='Parent directory for temporary disk-backed downstream features')
    args = parser.parse_args(argv)
    spec = get_model_spec(args.model)
    args.pretrained = args.pretrained if args.pretrained is not None else Path(spec.default_checkpoint)
    for name in ('nr_of_clients', 'batch_size', 'nr_of_epochs', 'steps_per_epoch', 'local_steps', 'accumulation_steps', 'lora_rank', 'head_epochs', 'head_batch_size'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    for name in ('learning_rate', 'min_learning_rate', 'lora_alpha', 'head_learning_rate'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name} must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('--weight_decay must be finite and non-negative')
    if args.min_learning_rate > args.learning_rate:
        parser.error('--min_learning_rate must not exceed --learning_rate')
    if not 0 <= args.split_layer < spec.depth:
        parser.error(f'--split_layer must be between 0 and {spec.depth - 1} for {args.model}')
    args.unfreeze_last = args.unfreeze_last if args.unfreeze_last is not None else (2 if args.dataset == 'chexpert' else 1)
    if not 1 <= args.unfreeze_last <= spec.depth - args.split_layer:
        parser.error('--unfreeze_last must fit entirely on the server')
    if any(getattr(args, name) < 0 for name in ('num_workers', 'warmup_epochs', 'teacher_warmup_epochs', 'freeze_head_epochs', 'random_seed')):
        parser.error('Worker counts, warmup lengths, freeze epochs and seed must be non-negative')
    if any(k < 0 for k in args.shots) or len(set(args.shots)) != len(args.shots):
        parser.error('--shots must contain distinct non-negative integers')
    if args.method == 'no-dapt' and args.stage == 'adapt':
        parser.error('no-dapt performs downstream evaluation only')
    if args.stage == 'evaluate' and args.method != 'no-dapt' and args.checkpoint is None:
        parser.error('--stage evaluate requires --checkpoint for adapted methods')
    if args.checkpoint is not None and args.stage != 'evaluate':
        parser.error('--checkpoint is used with --stage evaluate')
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in ('.', '..')):
        parser.error('--run_name must be a single directory name')
    return args


def schedule(step, total, start, end, warmup=0):
    """Warm up then cosine-decay, including well-defined one-step runs."""
    if total <= 1:
        return float(start)
    warmup = min(warmup, total - 1)
    if step < warmup:
        return float(start * (step + 1) / warmup)
    fraction = min(max((step - warmup) / max(total - warmup - 1, 1), 0.0), 1.0)
    return float(end + .5 * (start - end) * (1 + math.cos(math.pi * fraction)))


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select --device cpu or --device mps.')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; select --device cpu or --device cuda.')
    return torch.device(name)


def atomic_json(target, payload):
    target = Path(target)
    text = json.dumps(payload, indent=2, allow_nan=False, default=str) + '\n'
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.tmp')
    try:
        temp.write_text(text)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from None


def _state_on_cpu(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _state_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())


def _save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _precision(args, device):
    use_bf16 = args.precision == 'bfloat16' or (args.precision == 'auto' and device.type == 'cuda' and torch.cuda.is_bf16_supported())
    uses_bf16 = use_bf16 or args.parameter_dtype == 'bfloat16'
    if uses_bf16 and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support bfloat16; use --precision float32 --parameter_dtype float32.')
    if uses_bf16 and device.type == 'mps':
        raise ValueError('Use --precision float32 --parameter_dtype float32 on MPS.')
    return torch.bfloat16 if use_bf16 else torch.float32


def _loader(dataset, indices, args, client_id):
    from torch.utils.data import DataLoader, Subset
    generator = torch.Generator().manual_seed(args.random_seed + 1009 * client_id)
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size,
                      shuffle=True, drop_last=False, num_workers=args.num_workers,
                      generator=generator, worker_init_fn=seed_worker)


class CyclingLoader:
    """Deterministically reshuffle each local pass, without dropping tail samples."""
    def __init__(self, loader):
        if not len(loader):
            raise ValueError('An adaptation client has no training samples')
        self.loader, self.iterator = loader, iter(loader)

    def next(self):
        try:
            return next(self.iterator)[0]
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)[0]


def _protocol(args, train):
    manifest = (args.data_root / args.dataset / 'train.jsonl').read_bytes()
    return {'dataset': args.dataset, 'model': args.model, 'method': args.method, 'nr_of_clients': args.nr_of_clients,
            'random_seed': args.random_seed, 'split_layer': args.split_layer,
            'train_manifest_sha256': hashlib.sha256(manifest).hexdigest(),
            'train_examples': len(train), 'classes': list(train.classes)}


def _adapt(args, dataset, partitions, device, precision, directory):
    from .model import build_split_encoder
    from .training import SDAPTTrainer, DINOLocalTrainer, MoCoSFLTrainer, StateAccumulator
    student = build_split_encoder(args.pretrained, method=args.method, unfreeze_last=args.unfreeze_last,
                                  lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                                  split_depth=args.split_layer, model=args.model,
                                  parameter_dtype=getattr(torch, args.parameter_dtype),
                                  gradient_checkpointing=args.gradient_checkpointing).to(device)
    kwargs = {'accumulation_steps': args.accumulation_steps, 'transport_dtype': precision}
    if args.method == 'mocosfl':
        trainer = MoCoSFLTrainer(student, num_clients=args.nr_of_clients, **kwargs)
    else:
        trainer = (DINOLocalTrainer if args.method == 'fedavg' else SDAPTTrainer)(student, **kwargs)
    def optimizer():
        return torch.optim.AdamW(trainer.trainable_parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, betas=(.9, .999))
    opt = optimizer() if args.method != 'fedavg' else None
    loaders = [CyclingLoader(_loader(dataset, p, args, i)) for i, p in enumerate(partitions)]
    if args.method == 'fedavg':
        updates_per_epoch = sum(math.ceil(min(args.local_steps, args.steps_per_epoch - offset) / args.accumulation_steps)
                                for offset in range(0, args.steps_per_epoch, args.local_steps))
    else:
        updates_per_epoch = math.ceil(args.steps_per_epoch / args.accumulation_steps)
    total = args.nr_of_epochs * updates_per_epoch
    totals = {'uplink_bytes': 0, 'paper_uplink_bytes': 0, 'downlink_bytes': 0,
              'examples': 0, 'optimizer_steps': 0}
    started = time.monotonic()
    def autocast():
        return torch.autocast(device.type, dtype=precision) if precision == torch.bfloat16 else contextlib.nullcontext()
    def controls(step):
        lr = schedule(step, total, args.learning_rate, args.min_learning_rate, args.warmup_epochs * updates_per_epoch)
        momentum = 1.0 - .5 * (1.0 - .994) * (1.0 + math.cos(math.pi * step / total))
        warmup = args.teacher_warmup_epochs * updates_per_epoch
        temperature = .04 + .03 * min(step / max(warmup, 1), 1.0) if warmup else .07
        return lr, momentum, temperature
    def record(row, epoch, step):
        for key in ('uplink_bytes', 'paper_uplink_bytes', 'downlink_bytes', 'examples'):
            totals[key] += int(row.get(key, 0))
        totals['optimizer_steps'] += int(row['optimizer_step'])
        row = {'epoch': epoch + 1, 'step': step + 1, **row}
        with (directory / 'adaptation.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    for epoch in range(args.nr_of_epochs):
        losses = []
        if args.method == 'fedavg':
            # A round starts every client from exactly the same global state.
            # One local trainer is reused, avoiding N copies of the encoder.
            completed_updates = 0
            for offset in range(0, args.steps_per_epoch, args.local_steps):
                global_state = _state_on_cpu(trainer)
                accumulator = StateAccumulator()
                steps = min(args.local_steps, args.steps_per_epoch - offset)
                for client_id, loader in enumerate(loaders):
                    trainer.load_state_dict(global_state, strict=True)
                    opt = optimizer()
                    for local_step in range(steps):
                        step = epoch * args.steps_per_epoch + offset + local_step
                        lr, momentum, temperature = controls(epoch * updates_per_epoch + completed_updates + local_step // args.accumulation_steps)
                        for group in opt.param_groups:
                            group['lr'] = lr
                        with autocast():
                            row = trainer.train_step([loader.next()], optimizer=opt,
                                teacher_temperature=temperature, ema_momentum=momentum,
                                flush=local_step == steps - 1, freeze_head=epoch < args.freeze_head_epochs)
                        losses.append(row['loss'])
                        record({**row, 'client_id': client_id}, epoch, step)
                    accumulator.add(trainer.state_dict(), len(partitions[client_id]))
                    del opt
                trainer.load_state_dict(accumulator.average(), strict=True)
                # Both the online and EMA teacher/head states are federated.
                wire_bytes = _state_bytes(global_state) * args.nr_of_clients
                totals['uplink_bytes'] += wire_bytes
                totals['downlink_bytes'] += wire_bytes
                completed_updates += math.ceil(steps / args.accumulation_steps)
                del global_state, accumulator
        else:
            for local_step in range(args.steps_per_epoch):
                step = epoch * args.steps_per_epoch + local_step
                lr, momentum, temperature = controls(epoch * updates_per_epoch + local_step // args.accumulation_steps)
                for group in opt.param_groups:
                    group['lr'] = lr
                extra = {'freeze_head': epoch < args.freeze_head_epochs} if args.method == 'sdapt' else {}
                with autocast():
                    row = trainer.train_step([loader.next() for loader in loaders], optimizer=opt,
                        teacher_temperature=temperature, ema_momentum=momentum,
                        flush=local_step == args.steps_per_epoch - 1, **extra)
                record(row, epoch, step); losses.append(row['loss'])
        print(f"Epoch {epoch + 1}/{args.nr_of_epochs} · loss {np.mean(losses):.4f} · {totals['optimizer_steps']} updates", flush=True)
        atomic_json(directory / 'progress.json', {'completed_epochs': epoch + 1, 'loss': float(np.mean(losses)), **totals})
    state = trainer.teacher_state_dict()
    clients = [_state_on_cpu(client) for client in trainer.teacher_client_models] if args.method == 'mocosfl' else None
    totals['elapsed_seconds'] = time.monotonic() - started
    totals['trainable_parameters'] = sum(p.numel() for p in trainer.trainable_parameters())
    # No optimizer state is carried into frozen-backbone downstream training.
    del trainer, student, loaders
    if args.method != 'fedavg':
        del opt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return state, clients, totals


def run(args):
    from .augment import MultiCrop, eval_transform
    from .data import load_dataset, iid_partition
    from .evaluation import EvaluationConfig, run_evaluations
    from .model import build_encoder, ViTClient
    seed_everything(args.random_seed)
    device = resolve_device(args.device)
    precision = _precision(args, device)
    spec = get_model_spec(args.model)
    train = load_dataset(args.dataset, args.data_root, 'train', eval_transform(args.dataset, patch_size=spec.patch_size))
    partitions = iid_partition(train, args.nr_of_clients, args.random_seed)
    protocol = _protocol(args, train)
    name = args.run_name or f'{args.method}-{args.model}-{args.dataset}-seed{args.random_seed}' + ('-evaluation' if args.stage == 'evaluate' else '')
    directory = args.output / name
    if directory.exists():
        raise FileExistsError(f'Run already exists: {directory}. Choose another --run_name.')
    if args.checkpoint is None and not args.pretrained.is_file():
        raise FileNotFoundError(f'Pretrained weights missing: {args.pretrained}. Run python scripts/prepare_model.py')
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        for key, expected in protocol.items():
            if checkpoint.get('protocol', {}).get(key) != expected:
                raise ValueError(f'Checkpoint protocol mismatch for {key}; use its dataset, client count, seed and split.')
    test = load_dataset(args.dataset, args.data_root, 'test', eval_transform(args.dataset, patch_size=spec.patch_size)) if args.stage != 'adapt' else None
    if test is not None and set(train.paths) & set(test.paths):
        raise ValueError('Training and test images overlap')
    directory.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    atomic_json(directory / 'config.json', {**config, 'resolved_device': str(device), 'resolved_precision': str(precision), 'protocol': protocol})
    atomic_json(directory / 'partitions.json', [indices.tolist() for indices in partitions])
    totals = None
    if args.method == 'no-dapt':
        encoder = build_encoder(args.pretrained, model=args.model, parameter_dtype=getattr(torch, args.parameter_dtype)).to(device)
        client_states = None
    else:
        if checkpoint is None:
            adaptation = load_dataset(args.dataset, args.data_root, 'train', MultiCrop(args.dataset, local_crops=0 if args.method == 'mocosfl' else 8, patch_size=spec.patch_size))
            state, client_states, totals = _adapt(args, adaptation, partitions, device, precision, directory)
            checkpoint = {'encoder': state, 'client_states': client_states, 'protocol': protocol,
                          'config': config, 'adaptation': totals}
            _save_checkpoint(directory / 'teacher.pt', checkpoint)
            atomic_json(directory / 'adaptation-summary.json', totals)
        client_states = checkpoint.get('client_states')
        if args.stage == 'adapt':
            print(f'Adapted teacher: {directory / "teacher.pt"}', flush=True)
            return {'directory': str(directory), 'adaptation': totals}
        encoder = build_encoder(state_dict=checkpoint['encoder'], model=args.model, parameter_dtype=getattr(torch, args.parameter_dtype)).to(device)
        del checkpoint
    callback = None
    if args.method == 'mocosfl':
        if not isinstance(client_states, list) or len(client_states) != args.nr_of_clients:
            raise ValueError('MoCoSFL checkpoint requires the EMA frontend of every client')
        client_view = ViTClient(encoder.backbone, args.split_layer)
        def callback(client_id):
            client_view.load_state_dict(client_states[client_id], strict=True)
            return encoder
    configs = [EvaluationConfig(shots=shot or None, epochs=args.head_epochs,
                               learning_rate=args.head_learning_rate, batch_size=args.head_batch_size,
                               seed=args.random_seed, num_workers=args.num_workers,
                               feature_cache_dir=args.feature_cache_dir) for shot in args.shots]
    evaluations = run_evaluations(encoder, train, test, partitions, device, configs, encoder_for_client=callback)
    summaries = {}
    for shot, result in zip(args.shots, evaluations):
        label = f'{shot}-shot' if shot else 'linear-probe'
        _save_checkpoint(directory / f'heads-{label}.pt', {'heads': result.pop('heads'), 'config': result['config']})
        atomic_json(directory / f'{label}.json', result)
        summaries[label] = result['metrics']['global']
        print(f'{label}: {json.dumps(summaries[label], allow_nan=False)}', flush=True)
    atomic_json(directory / 'summary.json', {'protocol': protocol, 'evaluation': summaries, 'adaptation': totals})
    return {'directory': str(directory), 'evaluation': summaries, 'adaptation': totals}
