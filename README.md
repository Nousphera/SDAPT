# SDAPT

**Split-based Domain-Adaptive Pretraining for Vision Foundation Models**

This repository contains the training and evaluation code for **SDAPT**, the
method introduced in the accompanying thesis, and for its experimental
baselines.

The code supports experiments in
split learning with Vision Foundation Models (VFMs). It includes the SDAPT
domain-adaptive pretraining setup, as well as federated and MoCo-SFL-style
adaptation baselines.

The main model is DINOv2 ViT-B/14. The model is split between clients and a
server: clients keep the image data and run the early layers, while the server
runs the remaining backbone and the self-supervised objective. During
downstream evaluation, clients keep their labels and train task-specific heads
using the frozen adapted representation.

## What is included

- **SDAPT**: server-side DINOv2 adaptation using aggregated smashed features,
	frozen client layers, LoRA adapters, and a DINO/iBOT/Koleo objective.
- **FedAvg adaptation**: a federated DINOv2 adaptation baseline in which each
	client trains a complete local model.
- **MoCoSFL adaptation**: a split self-supervised MoCo-v3 baseline.
- **Downstream evaluation**: few-shot kNN evaluation, linear probing, and
	fine-tuning for classification and medical-image tasks.
- **Datasets**: fMoW, CheXpert/CheXlocalize, QuickDraw/DomainNet, and the
	chest X-ray segmentation datasets used by the experiments.

## Repository layout

```text
src/
	configs/       YAML configurations for adaptation and evaluation
	data/          Dataset implementations, loaders, and client partitioning
	models/        DINOv2, MoCo-v3, and split model definitions
	trainers/      SDAPT, FedAvg, MoCoSFL, and downstream trainers
	evaluators/    Evaluation implementations
	*_main.py      Command-line entry points
dataman/         Local CheXpert metadata and training data references
datasets/        Dataset-specific files and evaluation data
checkpoints/     Pretrained and experiment checkpoints
results/         Experiment outputs and evaluation results
```

## Requirements

The code is intended for Linux with Python 3.11 and a CUDA-capable PyTorch
installation. The checked-in `requirements.txt` contains the Python
dependencies, including PyTorch 2.4.1 and torchvision 0.19.1.

Create an environment and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On a cluster or other shared compute system, load the Python and CUDA
dependencies required by your environment before running the examples.

Set the repository on the Python path before running an entry point:

```bash
export PYTHONPATH="$PWD"
```

## Data and checkpoints

The repository includes the DINOv2 ViT-B/14 checkpoint referenced by the
default adaptation configuration:

```text
checkpoints/dinov2_vitb14_pretrain.pth
```

Large datasets are not all stored in this repository. Update the
`dataset_paths` section in [src/configs/adaption.yaml](src/configs/adaption.yaml)
or provide the corresponding environment variables:

- `WILDS_ROOT`: root directory for the WILDS/fMoW data.
- `DOMAINNET_ROOT`: root directory containing QuickDraw/DomainNet images and
	train/test manifests.
- `HPA_ROOT`: optional HPA image and manifest root.
- `TMPDIR`: path to the CheXpert image data directory.

For CheXpert, the repository expects the label metadata and CheXlocalize files
referenced in the default configuration. Set `TMPDIR` to the directory
containing the CheXpert images, or update the corresponding path in the
configuration.

The default configs use paths relative to the repository where possible. The
evaluation configs still contain an absolute checkpoint path in some fields;
replace it with the path to your local copy when needed.

## Running experiments

All commands below assume the repository root is the current directory and
that the environment has been activated.

### 1. Domain-adaptive pretraining

Run SDAPT on fMoW with the default configuration:

```bash
export PYTHONPATH="$PWD"
torchrun --nproc_per_node=1 --master_port=29500 \
	src/sdapt_adaption_main.py \
	--config_file=src/configs/adaption.yaml \
	--dataset=fmow \
	--eval_store_dir=results/adaption_sdapt_fmow \
	--alpha=100
```

The main adaptation entry points are:

```text
src/sdapt_adaption_main.py       SDAPT / split DINOv2 adaptation
src/fedavg_adaption_main.py     FedAvg DINOv2 adaptation
src/mocosfl_adaption_main.py    MoCoSFL adaptation
```

Use `--dataset=chexpert` or the dataset name supported by the corresponding
dataset implementation when switching domains. `--alpha` controls the
Dirichlet client partition; large values approximate IID partitions.

The default adaptation configuration trains for 100 epochs with 10 clients,
250 iterations per epoch, eight local crops, and one fully unfrozen server
block plus LoRA adaptation in the remaining server blocks. Adjust these
values in the YAML file for a small smoke test or a different experiment.

### 2. Few-shot evaluation

After adaptation, evaluate a saved teacher checkpoint with a local kNN head:

```bash
torchrun --nproc_per_node=1 --master_port=29501 \
	src/usl_few_shot_main.py \
	--config_file=src/configs/few_shot_dinov2.yaml \
	--pretrained_weight_path_server=results/adaption_sdapt_fmow/eval/training_100/teacher_checkpoint.pth \
	--checkpoint_key_server=teacher \
	--dataset=fmow \
	--nr_of_classes=62 \
	--logs_dir=results/few_shot_fmow
```

For CheXpert multi-label evaluation, use
`src/usl_few_shot_multi_label.py` and configure the CheXpert paths first.

### 3. Linear probing or fine-tuning

Evaluate a frozen adapted backbone with a client-side task head:

```bash
torchrun --nproc_per_node=1 --master_port=29502 \
	src/usl_finetuning_main.py \
	--config_file=src/configs/finetuning_evaluation.yaml \
	--pretrained_weight_path_server=results/adaption_sdapt_fmow/eval/training_100/teacher_checkpoint.pth \
	--checkpoint_key_server=teacher \
	--freeze_server_backbone \
	--dataset=fmow \
	--nr_of_classes=62 \
	--logs_dir=results/finetuning_fmow
```

Remove `--freeze_server_backbone` when a fine-tuned server backbone is
intended. For segmentation and multi-label classification, use the relevant
task-specific entry point and dataset configuration under `src`.

## Reproducibility notes

- Client partitions are controlled by `partitioner`, `alpha`, `nr_of_clients`,
	and `random_seed` in the YAML configurations.
- Adaptation checkpoints are saved at the configured evaluation epochs and can
	be passed to the downstream entry points using the `teacher` checkpoint key.
- SDAPT and the baselines should be compared using the same split, adaptation
	schedule, and downstream protocol.
- The experiment outputs already present under `results/` and
	`results_paper/` are reference artifacts; new runs should use separate
	output directories to avoid overwriting them.

## Thesis summary

The associated experiments study whether unlabeled, distributed in-domain
images can adapt a pretrained VFM without moving raw images or labels off the
client. SDAPT improves downstream performance over the non-adapted VFM across
the evaluated fMoW, CheXpert, and QuickDraw settings. MoCoSFL is generally
stronger at very small label or communication budgets, while DINOv2-based
self-distillation benefits more from additional adaptation and labeled data.

## Citation

If you use this code or the SDAPT method, cite the accompanying thesis:

```text
Stan van Honk, "Domain Adaptive Pretraining in Split Learning for Vision
Foundation Models," Master Thesis, Eindhoven University of Technology, 2026.
```
