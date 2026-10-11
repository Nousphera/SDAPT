#!/usr/bin/env bash
#SBATCH --job-name=finetuning_example
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:30:00
#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

export PYTHONPATH="$REPO_ROOT"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 100))}"

FINETUNING_CONFIG="$REPO_ROOT/src/configs/finetuning_evaluation.yaml"
FMOW_CHECKPOINT="${FMOW_CHECKPOINT:-$REPO_ROOT/results/adaption_fmow/eval/training_100/teacher_checkpoint.pth}"
CHEXPERT_CHECKPOINT="${CHEXPERT_CHECKPOINT:-$REPO_ROOT/results/adaption_chexpert/eval/training_100/teacher_checkpoint.pth}"

# Active example: FMoW linear probing with a frozen backbone.
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" src/usl_finetuning_main.py \
    --config_file="$FINETUNING_CONFIG" \
    --pretrained_weight_path_server="$FMOW_CHECKPOINT" \
    --checkpoint_key_server=teacher \
    --freeze_server_backbone \
    --dataset=fmow \
    --nr_of_classes=62 \
    --logs_dir="results/finetuning_fmow"

# CheXpert fine-tuning / linear probing:
# cp -a "$REPO_ROOT/dataman/train" "${TMPDIR:?}/"
# torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" src/usl_finetuning_main.py \
#     --config_file="$FINETUNING_CONFIG" \
#     --pretrained_weight_path_server="$CHEXPERT_CHECKPOINT" \
#     --checkpoint_key_server=teacher \
#     --dataset=chexpert \
#     --logs_dir="results/finetuning_chexpert"
