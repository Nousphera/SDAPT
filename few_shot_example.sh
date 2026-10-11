#!/usr/bin/env bash
#SBATCH --job-name=few_shot_example
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
## activate python env here

export PYTHONPATH="$REPO_ROOT"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 100))}"

FEW_SHOT_CONFIG="$REPO_ROOT/src/configs/few_shot_dinov2.yaml"
FMOW_CHECKPOINT="${FMOW_CHECKPOINT:-$REPO_ROOT/results/adaption_fmow/eval/training_100/teacher_checkpoint.pth}"
CHEXPERT_CHECKPOINT="${CHEXPERT_CHECKPOINT:-$REPO_ROOT/results/adaption_chexpert/eval/training_100/teacher_checkpoint.pth}"

# Active example: FMoW few-shot evaluation.
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" src/usl_few_shot_main.py \
    --config_file="$FEW_SHOT_CONFIG" \
    --pretrained_weight_path_server="$FMOW_CHECKPOINT" \
    --checkpoint_key_server=teacher \
    --dataset=fmow \
    --nr_of_classes=62 \
    --logs_dir="results/few_shot_fmow"
    --spli

# CheXpert few-shot evaluation:
# cp -a "$REPO_ROOT/dataman/train" "${TMPDIR:?}/"
# torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" src/usl_few_shot_multi_label.py \
#     --config_file="$FEW_SHOT_CONFIG" \
#     --pretrained_weight_path_server="$CHEXPERT_CHECKPOINT" \
#     --checkpoint_key_server=teacher \
#     --dataset=chexpert \
#     --logs_dir="results/few_shot_chexpert"
