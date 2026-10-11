#!/usr/bin/env bash
#SBATCH --job-name=adaption_example
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=32:00:00
#SBATCH --partition=gpu_h100
#SBATCH --constraint=scratch-node

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

export PYTHONPATH="$REPO_ROOT"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 100))}"

ADAPTATION_CONFIG="$REPO_ROOT/src/configs/adaption.yaml"
FMOW_CONFIG="$(mktemp)"
trap 'rm -f "$FMOW_CONFIG"' EXIT
sed 's/^dataset: chexpert$/dataset: fmow/' "$ADAPTATION_CONFIG" > "$FMOW_CONFIG"

# Active example: federated FMoW adaptation.
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" src/fedavg_adaption_main.py \
    --config_file="$FMOW_CONFIG" \
    --dataset=fmow \
    --eval_store_dir="results/adaption_fmow" \
    --alpha=100

torchrun --nproc_per_node=1 --master_port=2500 src/fedavg_adaption_main.py \
    --config_file="/home/svhonk/code/SDAPT/src/configs/adaption.yaml" \
    --dataset=fmow \
    --eval_store_dir="results/adaption_fmow" \
    --alpha=100 \
    --split_depth 1


# CheXpert federated adaptation:
# cp -a "$REPO_ROOT/dataman/train" "${TMPDIR:?}/"
# torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" src/fedavg_adaption_main.py \
#     --config_file="$ADAPTATION_CONFIG" \
#     --dataset=chexpert \
#     --eval_store_dir="results/adaption_chexpert" \
#     --alpha=100

# FMoW MoCo-SFL adaptation.
# Enable the MoCo profile in src/configs/adaption.yaml before running this block.
# torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" src/mocosfl_adaption_main.py \
#     --config_file="$ADAPTATION_CONFIG" \
#     --dataset=fmow \
#     --eval_store_dir="results/adaption_mocosfl_fmow" \
#     --alpha=0.5

# FMoW SDAPT-MoCo adaptation.
# Enable the MoCo profile in src/configs/adaption.yaml before running this block.
# torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" src/sdapt_adaption_main.py \
#     --config_file="$ADAPTATION_CONFIG" \
#     --dataset=fmow \
#     --eval_store_dir="results/adaption_sdapt_moco_fmow" \
#     --alpha=0.5
