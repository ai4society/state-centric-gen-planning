#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --account=nairr250014-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"

# 1. Setup Environment
module load modtree/gpu
module load gcc/11.2.0
module use "$HOME/privatemodules"

# Ensure 'uv' is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv run python -m code.encoding-generation.generate_graph_embeddings --domain "blocks"
uv run python -m code.modeling.train --domain "blocks" --epochs 5
uv run python -m code.modeling.inference --domain "blocks" --checkpoint "checkpoints/blocks_lstm_best.pt"

echo "Job finished at $(date)"
