#!/bin/bash
#SBATCH --job-name=graph_embed
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --partition=wholenode           # Use full node for RAM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # Sequential script
#SBATCH --cpus-per-task=128             # Reserve all cores so no one else shares the node
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"

# 1. Setup Environment
module use "$HOME/privatemodules"

# Ensure 'uv' is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Create output directories
mkdir -p slurm-out
mkdir -p data/encodings/graphs

# 2. Run Graph Embedding
echo "Starting Graph Embedding Generation (Trajectories + Goals)..."
uv run python -m code.encoding-generation.generate_graph_embeddings --iterations 2

echo "Job finished at $(date)"
