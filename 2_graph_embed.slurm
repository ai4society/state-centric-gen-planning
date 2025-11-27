#!/bin/bash
#SBATCH --job-name=graph_embed
#SBATCH --output=slurm-out/%x_%A_%a.out   # %A=JobId, %a=ArrayIdx
#SBATCH --error=slurm-out/%x_%A_%a.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                 # We only need 1 core per domain
#SBATCH --mem=16G                         # 16GB RAM is safe for WL
#SBATCH --time=4:00:00                    # Reduced time (since we run domains in parallel)
#SBATCH --array=0-3                       # Run 4 jobs (indices 0, 1, 2, 3)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"
echo "Processing Array Task ID: $SLURM_ARRAY_TASK_ID"

# 1. Setup Environment
module load modtree/gpu
module load gcc/11.2.0 # Ensure GCC compatibility
module use "$HOME/privatemodules"

# Ensure 'uv' is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Create output directories
mkdir -p slurm-out
mkdir -p data/encodings/graphs

# 2. Define Domains Mapping
# Map the SLURM array index (0-3) to domain names
DOMAINS=("blocks" "gripper" "logistics" "visitall-from-everywhere")
CURRENT_DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$CURRENT_DOMAIN" ]; then
    echo "Error: Invalid Array Index"
    exit 1
fi

echo "Running Graph Embedding for Domain: $CURRENT_DOMAIN"

# 3. Run Graph Embedding (Sequential per domain, but Parallel across domains)
uv run python -m code.encoding-generation.generate_graph_embeddings --domain "$CURRENT_DOMAIN"

echo "Job finished at $(date)"
