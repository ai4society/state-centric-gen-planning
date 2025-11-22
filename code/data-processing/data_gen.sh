#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --account=nairr250014-ai
#SBATCH --partition=wholenode           # Use full node (128 cores)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128           # 128 cores available
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"
# 1. Setup Environment
# $HOME: /home/x-ngupta6
module purge
module use "$HOME/privatemodules"

# Ensure 'uv' is in PATH (if installed in $HOME/.cargo/bin or similar)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Prevent numpy/FD from trying to multithread inside each process
# We want 128 separate processes, not 1 process using 128 threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Create slurm-out directory if it doesn't exist
mkdir -p slurm-out

# 2. Ensure correct files: run lowercasing
uv run python -m code.data-processing.utils.lowercase_pddl

# 3. Run Plan Generation
# We use 'uv run' which automatically detects the .venv in the project root.
# We set workers to 120 (leaving a few cores for system overhead).
echo "Starting Plan Generation..."
uv run python -m scripts.generate_plans --workers 120

# 4. Run State Generation
# Only runs if previous step succeeded (&&) or just run sequentially
echo "Starting State Generation..."
uv run python -m scripts.generate_states --workers 120

echo "Job finished at $(date)"
