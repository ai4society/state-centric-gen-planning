#!/bin/bash
#SBATCH --job-name=aggregate_results
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"

# Load environment
module use "$HOME/privatemodules"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 1. Generate Markdown Table (for reading)
echo "Generating Markdown Summary..."
uv run python -m code.analysis.aggregate_results \
    --results_dir results \
    --format markdown \
    --output results/summary_table.md

echo "Aggregation complete. Check 'results/summary_table.md' for the summary table."
