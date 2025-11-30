#!/bin/bash
#SBATCH --job-name=plansformer_eval
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --account=nairr250014-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"

# Load your environment (adapt to your cluster)
module load modtree/gpu
module load gcc/11.2.0
module use "$HOME/privatemodules"

# If you're using uv (like other scripts):
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 1) Run Plansformer to generate plans
uv run python -m code.plansformer.generate_plans \
    --data-dir data/pddl \
    --output-dir data/plansformer \
    --splits validation test-interpolation test-extrapolation \
    --num-beams 2 \
    --max-new-tokens 64 \
    --log-jsonl results/plansformer/plansformer_generation_log.jsonl

# 2) Evaluate against Fast Downward plans
uv run python -m code.plansformer.evaluate_plans \
    --pddl-dir data/pddl \
    --gold-plans-dir data/plans \
    --pred-plans-dir data/plansformer \
    --results-dir results/plansformer

echo "Job finished at $(date)"
