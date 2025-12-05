#!/bin/bash
#SBATCH --job-name=plansformer
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --mail-user=jaaydin@email.sc.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu-debug
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

echo "Job started on $(hostname) at $(date)"

# Load env
module load conda
module load modtree/gpu

cd /anvil/projects/x-nairr250014/state-centric-gen-planning

# Generate plansformer results
uv run python -m code.plansformer.inference_plansformer \
    --val_path ./VAL/bin/Validate \
    --data_path ./data/pddl \
    --save_path ./results/plansformer \
    --model_path "./plansformer-model/codet5-base/model_files"

