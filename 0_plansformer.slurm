#!/bin/bash
#SBATCH --job-name=plansformer
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --mail-user=jaaydin@email.sc.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=ai
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=nairr250014-ai
echo "Job started on $(hostname) for job $SLURM_JOB_NAME"
# Load env
module load conda
module load modtree/gpu
cd /anvil/projects/x-nairr250014/state-centric-plan
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Generate plansformer results

echo "================================================================="
echo "============================= Run 1 ============================="
echo "================================================================="

uv run python -m code.plansformer.inference_plansformer \
    --val_path ./VAL/bin/Validate \
    --data_path ./data/pddl \
    --save_path ./results/plansformer/r1/ \
    --model_path "./plansformer-model/codet5-base/model_files" \
    --seed 13

echo "================================================================="
echo "============================= Run 2 ============================="
echo "================================================================="

uv run python -m code.plansformer.inference_plansformer \
    --val_path ./VAL/bin/Validate \
    --data_path ./data/pddl \
    --save_path ./results/plansformer/r2/ \
    --model_path "./plansformer-model/codet5-base/model_files" \
    --seed 14

echo "================================================================="
echo "============================= Run 3 ============================="
echo "================================================================="

uv run python -m code.plansformer.inference_plansformer \
    --val_path ./VAL/bin/Validate \
    --data_path ./data/pddl \
    --save_path ./results/plansformer/r3/ \
    --model_path "./plansformer-model/codet5-base/model_files" \
    --seed 15