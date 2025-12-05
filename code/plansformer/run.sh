#!/bin/bash
#SBATCH --job-name=pddl-dset
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --mail-user=jaaydin@email.sc.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
echo "Hello"
module load conda
module load modtree/gpu
conda activate john_test
cd /anvil/projects/x-nairr250014/state-centric-gen-planning/code/plansformer
python main.py \
  --val_path /home/john/bin/val \
  --data_path ../../data \
  --save_path ../../results/plansformer \
  --model_path "/anvil/projects/x-nairr250014/plansformer/codet5-base/model_files"
