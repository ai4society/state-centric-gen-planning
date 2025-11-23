#!/bin/bash
#SBATCH --job-name=pddl-dset
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
# %x gives job name
# %A Job array's master job allocation number.
# %a Job array ID (index) number.
# %j Job allocation number.
# %N Node name. Only one file is created, so %N will be replaced by the name of the first node in the job, which is the one that runs the script.
# %u User name.
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
python main.py
