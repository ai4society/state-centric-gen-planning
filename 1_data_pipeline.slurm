#!/bin/bash
#SBATCH --job-name=data_pipeline
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --partition=wholenode           # Use full node for heavy generation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"

# 1. Setup Environment
module use "$HOME/privatemodules"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=1

mkdir -p slurm-out

# 2. Generate Dataset (PDDL Files)
echo "----------------------------------------------------------------"
echo "STEP 1: Generating PDDL Dataset (Procedural Generation)"
echo "----------------------------------------------------------------"
# This creates data/pddl/{domain}/{split}/*.pddl
uv run python -m code.data-generation.generate_dataset

# 3. Lowercase PDDL (Safety check)
echo "----------------------------------------------------------------"
echo "STEP 2: Normalizing PDDL (Lowercasing)"
echo "----------------------------------------------------------------"
uv run python -m code.data-processing.utils.lowercase_pddl

# 4. Generate Plans (Fast Downward)
echo "----------------------------------------------------------------"
echo "STEP 3: Generating Plans (Fast Downward)"
echo "----------------------------------------------------------------"
# Uses 120 cores to solve the thousands of problems generated in Step 1
uv run python -m code.data-processing.generate_plans --workers 120

# 5. Generate State Trajectories (VAL)
echo "----------------------------------------------------------------"
echo "STEP 4: Generating State Trajectories (VAL)"
echo "----------------------------------------------------------------"
uv run python -m code.data-processing.generate_states --workers 120

# 6. Generate Graph Embeddings (WL)
echo "----------------------------------------------------------------"
echo "STEP 5: Generating Graph Embeddings (WL)"
echo "----------------------------------------------------------------"
# This runs sequentially per domain but is fast enough for 1000s of small graphs.
uv run python -m code.encoding-generation.generate_graph_embeddings --iterations 2

echo "Pipeline finished at $(date)"
