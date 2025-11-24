#!/bin/bash
#SBATCH --job-name=train_eval
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --partition=gpu                 # Request GPU partition
#SBATCH --gpus=1                        # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # CPU cores for DataLoader
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"

# 1. Setup Environment
module use "$HOME/privatemodules"
module load conda
module load modtree/gpu

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Define Domains to process
DOMAINS=("blocks" "gripper" "logistics" "visitall-from-everywhere")

# Hyperparameters
EPOCHS=50
BATCH_SIZE=32
HIDDEN_DIM=256
LR=0.01

for DOMAIN in "${DOMAINS[@]}"; do
    echo "*** PROCESSING DOMAIN: $DOMAIN ***"

    # 2. Training
    echo "[1/2] Starting Training..."
    uv run python -m code.modeling.train \
        --domain "$DOMAIN" \
        --data_dir "data/encodings/graphs" \
        --save_dir "checkpoints" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --lr $LR

    # Check if training succeeded by looking for the best checkpoint
    CHECKPOINT="checkpoints/${DOMAIN}_lstm_best.pt"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint $CHECKPOINT not found. Training failed."
        continue
    fi

    # 3. Inference (Latent Space Search)
    echo "[2/2] Starting Inference..."

    # This script automatically runs on both 'test-interpolation' and 'test-extrapolation'
    uv run python -m code.modeling.inference \
        --domain "$DOMAIN" \
        --pddl_dir "data/pddl" \
        --states_dir "data/states" \
        --checkpoint "$CHECKPOINT" \
        --results_dir "results" \
        --hidden_dim $HIDDEN_DIM \
        --max_steps 100

    echo "Finished processing $DOMAIN"
    echo ""
done

echo "Job finished at $(date)"
