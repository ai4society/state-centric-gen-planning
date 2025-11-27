#!/bin/bash
#SBATCH --job-name=train_eval
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --account=nairr250014-ai    # Charge AI allocation
#SBATCH --partition=ai              # Use the AI partition (H100s)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"

# 1. Setup Environment
module load modtree/gpu
module load gcc/11.2.0
module use "$HOME/privatemodules"

# Ensure 'uv' is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Define Domains
DOMAINS=("blocks" "gripper" "logistics" "visitall-from-everywhere")

# Hyperparameters
EPOCHS=125
BATCH_SIZE=32
HIDDEN_DIM=256
LR=0.01

for DOMAIN in "${DOMAINS[@]}"; do
    echo "*** PROCESSING DOMAIN: $DOMAIN ***"

    echo "[0/2] Starting Encoding..."
    uv run python -m code.encoding-generation.generate_graph_embeddings --domain "$DOMAIN"

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
        --checkpoint "$CHECKPOINT" \
        --pddl_dir "data/pddl" \
        --data_dir "data" \
        --results_dir "results" \
        --hidden_dim $HIDDEN_DIM \
        --max_steps 100 \
        --beam_width 3

    echo "Finished processing $DOMAIN"
    echo ""
done

echo "Job finished at $(date)"
