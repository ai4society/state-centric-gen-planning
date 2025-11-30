#!/bin/bash
#SBATCH --job-name=train_eval
#SBATCH --output=slurm-out/%x_%A_%a.out
#SBATCH --error=slurm-out/%x_%A_%a.err
#SBATCH --account=nairr250014-ai      # Charge AI allocation
#SBATCH --partition=ai                # Use the AI partition (H100s)
#SBATCH --gres=gpu:1                  # 1 GPU per domain
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3                   # One array index per domain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

echo "Job started on $(hostname) at $(date)"
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"

# 1. Setup Environment
module load modtree/gpu
module load gcc/11.2.0
module use "$HOME/privatemodules"

# Ensure 'uv' is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p slurm-out

# 2. Map array index to domain
DOMAINS=("blocks" "gripper" "logistics" "visitall-from-everywhere")
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$DOMAIN" ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "*** PROCESSING DOMAIN: $DOMAIN ***"

# 3. Hyperparameters
EPOCHS=250
MODES=("state" "delta")
TYPE="gr-lstm"

for MODE in "${MODES[@]}"; do
    echo ""
    echo "*** DOMAIN: $DOMAIN | MODE: $MODE ***"

    # Choose save directory and flags based on mode
    if [ "$MODE" == "delta" ]; then
        SAVE_DIR="checkpoints/delta"
        DELTA_FLAG="--delta"
        TAG="delta"
    else
        SAVE_DIR="checkpoints/state"
        DELTA_FLAG=""
        TAG="state"
    fi

    mkdir -p "$SAVE_DIR"


    # echo ""
    # echo "[0/2] Starting Encoding..."
    # uv run python -m code.encoding-generation.generate_graph_embeddings --domain "$DOMAIN"

    echo ""
    echo "[1/2] Training ($MODE)..."
    uv run python -m code.modeling.train \
        --domain "$DOMAIN" \
        --epochs $EPOCHS \
        --save_dir "$SAVE_DIR" \
        $DELTA_FLAG

    # Check if training succeeded by looking for the best checkpoint
    CHECKPOINT="${SAVE_DIR}/${DOMAIN}_lstm_best.pt"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint $CHECKPOINT not found. Training may have failed. Skipping inference for MODE=$MODE."
        continue
    fi

    echo ""
    echo "[2/2] Inference ($MODE)..."
    uv run python -m code.modeling.inference \
        --domain "$DOMAIN" \
        --checkpoint "$CHECKPOINT" \
        --results_dir "results/${TYPE}" \
        $DELTA_FLAG \
        --tag "$TAG"
done

echo "Job finished at $(date)"
