#!/bin/bash
#SBATCH --job-name=gen_plan_unified
#SBATCH --output=slurm-out/%x_%A_%a.out
#SBATCH --error=slurm-out/%x_%A_%a.err
#SBATCH --account=nairr250014-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3                   # Array for Domains
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niting@email.sc.edu

# CONFIGURATION

# 1. Which Models and Encodings to run? (Space separated: "lstm" "xgboost" "llama")
# Can override this by passing arguments to sbatch, e.g.:
# `sbatch script.slurm "lstm"`
TARGET_MODELS=${1:-"lstm xgboost"}
ENCODING=${2:-"graphs"} # "graphs" or "fsf"

# 2. Which Modes? ("state" "delta")
TARGET_MODES=("state" "delta")

# 3. Hyperparameters (override via env: EPOCHS=100 sbatch ...)
EPOCHS=${EPOCHS:-250}
BATCH_SIZE=${BATCH_SIZE:-32}
HIDDEN_DIM=${HIDDEN_DIM:-256}
LR=${LR:-1e-2}
MAX_DEPTH=${MAX_DEPTH:-8}

# SETUP
echo "Job started on $(hostname) at $(date)"
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"

module load modtree/gpu
module load gcc/11.2.0
module use "$HOME/privatemodules"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p slurm-out

# Map array index to domain
DOMAINS=("blocks" "gripper" "logistics" "visitall-from-everywhere")
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$DOMAIN" ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

DATA_DIR="data/encodings/$ENCODING"

echo " CONFIGURATION SUMMARY"
echo " Domain:    $DOMAIN"
echo " Encoding:  $ENCODING"
echo " Models:    $TARGET_MODELS"
echo " Modes:     ${TARGET_MODES[*]}"
echo " Data Dir:  $DATA_DIR"
echo " Epochs:    $EPOCHS"
echo " Batch Sz:  $BATCH_SIZE"
echo " HiddenDim: $HIDDEN_DIM"
echo " LR:        $LR"

# If FSF, ensure data exists (Quick check/generation)
if [ "$ENCODING" == "fsf" ]; then
    echo "Ensuring FSF embeddings exist for $DOMAIN..."
    uv run python -m code.encoding-generation.generate_fsf_embeddings --domain "$DOMAIN"
fi

# EXECUTION LOOP
for MODEL in $TARGET_MODELS; do
    for MODE in "${TARGET_MODES[@]}"; do

        # 1. Define Paths based on hierarchy
        # Structure: checkpoints/<encoding>/<model>/<mode>/
        # Structure: results/<encoding>/<model>/<mode>/

        CHECKPOINT_DIR="checkpoints/${ENCODING}/${MODEL}_${MODE}"
        RESULTS_DIR="results/${ENCODING}/${MODEL}_${MODE}"

        mkdir -p "$CHECKPOINT_DIR"
        mkdir -p "$RESULTS_DIR"

        # 2. Set Flags
        if [ "$MODE" == "delta" ]; then
            DELTA_FLAG="--delta"
        else
            DELTA_FLAG=""
        fi

        echo ""
        echo "**********************************"
        echo " RUNNING: $MODEL | MODE: $MODE"
        echo "**********************************"

        # 3. Dispatcher Switch
        case $MODEL in

        "lstm")
            # LSTM TRAINING
            echo "[LSTM] Training..."
            uv run python -m code.modeling.train_lstm \
                --domain "$DOMAIN" \
                --data_dir "$DATA_DIR" \
                --save_dir "$CHECKPOINT_DIR" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --hidden_dim "$HIDDEN_DIM" \
                --lr "$LR" \
                $DELTA_FLAG

            # LSTM INFERENCE
            echo "[LSTM] Inference..."
            BEST_PT="${CHECKPOINT_DIR}/${DOMAIN}_lstm_best.pt"
            if [ -f "$BEST_PT" ]; then
                uv run python -m code.modeling.inference_lstm \
                    --domain "$DOMAIN" \
                    --checkpoint "$BEST_PT" \
                    --encoding "$ENCODING" \
                    --results_dir "$RESULTS_DIR" \
                    --hidden_dim "$HIDDEN_DIM" \
                    $DELTA_FLAG \
                    --tag "$MODE"
            else
                echo "Skipping Inference: Checkpoint not found ($BEST_PT)"
            fi
            ;;

        "xgboost")
            # XGBOOST TRAINING
            echo "[XGB] Training..."
            uv run python -m code.modeling.train_xgb \
                --domain "$DOMAIN" \
                --save_dir "$CHECKPOINT_DIR" \
                --data_dir "$DATA_DIR" \
                --n_estimators "$EPOCHS" \
                --max_depth "$MAX_DEPTH" \
                --lr "$LR" \
                $DELTA_FLAG

            # XGBOOST INFERENCE
            echo "[XGB] Inference..."
            # XGB saves .json models
            if [ -f "${CHECKPOINT_DIR}/${DOMAIN}_xgb.json" ]; then
                uv run python -m code.modeling.inference_xgb \
                    --domain "$DOMAIN" \
                    --checkpoint_dir "$CHECKPOINT_DIR" \
                    --results_dir "$RESULTS_DIR" \
                    $DELTA_FLAG \
                    --tag "$MODE"
            else
                echo "Skipping Inference: Model not found"
            fi
            ;;

        "llama")
            # Placeholder for future integration
            echo "[Llama] Not yet implemented in this script."
            ;;

        *)
            echo "Error: Unknown model '$MODEL'"
            ;;
        esac

    done
done

echo "Job finished at $(date)"
