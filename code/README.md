# Codebase Documentation

## 📦 Modules

### 1. Data Processing (`code/data-processing/`)

Handles the symbolic aspects of planning.

- `generate_plans.py`: Wraps Fast Downward to solve PDDL problems.
- `generate_states.py`: Wraps VAL to execute plans and extract state traces.
- `utils/`: PDDL parsing and normalization utilities.

### 2. Encoding (`code/encoding-generation/`)

Converts symbolic states to vectors.

- `generate_graph_embeddings.py`: Uses `wlplan` to generate graph hashes.
- `generate_fsf_embeddings.py`: Generates fixed-size vectors based on object indices.

### 3. Modeling (`code/modeling/`)

Contains PyTorch and XGBoost implementations.

| Script              | Description                                                             |
| :------------------ | :---------------------------------------------------------------------- |
| `models.py`         | PyTorch architectures (StateCentricLSTM, Delta variants).               |
| `train_lstm.py`     | Trains LSTM with Cosine Embedding Loss (State) or MSE (Delta).          |
| `inference_lstm.py` | Performs **Latent Beam Search** using the trained LSTM.                 |
| `train_xgb.py`      | Flattens trajectories into $(S_t, Goal) \to S_{t+1}$ pairs for XGBoost. |
| `inference_xgb.py`  | Latent Beam Search adapted for non-sequential XGBoost models.           |

## ⚙️ Common Arguments

All training and inference scripts share these core flags:

- `--domain`: The PDDL domain name (e.g., `blocks`, `logistics`).
- `--delta`: **Crucial.** If set, the model learns physics residuals ($S_{t+1} - S_t$). If unset, it learns to reconstruct the raw next state.
- `--encoding`: Choose between `graphs` (WL) or `fsf` (Fixed-Size).
- `--beam_width`: (Inference only) Width of the latent beam search (Default: 3).

## 🧪 Reproducing Inference

To run inference manually without SLURM:

```bash
# LSTM Inference
python -m code.modeling.inference_lstm \
  --domain blocks \
  --checkpoint checkpoints/graphs/lstm_delta/blocks_lstm_best.pt \
  --encoding graphs \
  --results_dir results/manual_test \
  --delta

# XGBoost Inference
python -m code.modeling.inference_xgb \
  --domain blocks \
  --checkpoint_dir checkpoints/graphs/xgboost_delta \
  --results_dir results/manual_test \
  --delta
```
