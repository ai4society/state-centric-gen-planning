# Codebase Documentation

This directory contains the scripts required to generate data, train models, and run inference.

## Directory Structure

```text
code/
├── data-processing/           # Step 1 & 2: Symbolic Processing (PDDL/FastDownward/VAL)
├── encoding-generation/       # Step 3: Vectorization (WL Graph Kernels)
├── modeling/                  # Step 4 & 5: Machine Learning
│   ├── dataset.py             # PyTorch Datasets & XGBoost Flattening logic
│   ├── models.py              # PyTorch Architectures (LSTM)
│   ├── train_lstm.py          # LSTM Training Script
│   ├── inference_lstm.py      # LSTM Latent Beam Search
│   ├── train_xgb.py           # XGBoost Training Script
│   └── inference_xgb.py       # XGBoost Latent Beam Search
└── common/                    # Shared Utilities (Seeding, VAL wrapper)
```

## Machine Learning Pipeline (`code/modeling/`)

We support multiple architectures. All scripts accept a `--delta` flag to toggle between predicting raw states or state differences.

### 1. LSTM (Recurrent)

- **Script**: `train_lstm.py` / `inference_lstm.py`
- **Logic**: Uses a standard LSTM with a projection layer.
- **Input**: Sequence $[S_0, S_1, \dots, S_t]$
- **Output**: $S_{t+1}$ (State mode) or $S_{t+1} - S_t$ (Delta mode).

### 2. XGBoost (Gradient Boosting)

- **Script**: `train_xgb.py` / `inference_xgb.py`
- **Logic**: Flattens the trajectory into independent transition pairs.
- **Input**: Concatenation of $[S_t, Goal]$.
- **Output**: $S_{t+1}$ (State mode) or $S_{t+1} - S_t$ (Delta mode).
- **Note**: XGBoost uses the `multi:squarederror` objective to handle the multi-dimensional output vector (size $D \approx 500+$).

### 3. Unified Execution

The `2_unified_train_eval.slurm` script in the root directory acts as a dispatcher. It automatically organizes outputs into the following hierarchy:

```text
checkpoints/
└── <encoding>/          # e.g., "graphs"
    ├── lstm_state/      # Model + Mode
    ├── lstm_delta/
    ├── xgboost_state/
    └── xgboost_delta/
```

## Data Processing Pipeline (`code/data-processing/`)

1.  **Plan Generation**: `generate_plans.py` runs Fast Downward.
2.  **State Generation**: `generate_states.py` runs VAL to create `.traj` files.
3.  **Graph Embedding**: `generate_graph_embeddings.py` runs Weisfeiler-Leman to create `.npy` vectors.

## Arguments & Flags

Common arguments for training scripts:

- `--domain`: The PDDL domain name (e.g., `blocks`).
- `--delta`: **Crucial**. If set, the model learns physics residuals ($S_{t+1} - S_t$). If unset, it learns to reconstruct the entire state.
- `--save_dir`: Where to save checkpoints.

Common arguments for inference scripts:

- `--beam_width`: Width of the latent beam search (default: 2 or 3).
- `--max_steps`: Maximum plan length before aborting.
