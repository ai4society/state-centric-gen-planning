# Codebase Documentation

## 📦 Modules

### 1. Data Processing (`code/data-processing/`)

Handles the symbolic aspects of planning.

- `generate_plans.py`: Wraps Fast Downward to solve PDDL problems.
- `generate_states.py`: Wraps VAL to execute plans and extract state traces.

### 2. Encoding (`code/encoding-generation/`)

Converts symbolic states to vectors.

- `generate_graph_embeddings.py`: Uses `wlplan` to generate graph hashes (WL).
- `generate_fsf_embeddings.py`: Generates fixed-size factored vectors (FSF).

### 3. Modeling (`code/modeling/`)

Contains PyTorch and XGBoost implementations.

- `train_lstm.py` / `inference_lstm.py`: Recurrent transition models.
- `train_xgb.py` / `inference_xgb.py`: Non-parametric transition models.

### 4. Analysis (`code/analysis/`)

- `aggregate_results.py`: Scans the `results/` directory and produces a summary table of coverage rates across domains and splits.

## ⚙️ Key Arguments

- `--delta`: **Crucial.** If set, the model learns physics residuals ($S_{t+1} - S_t$). If unset, it learns to reconstruct the raw next state.
- `--encoding`: Choose between `graphs` (WL) or `fsf` (Fixed-Size).
- `--beam_width`: (Inference only) Width of the latent beam search (Default: 3).
