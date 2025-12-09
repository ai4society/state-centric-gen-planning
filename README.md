# State-Centric Generalized Planning

This repository contains the official implementation for the paper **"On Sample-Efficient Generalized Planning via Learned Transition Models"**.

We propose a shift from action-centric planning (predicting actions) to **state-centric planning** (predicting future states). By learning the physics of the domain (state transitions) rather than policy, we demonstrate superior Out-of-Distribution (OOD) generalization on classical planning domains.

## 📂 Project Structure

```text
.
├── code/                  # Source code for data gen, training, and inference
├── data/                  # PDDL files, generated plans, and ML encodings
├── checkpoints/           # Saved model weights (LSTM .pt, XGBoost .json)
├── results/               # Inference logs and JSON metrics
├── 1_data_pipeline.slurm  # SLURM script for full data generation
└── 2_train_eval.slurm     # SLURM script for training and evaluation
```

## 🚀 Quick Start

### Prerequisites

We use [`uv`](https://docs.astral.sh/uv/) for fast Python dependency management.

```bash
pip install uv
uv sync
```

### 1. Data Generation

The pipeline converts PDDL $\to$ Plans $\to$ State Trajectories $\to$ Graph Embeddings.

```bash
# Runs the full pipeline (Fast Downward -> VAL -> WL Hashing)
sbatch 1_data_pipeline.slurm
```

### 2. Training & Evaluation

We provide a unified dispatcher to train models and immediately run inference on OOD test sets.

**Syntax:** `sbatch 2_train_eval.slurm "<models>" "<encoding>"`

```bash
# Experiment A: LSTM with Weisfeiler-Leman (Graph) Encodings
sbatch 2_train_eval.slurm "lstm" "graphs"

# Experiment B: XGBoost with Fixed-Size Factored (FSF) Encodings
sbatch 2_train_eval.slurm "xgboost" "fsf"

# Experiment C: Run both models on Graphs
sbatch 2_train_eval.slurm "lstm xgboost" "graphs"
```

## 🧠 Key Concepts

### State vs. Delta Prediction

We evaluate two prediction modes (controlled via `--delta`):

1.  **State Prediction:** $f(S_t, Goal) \to S_{t+1}$. The model reconstructs the entire next state.
2.  **Delta Prediction:** $f(S_t, Goal) \to \Delta$. The model predicts the _change_ ($S_{t+1} - S_t$). This is crucial for non-deep baselines (XGBoost) to learn inertia.

### Latent Beam Search

Unlike standard planners that search in the symbolic space, we search in the **latent embedding space**.

1.  Current state $S_t$ is embedded into vector $z_t$.
2.  Model predicts $\hat{z}_{t+1}$.
3.  We generate symbolic successors of $S_t$, embed them, and select the one closest to $\hat{z}_{t+1}$ (via Cosine or Euclidean distance).

## 📊 Baselines

- **Fast Downward (FD):** Classical symbolic planner (A\*).
- **Plansformer:** Transformer-based action sequence generator.
- **XGBoost:** Non-sequential gradient boosting (tests if memory is required).
- **LSTM:** Recurrent neural network (our primary efficient architecture).
