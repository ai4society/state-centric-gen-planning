# State-Centric Generalized Planning

This repository contains the implementation for **"On Efficient Generalized Planning with State-Centric Learning"**. We hypothesize that predicting the sequence of future states (state-centric) is superior to predicting the sequence of actions (action-centric) for generalized planning.

## Project Structure

- **`code/`**: Python scripts for data generation, training, and inference.
- **`data/`**: Stores PDDL files, generated plans, state trajectories, and ML encodings.
- **`checkpoints/`**: Saved models, organized by encoding and model type.
- **`results/`**: Inference logs and JSON metrics.

## Supported Models

1.  **LSTM (Recurrent)**: A sequence-to-sequence model maintaining a hidden state.
2.  **XGBoost (Non-Sequential)**: A gradient-boosted tree regressor treating $(S_t, Goal) \to S_{t+1}$ as a tabular problem.
3.  **Llama (Transformer)**: _[Coming Soon]_ Fine-tuned causal language model.

## Quick Start

### 1. Environment Setup

- Ensure you have [`uv`](https://docs.astral.sh/uv/) installed for dependency management.
  _If not prefered, you can use `pip` or `conda` too._

- This project primarily ran on HPC clusters with SLURM. Those scripts are in the root directory and start with a number prefix (e.g., `1_data_pipeline.slurm`). _Please adjust those scripts for local runs as needed._

  - Data Pipeline: Run the full data generation pipeline (PDDL $\to$ Plans $\to$ States $\to$ Graph Embeddings).

    ```bash
    sbatch 1_data_pipeline.slurm
    ```

  - Training & Evaluation: We use a unified dispatcher to run experiments. This script handles training and immediate inference for specified models and domains.

    ```bash
    # Run ALL models on ALL domains
    sbatch 2_unified_train_eval.slurm

    # Run ONLY XGBoost
    sbatch 2_unified_train_eval.slurm "xgboost"

    # Run ONLY LSTM
    sbatch 2_unified_train_eval.slurm "lstm"
    ```

## Key Hypotheses

- **State vs. Delta**: We compare predicting the raw next state $S_{t+1}$ vs. the difference $\Delta = S_{t+1} - S_t$.
  - _Hypothesis:_ Delta prediction is crucial for non-deep models like XGBoost to learn "inertia" (most things don't change).
- **OOD Generalization**: Models are trained on small instances (2-5 objects) and tested on large instances (10+ objects).
