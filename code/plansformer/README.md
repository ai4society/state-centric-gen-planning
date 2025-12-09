# Plansformer Baseline

This directory contains the inference scripts for the **Plansformer** baseline (an action-centric Transformer).

## Usage

Plansformer takes raw PDDL text as input and outputs a sequence of actions. Unlike our State-Centric models, it does not track the world state explicitly.

### Running Inference

```bash
python -m code.plansformer.inference_plansformer \
  --data_path data/ \
  --save_path results/plansformer/ \
  --model_path <path_to_pretrained_weights> \
  --val_path <path_to_VAL_binary>
```

### Input Requirements

- The script expects the standard directory structure (`data/pddl/<domain>/<split>`).
- It automatically handles tokenization and PDDL parsing specific to the Plansformer architecture.
