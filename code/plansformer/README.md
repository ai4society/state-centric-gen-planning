# Plansformer Baseline Integration

This folder contains code to run Plansformer (CodeT5-based) as an action-centric baseline in the `state-centric-gen-planning` project, and evaluate it against Fast Downward ground-truth plans.

It handles:

1. Prompt construction from PDDL (`prompting.py`)
2. Plan generation with the pretrained Plansformer model (`generate_plans.py`)
3. Evaluation against ground-truth SAS+ plans (`evaluate_plans.py`)
4. Optional SLURM script to run generation + evaluation on a cluster

## Expected Data Layout

Plansformer uses the same PDDL layout as the rest of the project:

```text
data/
  pddl/
    <domain>/
      domain.pddl
      train/
      validation/
      test-interpolation/
      test-extrapolation/
        *.pddl

  plans/
    <domain>/
      train/
      validation/
      test-interpolation/
      test-extrapolation/
        *.plan   # Fast Downward ground-truth plans

  plansformer/
    <domain>/
      validation/
      test-interpolation/
      test-extrapolation/
        *.plan   # Plansformer-generated plans (created by this pipeline)
```

* Each PDDL problem file: `data/pddl/<domain>/<split>/<problem>.pddl`
* Ground-truth plan from Fast Downward: `data/plans/<domain>/<split>/<problem>.plan`
* Plansformer output: `data/plansformer/<domain>/<split>/<problem>.plan`
