# Data Processing Utilities

This directory the scripts that turn raw PDDL tasks into plans, trajectories (states), and supporting reports.

## Contents

- `generate_plans.py` – Batch solver that runs Fast Downward on every PDDL problem, first with an A* baseline and then with a greedy fallback, logging outcomes and producing `.plan` files, by default in the `data/plans` dir.
- `generate_states.py` – Converts generated plans into state trajectories by replaying them with VAL, extracting each timestep state, and saving per-problem `.traj` files, by default in the `data/states` dir., plus a CSV status report.
- `utils/pddl_utils.py` – Helper functions shared by the generators, including PDDL initial state extraction (`pddlpy` + regex fallback) and parsing verbose VAL logs into state sequences.
- `utils/lowercase_pddl.py` – Utility script that normalizes all `.pddl` files in a directory tree to lowercase so downstream parsers behave consistently. This was used especially since VAL was giving issues parsing non-lowercase input, such as `:INIT`.
- `logs/` – Output directory where plan and state generation scripts deposit their CSV reports (`plan_generation_report.csv`, `state_generation_report.csv`).

Directions: Use the plan generator first to populate `data/plans/`, then run the state generator to derive trajectories in `data/states/`. Both scripts accept CLI flags to point at custom input/output locations and control parallelism.
