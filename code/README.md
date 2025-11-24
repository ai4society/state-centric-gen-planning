# Codebase Documentation

This directory contains the scripts required to generate the dataset found in `../data`. The pipeline transforms raw PDDL into machine-learning-ready graph embeddings.

## Directory Structure

```text
code/
├── data-processing/           # Step 1 & 2: Symbolic Processing
│   ├── generate_plans.py      # Runs Fast Downward
│   ├── generate_states.py     # Runs VAL to generate trajectories
│   └── utils/
│       ├── pddl_utils.py      # Parsing helpers (Regex/pddlpy)
│       └── lowercase_pddl.py  # Normalization utility
└── encoding-generation/       # Step 3: Vectorization
    └── generate_graph_embeddings.py  # Runs wlplan (Graph Kernels)
```

## Pipeline Execution

### Step 1: Plan Generation

Script: `code/data-processing/generate_plans.py`

- Input: `data/pddl/`
- Output: `data/plans/`
- Description: Runs the Fast Downward planner on every problem file. It attempts an optimal search (A\* `lmcut`) first. If that times out (60s), it falls back to a greedy search (GBFS `ff`) with a higher timeout (300s).
- Command from the root directory (alternatively see the `data_gen.sh` file at the root):
  ```bash
  uv run python -m code.data-processing.generate_plans --workers 120
  ```

### Step 2: State Trajectory Generation

Script: `code/data-processing/generate_states.py`

- Input: `data/plans/` and `data/pddl/`
- Output: `data/states/`
- Description: Uses the `VAL` binary to validate the generated plans. It parses the verbose output of VAL to reconstruct the exact state (set of true predicates) at every time step of the plan.
- Command from the root directory (alternatively see the `data_gen.sh` file at the root):
  ```bash
  uv run python -m code.data-processing.generate_states --workers 120
  ```

### Step 3: Graph Embedding Generation

Script: `code/encoding-generation/generate_graph_embeddings.py`

- Input: `data/states/` (Trajectories) and `data/pddl/` (Goals)
- Output: `data/encodings/graphs/`
- Description: Converts PDDL states and goals into **Instance Learning Graphs (ILG)** and applies the **Weisfeiler-Leman (WL)** algorithm to extract fixed-size feature vectors.
- Logic:
  1.  Phase 1 (Collection): Iterates over the `train` split (both trajectories and goals). Builds a global hash map (vocabulary) of graph substructures.
  2.  Phase 2 (Embedding): Freezes the vocabulary and processes `train`, `validation`, and `test` splits. Converts graphs into NumPy arrays using the frozen hash map.
- Command from the root directory (alternatively see the `embeddings.sh` file at the root):
  ```bash
  # Note: This script runs sequentially to ensure hash map stability in C++ bindings
  uv run python -m code.encoding-generation.generate_graph_embeddings --iterations 2
  ```

## Dependencies

- Python Packages:
  - `pddlpy`: For parsing PDDL structures.
  - `wlplan`: For graph generation and WL kernel hashing.
  - `numpy`: For vector storage.
  - `tqdm`: For progress bars.
- External Binaries:
  - `fast-downward`: Required for Step 1.
  - `VAL` (Validate): Required for Step 2.

## Key Implementation Details

- Lowercase Normalization: PDDL is case-insensitive, but Python strings are not. The pipeline normalizes all predicates to lowercase to ensure `(ON A B)` matches `(on a b)`. _See the `code/data-processing/utils/lowercase_pddl.py` file and it run in the `data_gen.sh` file at the root._
- Closed World Assumption: The `.traj` files only list _true_ predicates. Any predicate defined in the domain but missing from a line in the `.traj` file is implicitly false.
- Goal Parsing: The embedding script explicitly parses the `:goal` section of the PDDL files to create `_goal.npy` files. This allows downstream models to be conditioned on the target state.
