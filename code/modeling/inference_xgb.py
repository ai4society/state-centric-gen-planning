import argparse
import json
import os
from code.common.utils import set_seed, validate_plan

import numpy as np
import xgboost as xgb
from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from scipy.spatial.distance import cosine, euclidean
from torch import cuda
from tqdm import tqdm
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import load_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem


def embed_state_to_numpy(atoms_set, feature_gen, wl_domain, wl_prob, pred_map):
    """
    Helper to convert a set of atoms (strings) into a Numpy Array [1, D].
    """
    # 1. Convert Strings to WL Atoms
    wl_atoms = []
    for a_str in atoms_set:
        content = a_str.replace("(", "").replace(")", "")
        parts = content.split()
        if not parts:
            continue
        p_name = parts[0]
        p_args = parts[1:]
        if p_name in pred_map:
            wl_atoms.append(Atom(pred_map[p_name], p_args))

    # 2. Create State & Dataset
    state = State(wl_atoms)
    ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [state])])

    # 3. Embed
    embs = feature_gen.embed(ds)
    vec_raw = np.array(embs[0], dtype=np.float32)

    # Return [1, D]
    return vec_raw.reshape(1, -1)


def solve_problem(
    prob_file,
    domain_path,
    prob_path,
    feature_gen,
    model,
    max_steps,
    wl_domain,
    pred_map,
    delta,
    beam_width=3,
):
    """
    Runs Latent Space Search using Beam Search (XGBoost version).
    """
    # 1. Pyperplan Parsing
    try:
        parser = Parser(domain_path, prob_path)
        dom = parser.parse_domain()
        prob = parser.parse_problem(dom)
        task = ground(prob)
    except Exception as e:
        print(f"Pyperplan Parsing Error on {prob_file}: {e}")
        raise e

    # 2. WLPlan Context
    wl_prob = parse_problem(domain_path, prob_path)

    # 3. Embed Goal
    goal_atoms = list(wl_prob.positive_goals)
    goal_state = State(goal_atoms)
    goal_ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [goal_state])])
    goal_embs = feature_gen.embed(goal_ds)
    goal_vec = np.array(goal_embs[0], dtype=np.float32).reshape(1, -1)  # [1, D]

    # 4. Initialize Beam
    initial_atoms = task.initial_state
    goal_set = set(task.goals)

    # Embed Initial State
    init_vec = embed_state_to_numpy(
        initial_atoms, feature_gen, wl_domain, wl_prob, pred_map
    )

    # Beam Element: (score, current_vec, atoms, plan, visited_hashes)
    # Note: XGBoost is stateless (no hidden state), unlike LSTM.
    initial_hash = frozenset(initial_atoms)
    beam = [(0.0, init_vec, initial_atoms, [], {initial_hash})]

    for _ in range(max_steps):
        candidates = []

        for score, current_vec, current_atoms, plan, visited in beam:
            # Check Goal
            if goal_set.issubset(current_atoms):
                return {
                    "problem": prob_file,
                    "search_solved": True,
                    "plan_len": len(plan),
                    "plan": plan,
                }

            # A. Predict Next Latent State/Delta
            # Input: Concat [State, Goal] -> [1, 2D]
            model_input = np.hstack([current_vec, goal_vec])

            # Predict
            pred = model.predict(model_input)  # [1, D]

            if delta:
                pred_next_emb = current_vec + pred
            else:
                pred_next_emb = pred

            # B. Generate Successors
            successors = []
            for op in task.operators:
                if op.applicable(current_atoms):
                    next_atoms = op.apply(current_atoms)
                    successors.append((op.name, next_atoms))

            if not successors:
                continue

            # C. Score Successors
            for op_name, next_atoms in successors:
                next_hash = frozenset(next_atoms)
                if next_hash in visited:
                    continue

                cand_vec = embed_state_to_numpy(
                    next_atoms, feature_gen, wl_domain, wl_prob, pred_map
                )

                # Distance Calculation
                # Flatten for scipy
                u = pred_next_emb.flatten()
                v = cand_vec.flatten()

                if not delta:
                    # Cosine Distance (0 to 2)
                    # Handle zero vectors to avoid NaN
                    if np.all(u == 0) or np.all(v == 0):
                        dist = 1.0
                    else:
                        dist = cosine(u, v)
                else:
                    # Euclidean Distance
                    dist = euclidean(u, v)

                new_score = score + dist
                new_visited = visited.copy()
                new_visited.add(next_hash)

                candidates.append(
                    (new_score, cand_vec, next_atoms, plan + [op_name], new_visited)
                )

            # D. Prune Beam
            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_width]

    best_attempt = beam[0] if beam else (0, None, None, [], set())

    return {
        "problem": prob_file,
        "search_solved": False,
        "plan_len": len(best_attempt[3]),
        "plan": best_attempt[3],
    }


def run_inference(args):
    set_seed(args.seed)

    # 1. Load WL Generator
    model_path = os.path.join(
        args.data_dir, "encodings", "models", f"{args.domain}_wl.json"
    )
    if not os.path.exists(model_path):
        print(f"Error: WL Model not found at {model_path}")
        return

    print(f"Loading WL Model from {model_path}...")
    feature_gen = load_feature_generator(model_path)

    # 2. Load Domain
    domain_pddl = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
    wl_domain = parse_domain(domain_pddl)
    pred_map = {p.name: p for p in wl_domain.predicates}

    # 3. Load XGBoost
    xgb_path = os.path.join(args.checkpoint_dir, f"{args.domain}_xgb.json")
    print(f"Loading XGBoost from {xgb_path}...")

    # Check for GPU
    device = "cuda" if cuda.is_available() else "cpu"

    model = xgb.XGBRegressor(device=device)
    model.load_model(xgb_path)

    # 4. Run on Splits
    splits = ["validation", "test-interpolation", "test-extrapolation"]

    for split in splits:
        print(f"\n=== Testing on {split} ===")
        split_dir = os.path.join(args.pddl_dir, args.domain, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} (not found)")
            continue

        results = []
        solved_count = 0
        executable_count = 0
        prob_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pddl")])

        for prob_file in tqdm(prob_files, desc=f"Solving {split}"):
            prob_path = os.path.join(split_dir, prob_file)
            try:
                res = solve_problem(
                    prob_file=prob_file,
                    domain_path=domain_pddl,
                    prob_path=prob_path,
                    feature_gen=feature_gen,
                    model=model,
                    max_steps=args.max_steps,
                    wl_domain=wl_domain,
                    pred_map=pred_map,
                    delta=args.delta,
                    beam_width=args.beam_width,
                )

                # Validate
                is_solved, is_executable = validate_plan(
                    domain_path=domain_pddl,
                    problem_path=prob_path,
                    plan_actions=res["plan"],
                    val_path=args.val_path,
                )

                res["val_solved"] = is_solved
                res["val_executable"] = is_executable
                res["solved"] = is_solved

                results.append(res)
                if is_solved:
                    solved_count += 1
                if is_executable:
                    executable_count += 1

            except Exception as e:
                # import traceback
                # traceback.print_exc()
                results.append({"problem": prob_file, "solved": False, "error": str(e)})

        total = len(prob_files)
        acc = solved_count / total if total else 0
        print(f"Result {split}: Solved {solved_count}/{total} ({acc:.2%})")

        # Save
        os.makedirs(args.results_dir, exist_ok=True)
        tag_suffix = f"_{args.tag}" if getattr(args, "tag", "") else ""
        out_file = os.path.join(
            args.results_dir, f"{args.domain}_{split}{tag_suffix}_results.json"
        )
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--pddl_dir", default="data/pddl")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--beam_width", type=int, default=2)
    parser.add_argument("--delta", action="store_true")
    parser.add_argument("--tag", default="state")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--val_path",
        default=os.environ.get("VAL_PATH", "VAL/bin/Validate"),
    )

    args = parser.parse_args()
    run_inference(args)
