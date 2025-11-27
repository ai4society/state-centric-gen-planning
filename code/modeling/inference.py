import argparse
import json
import os
from code.modeling.models import StateCentricLSTM

import numpy as np
import torch
from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from tqdm import tqdm
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import load_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem

# def normalize_np(vec):
#     """L1 Normalization for single vector"""
#     s = vec.sum()
#     return vec / s if s > 0 else vec


def solve_problem(
    prob_file,
    domain_path,
    prob_path,
    feature_gen,
    model,
    device,
    max_steps,
    wl_domain,
    pred_map,
):
    """
    Runs the Latent Space Search for a single problem.
    """
    # 1. Pyperplan Parsing (for successors)
    parser = Parser(domain_path, prob_path)
    dom = parser.parse_domain()
    prob = parser.parse_problem(dom)
    task = ground(prob)

    # 2. WLPlan Context (for embedding)
    wl_prob = parse_problem(domain_path, prob_path)

    # 3. Embed Goal
    goal_atoms = list(wl_prob.positive_goals)
    goal_state = State(goal_atoms)

    # Create mini dataset for goal
    goal_ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [goal_state])])

    # Use [0] to get the vector, not [0][0]
    goal_embs = feature_gen.embed(goal_ds)
    goal_vec_raw = np.array(goal_embs[0], dtype=np.float32)

    # NORMALIZE
    # goal_vec = normalize_np(goal_vec_raw)
    goal_vec = goal_vec_raw
    goal_tensor = torch.tensor(goal_vec).float().to(device).unsqueeze(0)  # [1, D]

    # 4. Search Loop
    current_atoms = task.initial_state  # Set of strings: {"(on a b)", ...}
    plan = []
    hidden = None
    solved = False
    goal_set = set(task.goals)

    for step in range(max_steps):
        # Check Goal
        if goal_set.issubset(current_atoms):
            solved = True
            break

        # Convert Pyperplan State (strings) -> WLPlan State (Atoms)
        wl_atoms = []
        for a_str in current_atoms:
            # Parse "(on a b)" -> name="on", args=["a", "b"]
            content = a_str.replace("(", "").replace(")", "")
            parts = content.split()
            if not parts:
                continue

            p_name = parts[0]
            p_args = parts[1:]

            if p_name in pred_map:
                wl_atoms.append(Atom(pred_map[p_name], p_args))

        curr_state = State(wl_atoms)

        # Embed Current State
        curr_ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [curr_state])])

        # Use [0] to get the vector
        curr_embs = feature_gen.embed(curr_ds)
        curr_vec_raw = np.array(curr_embs[0], dtype=np.float32)

        # Normalize
        # curr_vec = normalize_np(curr_vec_raw)
        curr_vec = curr_vec_raw
        curr_tensor = (
            torch.tensor(curr_vec).float().to(device).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, D]

        # Predict Next Latent State
        with torch.no_grad():
            pred_next_emb, hidden = model(curr_tensor, goal_tensor, hidden=hidden)
            target_vec = pred_next_emb.squeeze().cpu().numpy()

        # Generate Successors
        candidates = []
        for op in task.operators:
            if op.applicable(current_atoms):
                next_atoms = op.apply(current_atoms)
                candidates.append((op.name, next_atoms))

        if not candidates:
            break

        # Score Candidates
        best_action = None
        best_dist = float("inf")
        best_next_atoms = None

        for op_name, next_atoms in candidates:
            # Convert Candidate -> WL State
            cand_wl_atoms = []
            for a_str in next_atoms:
                content = a_str.replace("(", "").replace(")", "")
                parts = content.split()
                if parts and parts[0] in pred_map:
                    cand_wl_atoms.append(Atom(pred_map[parts[0]], parts[1:]))

            cand_state = State(cand_wl_atoms)
            cand_ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [cand_state])])

            # Use [0] to get the vector
            cand_embs = feature_gen.embed(cand_ds)
            cand_vec_raw = np.array(cand_embs[0], dtype=np.float32)

            # NORMALIZE
            # cand_vec = normalize_np(cand_vec_raw)
            cand_vec = cand_vec_raw

            # Distance
            dist = np.linalg.norm(cand_vec - target_vec)

            if dist < best_dist:
                best_dist = dist
                best_action = op_name
                best_next_atoms = next_atoms

        if best_action:
            plan.append(best_action)
            current_atoms = best_next_atoms
        else:
            break

    return {"problem": prob_file, "solved": solved, "plan_len": len(plan), "plan": plan}


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load WL Generator
    model_path = os.path.join(
        args.data_dir, "encodings", "models", f"{args.domain}_wl.json"
    )
    if not os.path.exists(model_path):
        print(f"Error: WL Model not found at {model_path}")
        return

    print(f"Loading WL Model from {model_path}...")
    feature_gen = load_feature_generator(model_path)
    input_dim = feature_gen.get_n_features()
    print(f"Feature Dimension: {input_dim}")

    # 2. Load Domain (for parsing)
    domain_pddl = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
    wl_domain = parse_domain(domain_pddl)
    pred_map = {p.name: p for p in wl_domain.predicates}

    # 3. Load LSTM
    print(f"Loading LSTM from {args.checkpoint}...")
    model = StateCentricLSTM(input_dim, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 4. Run on Splits
    splits = ["test-interpolation", "test-extrapolation"]

    for split in splits:
        print(f"\n=== Testing on {split} ===")
        split_dir = os.path.join(args.pddl_dir, args.domain, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} (not found)")
            continue

        prob_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pddl")])
        results = []
        solved_count = 0

        for prob_file in tqdm(prob_files, desc=f"Solving {split}"):
            prob_path = os.path.join(split_dir, prob_file)
            try:
                res = solve_problem(
                    prob_file=prob_file,
                    domain_path=domain_pddl,
                    prob_path=prob_path,
                    feature_gen=feature_gen,
                    model=model,
                    device=device,
                    max_steps=args.max_steps,
                    wl_domain=wl_domain,
                    pred_map=pred_map,
                )
                results.append(res)
                if res["solved"]:
                    solved_count += 1
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error: {e}")
                results.append({"problem": prob_file, "solved": False, "error": str(e)})

        # Report
        accuracy = solved_count / len(prob_files) if prob_files else 0
        print(f"Result {split}: {solved_count}/{len(prob_files)} ({accuracy:.2%})")

        # Save
        os.makedirs(args.results_dir, exist_ok=True)
        out_file = os.path.join(args.results_dir, f"{args.domain}_{split}_results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pddl_dir", default="data/pddl")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    run_inference(args)
