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


def embed_state_to_tensor(atoms_set, feature_gen, wl_domain, wl_prob, pred_map, device):
    """
    Helper to convert a set of Pyperplan strings -> WLPlan State -> Tensor [1, 1, D]
    """
    wl_atoms = []
    for a_str in atoms_set:
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

    # Create mini dataset
    curr_ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [curr_state])])

    # Embed
    # Use [0] to get the vector
    curr_embs = feature_gen.embed(curr_ds)
    curr_vec = np.array(curr_embs[0], dtype=np.float32)

    # Shape: [1, 1, D]
    curr_tensor = torch.tensor(curr_vec).float().to(device).unsqueeze(0).unsqueeze(0)

    return curr_tensor, curr_vec


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
    beam_width=3,
):
    """
    Runs Latent Space Search using Beam Search.
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
    goal_ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [goal_state])])
    goal_embs = feature_gen.embed(goal_ds)
    goal_vec = np.array(goal_embs[0], dtype=np.float32)

    goal_tensor = torch.tensor(goal_vec).float().to(device).unsqueeze(0)  # [1, D]

    # 4. Initialize Beam
    # Beam Element: (cumulative_score, hidden_state, last_input_tensor, current_atoms, plan)
    # Score: Cumulative Euclidean distance (Lower is better)

    initial_atoms = task.initial_state
    goal_set = set(task.goals)

    # Embed Initial State
    init_tensor, _ = embed_state_to_tensor(
        initial_atoms, feature_gen, wl_domain, wl_prob, pred_map, device
    )

    # (score, hidden, last_tensor, atoms, plan)
    beam = [(0.0, None, init_tensor, initial_atoms, [])]

    for _ in range(max_steps):
        candidates = []

        # Expand every node in the current beam
        for score, hidden, last_tensor, current_atoms, plan in beam:
            # Check Goal
            if goal_set.issubset(current_atoms):
                return {
                    "problem": prob_file,
                    "solved": True,
                    "plan_len": len(plan),
                    "plan": plan,
                }

            # A. Predict Next Latent State
            # Input: last_tensor [1, 1, D], goal_tensor [1, D]
            # Output: pred_next_emb [1, 1, D], next_hidden (tuple)
            with torch.no_grad():
                pred_next_emb, next_hidden = model(
                    last_tensor, goal_tensor, hidden=hidden
                )
                target_vec = pred_next_emb.squeeze().cpu().numpy()

            # B. Generate Successors
            successors = []
            for op in task.operators:
                if op.applicable(current_atoms):
                    next_atoms = op.apply(current_atoms)
                    successors.append((op.name, next_atoms))

            if not successors:
                continue  # Dead end for this path

            # C. Score Successors
            for op_name, next_atoms in successors:
                # Embed Candidate
                cand_tensor, cand_vec = embed_state_to_tensor(
                    next_atoms, feature_gen, wl_domain, wl_prob, pred_map, device
                )

                # Calculate Distance (Prediction Error)
                dist = np.linalg.norm(cand_vec - target_vec)

                # New Score = Previous Score + Current Error
                new_score = score + dist

                # Add to candidates list
                candidates.append(
                    (new_score, next_hidden, cand_tensor, next_atoms, plan + [op_name])
                )

        # D. Prune Beam
        if not candidates:
            break  # All paths led to dead ends

        # Sort by score (ascending) and take top K
        # Note: We rely on Python's stable sort.
        # Tensors/Sets are not comparable, so we might need a wrapper if scores are identical.
        # But floats rarely collide exactly.
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    # If we exit loop, we failed to find goal within max_steps
    # Return the best path found so far (lowest error)
    best_attempt = beam[0] if beam else (0, None, None, None, [])

    return {
        "problem": prob_file,
        "solved": False,
        "plan_len": max_steps,
        "plan": best_attempt[4],
    }


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
                    beam_width=args.beam_width,
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
    parser.add_argument(
        "--beam_width", type=int, default=5, help="Beam width for search"
    )
    args = parser.parse_args()

    run_inference(args)
