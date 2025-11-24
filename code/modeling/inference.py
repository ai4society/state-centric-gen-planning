import argparse
import json
import os
from code.common.wl_wrapper import WLEncoder
from code.modeling.models import StateCentricLSTM

import numpy as np
import pddlpy
import torch
from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from tqdm import tqdm


def solve_problem(prob_file, domain_path, prob_path, encoder, model, device, max_steps):
    """
    Runs the Latent Space Search for a single problem.
    """
    # 1. Parse Initial State
    domprob = pddlpy.DomainProblem(domain_path, prob_path)
    current_atoms = set()
    for atom in domprob.initialstate():
        if hasattr(atom, "predicate"):
            s = f"({atom.predicate[0]} {' '.join(atom.predicate[1:])})"
        else:
            s = f"({atom[0]} {' '.join(atom[1:])})"
        current_atoms.add(s)

    # 2. Embed Goal
    goal_wl = encoder.parse_pddl_goal_to_wl_state(prob_path)
    goal_emb = (
        torch.tensor(encoder.embed_state(goal_wl, prob_path))
        .float()
        .to(device)
        .unsqueeze(0)
    )

    # 3. Setup Pyperplan Task (for successor generation)
    parser = Parser(domain_path, prob_path)
    task = ground(parser.parse_problem(parser.parse_domain()))

    # 4. Search Loop
    plan = []
    hidden = None
    solved = False

    for step in range(max_steps):
        # Check Goal
        if task.goal.issubset(current_atoms):
            solved = True
            break

        # Embed Current State
        curr_wl = encoder.parse_state_string_to_wl_state(list(current_atoms))
        curr_emb = (
            torch.tensor(encoder.embed_state(curr_wl, prob_path)).float().to(device)
        )
        curr_emb_in = curr_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, D]

        # Predict Next Latent State
        with torch.no_grad():
            pred_next_emb, hidden = model(curr_emb_in, goal_emb, hidden=hidden)
            target_vec = pred_next_emb.squeeze().cpu().numpy()

        # Generate Successors
        candidates = []
        for op in task.operators:
            if op.applicable(current_atoms):
                next_atoms = op.apply(current_atoms)
                candidates.append((op.name, next_atoms))

        if not candidates:
            break  # Dead end

        # Score Candidates
        best_action = None
        best_dist = float("inf")
        best_next_atoms = None

        for op_name, next_atoms in candidates:
            cand_wl = encoder.parse_state_string_to_wl_state(list(next_atoms))
            cand_emb = encoder.embed_state(cand_wl, prob_path)

            dist = np.linalg.norm(cand_emb - target_vec)

            if dist < best_dist:
                best_dist = dist
                best_action = op_name
                best_next_atoms = next_atoms

        # Step
        if best_action:
            plan.append(best_action)
            current_atoms = best_next_atoms
        else:
            break

    return {"problem": prob_file, "solved": solved, "plan_len": len(plan), "plan": plan}


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup WL Encoder (this must match training)
    print("Initializing WL Encoder (Re-collecting training vocab)...")
    domain_pddl_path = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
    encoder = WLEncoder(domain_pddl_path)
    encoder.collect_vocabulary(os.path.join(args.states_dir, args.domain, "train"))

    # 2. Load Model
    # Hack to get input dim: embed a dummy goal
    dummy_prob_path = os.path.join(
        args.pddl_dir,
        args.domain,
        "train",
        os.listdir(os.path.join(args.pddl_dir, args.domain, "train"))[0],
    )
    if not dummy_prob_path.endswith(".pddl"):
        dummy_prob_path = dummy_prob_path.replace(".traj", ".pddl")  # safety

    dummy_goal = encoder.parse_pddl_goal_to_wl_state(dummy_prob_path)
    dummy_emb = encoder.embed_state(dummy_goal, dummy_prob_path)
    input_dim = dummy_emb.shape[0]

    print(f"Loading model from {args.checkpoint} (Dim: {input_dim})...")
    model = StateCentricLSTM(input_dim, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 3. Run on Splits
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
                    prob_file,
                    domain_pddl_path,
                    prob_path,
                    encoder,
                    model,
                    device,
                    args.max_steps,
                )
                results.append(res)
                if res["solved"]:
                    solved_count += 1
            except Exception as e:
                print(f"Error solving {prob_file}: {e}")
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
    parser.add_argument("--pddl_dir", default="data/pddl")
    parser.add_argument("--states_dir", default="data/states")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    run_inference(args)
