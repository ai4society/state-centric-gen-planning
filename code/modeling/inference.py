import argparse
import json
import os
from code.common.wl_wrapper import WLEncoder
from code.modeling.models import StateCentricLSTM

import numpy as np
import torch
from pyperplan.grounding import ground

# Pyperplan imports
from pyperplan.pddl.parser import Parser
from tqdm import tqdm


def solve_problem(prob_file, domain_path, prob_path, encoder, model, device, max_steps):
    """
    Runs the Latent Space Search for a single problem.
    """
    # 1. Setup Pyperplan Task
    # Parse and ground the problem to get operators and initial state
    parser = Parser(domain_path, prob_path)
    dom = parser.parse_domain()
    prob = parser.parse_problem(dom)
    task = ground(prob)

    # 2. Initialize Current State
    # task.initial_state is a set of strings (e.g., "(on a b)")
    current_atoms = task.initial_state

    # 3. Embed Goal
    # We use task.goals (list of strings) to get the goal embedding.
    # The encoder needs the prob_path to build the goal-aware graph structure.
    goal_atoms_list = task.goals
    goal_vec = encoder.embed_state(goal_atoms_list, prob_path)
    goal_emb = torch.tensor(goal_vec).float().to(device).unsqueeze(0)  # [1, D]

    # 4. Search Loop
    plan = []
    hidden = None
    solved = False

    # Convert goals to set for O(1) subset checking
    goal_set = set(task.goals)

    for step in range(max_steps):
        # Check Goal
        if goal_set.issubset(current_atoms):
            solved = True
            break

        # Embed Current State
        # Convert set to list for the encoder
        curr_vec = encoder.embed_state(list(current_atoms), prob_path)
        curr_emb = torch.tensor(curr_vec).float().to(device)
        curr_emb_in = curr_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, D]

        # Predict Next Latent State
        with torch.no_grad():
            pred_next_emb, hidden = model(curr_emb_in, goal_emb, hidden=hidden)
            target_vec = pred_next_emb.squeeze().cpu().numpy()

        # Generate Successors
        candidates = []
        for op in task.operators:
            if op.applicable(current_atoms):
                # op.apply returns a new set of atoms
                next_atoms = op.apply(current_atoms)
                candidates.append((op.name, next_atoms))

        if not candidates:
            break  # Dead end

        # Score Candidates
        best_action = None
        best_dist = float("inf")
        best_next_atoms = None

        for op_name, next_atoms in candidates:
            # Embed candidate state
            cand_vec = encoder.embed_state(list(next_atoms), prob_path)

            # Calculate distance to predicted state
            dist = np.linalg.norm(cand_vec - target_vec)

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
    print(f"Using device: {device}")

    # 1. Setup WL Encoder (this must match training)
    print("Initializing WL Encoder (Re-collecting training vocab)...")
    domain_pddl_path = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
    encoder = WLEncoder(domain_pddl_path, iterations=args.iterations)
    encoder.collect_vocabulary(os.path.join(args.states_dir, args.domain, "train"))

    # 2. Load Model
    # Hack to get input dim: embed a dummy goal
    dummy_prob_path = os.path.join(
        args.pddl_dir,
        args.domain,
        "train",
        os.listdir(os.path.join(args.pddl_dir, args.domain, "train"))[0],
    )
    # Ensure we picked a pddl file
    if not dummy_prob_path.endswith(".pddl"):
        dummy_prob_path = dummy_prob_path.replace(".traj", ".pddl")

    # Embed dummy goal to get dimension
    dummy_vec = encoder.embed_state([], dummy_prob_path)  # Empty state is valid
    input_dim = dummy_vec.shape[0]

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
                # import traceback
                # traceback.print_exc()
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
    parser.add_argument(
        "--iterations", type=int, default=2, help="Must match generation"
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    run_inference(args)
