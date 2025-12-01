import argparse
import json
import os
import subprocess
import tempfile
from code.common.utils import set_seed
from code.modeling.models import StateCentricLSTM, StateCentricLSTM_Delta

import numpy as np
import torch
import torch.nn.functional as F
from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from tqdm import tqdm
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import load_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem


def validate_plan(domain_path, problem_path, plan_actions, val_path):
    """
    Writes the plan to a temp file and runs VAL to verify correctness.
    Returns: (is_valid: bool, output_message: str)
    """
    if not plan_actions:
        return False, "Empty plan generated"

    # 1. Write plan to temporary file
    # VAL expects actions on separate lines: (action arg1 arg2)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".plan") as tmp:
        for action in plan_actions:
            # Ensure action is wrapped in parens if not already (pyperplan usually keeps them)
            act_str = str(action)
            if not act_str.startswith("("):
                act_str = f"({act_str})"
            tmp.write(f"{act_str}\n")
        tmp_plan_path = tmp.name

    # 2. Run VAL
    # Command: Validate -v domain.pddl problem.pddl plan.plan
    cmd = [val_path, str(domain_path), str(problem_path), tmp_plan_path]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = result.stdout

        # 3. Parse Output
        # VAL usually prints "Plan valid" or "Plan executed successfully"
        if "Plan valid" in output or "Plan executed successfully" in output:
            is_valid = True
            msg = "Plan valid"
        else:
            is_valid = False
            # Try to grab the error line
            lines = output.splitlines()
            errors = [l for l in lines if "Error" in l or "Failed" in l]
            msg = errors[-1] if errors else "VAL returned failure (unknown reason)"

    except Exception as e:
        is_valid = False
        msg = f"VAL execution error: {str(e)}"
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_plan_path):
            os.remove(tmp_plan_path)

    return is_valid, msg


def embed_state_to_tensor(atoms_set, feature_gen, wl_domain, wl_prob, pred_map, device):
    """
    Helper to convert a set of atoms (strings) into a Tensor [1, 1, D] and Numpy Array [D].
    """
    # 1. Convert Strings to WL Atoms
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

    # 2. Create State & Dataset
    state = State(wl_atoms)

    # Create mini dataset
    ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [state])])

    # 3. Embed
    # Returns list of vectors. We take the first one.
    embs = feature_gen.embed(ds)
    vec_raw = np.array(embs[0], dtype=np.float32)

    # 4. To Tensor (No Normalization!)
    tensor = (
        torch.tensor(vec_raw).float().to(device).unsqueeze(0).unsqueeze(0)
    )  # [1, 1, D]

    return tensor, vec_raw


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
    delta,
    beam_width=3,
):
    """
    Runs Latent Space Search using Beam Search with loop detection.
    """
    print(
        f"Inference using {'Delta Prediction' if delta else 'State Prediction'} for {prob_file}"
    )

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
    # Beam Element: (cumulative_score, hidden, last_tensor, current_atoms, plan, visited)
    # Score: Cumulative Euclidean Distance (Lower is better)

    initial_atoms = task.initial_state
    goal_set = set(task.goals)

    # Embed Initial State
    init_tensor, _ = embed_state_to_tensor(
        initial_atoms, feature_gen, wl_domain, wl_prob, pred_map, device
    )

    # Beam Element: (score, hidden, last_tensor, atoms, plan, visited_hashes)
    initial_hash = frozenset(initial_atoms)
    # Tuple size: 6
    beam = [(0.0, None, init_tensor, initial_atoms, [], {initial_hash})]

    for _ in range(max_steps):
        candidates = []

        # Unpack 6 items
        for score, hidden, last_tensor, current_atoms, plan, visited in beam:
            # Check Goal (Internal Check)
            if goal_set.issubset(current_atoms):
                return {
                    "problem": prob_file,
                    "search_solved": True,
                    "plan_len": len(plan),
                    "plan": plan,
                }

            # A. Predict Next Latent State/Delta
            # Input: last_tensor [1, 1, D], goal_tensor [1, D]
            # Output: pred_next_emb [1, 1, D], next_hidden (tuple)
            with torch.no_grad():
                # The model now predicts the State directly
                pred, next_hidden = model(last_tensor, goal_tensor, hidden=hidden)

                if delta:
                    # Reconstruct Next State: S_t + Delta
                    pred_next_emb = last_tensor + pred
                else:
                    # Model already predicts the next state directly
                    pred_next_emb = pred

            # B. Generate Successors
            successors = []
            for op in task.operators:
                if op.applicable(current_atoms):
                    next_atoms = op.apply(current_atoms)
                    successors.append((op.name, next_atoms))

            if not successors:
                continue

            # debugging
            # print(f"\nStep {len(plan)} | Current Best Score: {score:.4f}")

            # C. Score Successors
            for op_name, next_atoms in successors:
                # 1. Cycle Detection
                next_hash = frozenset(next_atoms)
                if next_hash in visited:
                    continue  # Skip cycles

                # 2. Embed Candidate
                cand_tensor, _ = embed_state_to_tensor(
                    next_atoms, feature_gen, wl_domain, wl_prob, pred_map, device
                )

                if not delta:
                    # Cosine similarity in [-1, 1], we want a cost where lower is better
                    cos = F.cosine_similarity(pred_next_emb, cand_tensor, dim=-1).item()
                    sim = 1.0 - cos  # cost in [0, 2]
                else:
                    # Euclidean Distance (L2)
                    # We want the candidate that is closest to our prediction
                    # Distance = ||Pred - Cand||
                    sim = torch.norm(pred_next_emb - cand_tensor, p=2).item()

                # Print the op name and the similarity score
                # If these are all 0.99+, the model is predicting Identity.
                # If the 'correct' action is lower than others, the physics are wrong.
                # print(f"  Op: {op_name:<30} | Sim: {sim:.5f}")

                # Update Score
                # Beam search usually minimizes cost.
                # Here, 'score' is cumulative error.
                new_score = score + sim

                # 5. Update Visited
                new_visited = visited.copy()
                new_visited.add(next_hash)

                # Append tuple of size 6
                candidates.append(
                    (
                        new_score,
                        next_hidden,
                        cand_tensor,
                        next_atoms,
                        plan + [op_name],
                        new_visited,
                    )
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
    best_attempt = beam[0] if beam else (0, None, None, None, [], set())

    return {
        "problem": prob_file,
        "search_solved": False,
        "plan_len": len(best_attempt[4]),
        "plan": best_attempt[4],
    }


def run_inference(args):
    set_seed(args.seed)

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
    if args.delta:
        model = StateCentricLSTM_Delta(input_dim, hidden_dim=args.hidden_dim).to(device)
    else:
        model = StateCentricLSTM(input_dim, hidden_dim=args.hidden_dim).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # optional tag so we don't overwrite baseline/delta results
    tag_suffix = f"_{args.tag}" if getattr(args, "tag", "") else ""

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
        prob_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pddl")])
        print(f" Found {len(prob_files)} problems for {split}")

        for prob_file in tqdm(prob_files, desc=f"Solving {split}"):
            prob_path = os.path.join(split_dir, prob_file)
            try:
                # A. Generate Plan
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
                    delta=args.delta,
                    beam_width=args.beam_width,
                )

                # B. Validate Plan with VAL (External Verification)
                # We validate even if search_solved is False, just in case the search
                # stopped exactly at the goal but logic missed it, or to confirm failure.
                # But typically we care if the plan generated so far is valid.

                is_valid, val_msg = validate_plan(
                    domain_path=domain_pddl,
                    problem_path=prob_path,
                    plan_actions=res["plan"],
                    val_path=args.val_path,
                )

                res["val_solved"] = is_valid
                res["val_reason"] = val_msg

                # The final "solved" metric should rely on VAL
                res["solved"] = is_valid

                results.append(res)
                if is_valid:
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
        tag_suffix = f"_{args.tag}" if getattr(args, "tag", "") else ""
        out_file = os.path.join(
            args.results_dir, f"{args.domain}_{split}{tag_suffix}_results.json"
        )
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
    parser.add_argument("--beam_width", type=int, default=2, help="Search beam width")
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Flag to whether perform delta-based preds. Def. is False",
    )
    parser.add_argument(
        "--tag",
        default="state",
        help="Optional tag to disambiguate results, e.g., 'state' or 'delta'",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--val_path", default="VAL/bin/Validate", help="Path to VAL binary"
    )

    args = parser.parse_args()

    run_inference(args)
