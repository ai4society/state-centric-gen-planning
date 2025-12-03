import argparse
import json
import os
from code.common.fsf_wrapper import FSFEncoder
from code.common.utils import set_seed, validate_plan
from code.modeling.models import StateCentricLSTM, StateCentricLSTM_Delta

import numpy as np
import torch
import torch.nn.functional as F
from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from tqdm import tqdm

# Import Wrappers
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import load_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem


def get_fsf_tensor(atoms_set, encoder, objects, obj_map, device):
    """Helper for FSF Inference embedding"""
    # Convert set of strings to list of tuples
    atom_tuples = []
    for a in atoms_set:
        content = a.replace("(", "").replace(")", "").lower()
        atom_tuples.append(tuple(content.split()))

    vec = encoder._state_to_vector(atom_tuples, objects, obj_map)
    # [1, 1, D]
    return torch.tensor(vec).float().to(device).unsqueeze(0).unsqueeze(0)


def solve_problem(args, prob_file, model, device, encoder_type, feature_gen_or_encoder):
    """Unified Solver for WL and FSF"""
    print(
        f"Inference using {'Delta Prediction' if args.delta else 'State Prediction'} for {prob_file}"
    )

    domain_path = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
    prob_path = os.path.join(args.pddl_dir, args.domain, args.split, prob_file)

    # 1. Pyperplan Parsing (for successors)
    try:
        parser = Parser(domain_path, prob_path)
        dom = parser.parse_domain()
        prob = parser.parse_problem(dom)
        task = ground(prob)
    except Exception as e:
        print(f"Pyperplan Parsing Error on {prob_file}: {e}")
        raise e

    initial_atoms = task.initial_state
    goal_set = set(task.goals)

    # 2. Embedding Setup
    if encoder_type == "fsf":
        encoder = feature_gen_or_encoder
        objects = encoder._get_sorted_objects(prob_path)
        obj_map = encoder._get_object_indices(objects)

        # Embed Goal
        goal_vec = encoder.embed_goal(prob_path)
        goal_tensor = torch.tensor(goal_vec).float().to(device).unsqueeze(0)  # [1, D]

        # Embed Init
        init_tensor = get_fsf_tensor(initial_atoms, encoder, objects, obj_map, device)

    else:
        # WL Logic
        feature_gen = feature_gen_or_encoder
        wl_domain = parse_domain(domain_path)
        wl_prob = parse_problem(domain_path, prob_path)
        pred_map = {p.name: p for p in wl_domain.predicates}

        # Embed Goal
        goal_atoms = list(wl_prob.positive_goals)
        goal_state = State(goal_atoms)
        goal_ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [goal_state])])
        goal_embs = feature_gen.embed(goal_ds)
        goal_vec = np.array(goal_embs[0], dtype=np.float32)
        goal_tensor = torch.tensor(goal_vec).float().to(device).unsqueeze(0)

        # Helper for WL
        def get_wl_tensor(atoms):
            wl_atoms = []
            for a_str in atoms:
                parts = a_str.replace("(", "").replace(")", "").split()
                if parts and parts[0] in pred_map:
                    wl_atoms.append(Atom(pred_map[parts[0]], parts[1:]))
            ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [State(wl_atoms)])])
            vec = np.array(feature_gen.embed(ds)[0], dtype=np.float32)
            return torch.tensor(vec).float().to(device).unsqueeze(0).unsqueeze(0)

        init_tensor = get_wl_tensor(initial_atoms)

    # 3. Beam Search
    beam = [
        (0.0, None, init_tensor, initial_atoms, [], set())
    ]  # score, hidden, tensor, atoms, plan, visited

    for _ in range(args.max_steps):
        candidates = []
        for score, hidden, last_tensor, current_atoms, plan, visited in beam:
            # Check Goal (Internal Check)
            if goal_set.issubset(current_atoms):
                return {
                    "problem": prob_file,
                    "search_solved": True,
                    "plan_len": len(plan),
                    "plan": plan,
                }

            # Predict Next Latent State/Delta
            with torch.no_grad():
                # The model predicts the State directly
                pred, next_hidden = model(last_tensor, goal_tensor, hidden=hidden)

                # reconstruct the next state (S_t + Delta) if delta
                # else, the model already predicts S_t+1 directly
                pred_next_emb = (last_tensor + pred) if args.delta else pred

            # Successors
            successors = []
            for op in task.operators:
                if op.applicable(current_atoms):
                    successors.append((op.name, op.apply(current_atoms)))

            # print(f"\nStep {len(plan)} | Current Best Score: {score:.4f}")  # debugging

            # Sort successors by name to ensure processing order is deterministic
            successors.sort(key=lambda x: x[0])

            # Score Successors
            for op_name, next_atoms in successors:
                # Cycle Detection
                next_hash = frozenset(next_atoms)
                if next_hash in visited:
                    continue  # Skip cycles

                # Embed Candidate
                if encoder_type == "fsf":
                    cand_tensor = get_fsf_tensor(
                        next_atoms, encoder, objects, obj_map, device
                    )
                else:
                    cand_tensor = get_wl_tensor(next_atoms)

                # Distance
                if args.delta:
                    # Euclidean Distance (L2)
                    # We want the candidate that is closest to our prediction
                    # Distance = ||Pred - Cand||
                    sim = torch.norm(pred_next_emb - cand_tensor, p=2).item()
                else:
                    # Cosine similarity in [-1, 1], we want a cost where lower is better
                    cos = F.cosine_similarity(pred_next_emb, cand_tensor, dim=-1).item()
                    sim = 1.0 - cos  # cost in [0, 2]

                # DEBUG: Print the op name and the similarity score
                # If these are all 0.99+, the model is predicting Identity.
                # If the 'correct' action is lower than others, the physics are wrong.
                # print(f"  Op: {op_name:<30} | Sim: {sim:.5f}")

                # Update Score
                # Beam search usually minimizes cost.
                # Here, 'score' is cumulative error.
                new_score = score + sim

                # Update Visited
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

            # Prune Beam
            if not candidates:
                break  # All paths led to dead ends

            # Stable Sort:
            # Primary Key: Score (float)
            # Secondary Key: String representation of the plan (deterministic tie-breaker)
            candidates.sort(key=lambda x: (x[0], str(x[4])))

            beam = candidates[: args.beam_width]

    # If we exit loop, we failed to find goal within max_steps
    # Return the best path found so far (lowest error)
    best = beam[0] if beam else (0, 0, 0, [], [], 0)

    return {
        "problem": prob_file,
        "search_solved": False,
        "plan_len": len(best[4]),
        "plan": best[4],
    }


def run_inference(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Encoder
    if args.encoding == "fsf":
        # 1. Load Config
        config_path = os.path.join(
            args.data_dir, "encodings", "models", f"{args.domain}_fsf_config.json"
        )
        if not os.path.exists(config_path):
            print(f"Error: FSF Config not found at {config_path}")
            return

        with open(config_path, "r") as f:
            config = json.load(f)
            max_objects = config["max_objects"]

        # 2. Init Encoder
        domain_pddl = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
        feature_gen = FSFEncoder(args.domain, domain_pddl, max_objects)

        # 3. Set Input Dim (Max Objects + 1 Global)
        input_dim = feature_gen.vector_size
        print(f"FSF Input Dimension: {input_dim}")
    else:
        model_path = os.path.join(
            args.data_dir, "encodings", "models", f"{args.domain}_wl.json"
        )
        feature_gen = load_feature_generator(model_path)
        input_dim = feature_gen.get_n_features()

    # # 1. Load WL Generator
    # model_path = os.path.join(
    #     args.data_dir, "encodings", "models", f"{args.domain}_wl.json"
    # )
    # if not os.path.exists(model_path):
    #     print(f"Error: WL Model not found at {model_path}")
    #     return

    # print(f"Loading WL Model from {model_path}...")
    # feature_gen = load_feature_generator(model_path)
    # input_dim = feature_gen.get_n_features()
    # print(f"Feature Dimension: {input_dim}")

    # # 2. Load Domain (for parsing)
    # domain_pddl = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
    # wl_domain = parse_domain(domain_pddl)
    # pred_map = {p.name: p for p in wl_domain.predicates}

    # Load Model
    print(f"Loading LSTM from {args.checkpoint}...")
    if args.delta:
        model = StateCentricLSTM_Delta(input_dim, hidden_dim=args.hidden_dim).to(device)
    else:
        model = StateCentricLSTM(input_dim, hidden_dim=args.hidden_dim).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 4. Run on Splits
    splits = ["validation", "test-interpolation", "test-extrapolation"]

    for split in splits:
        print(f"\n*** Testing on {split} ***")
        split_dir = os.path.join(args.pddl_dir, args.domain, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} (not found)")
            continue

        results = []
        solved_count = 0
        executable_count = 0
        prob_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pddl")])
        print(f" Found {len(prob_files)} problems for {split}")

        for prob_file in tqdm(prob_files, desc=f"Solving {split}"):
            prob_path = os.path.join(split_dir, prob_file)
            try:
                # Generate Plan
                res = solve_problem(
                    args, prob_file, model, device, args.encoding, feature_gen
                )

                # Validate Plan with VAL
                print(" Validating plan with VAL...")
                is_solved, is_executable = validate_plan(
                    os.path.join(args.pddl_dir, args.domain, "domain.pddl"),
                    os.path.join(split_dir, prob_file),
                    res["plan"],
                    args.val_path,
                )

                # Store distinct metrics
                res["val_solved"] = is_solved  # Goal Reached
                res["val_executable"] = is_executable  # Physics Respected

                # The final "solved" metric should rely on VAL
                res["solved"] = is_solved

                results.append(res)

                if is_solved:
                    solved_count += 1
                if is_executable:
                    executable_count += 1

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error processing {prob_file}: {e}")
                results.append({"problem": prob_file, "solved": False, "error": str(e)})

        # Report
        total = len(prob_files)
        accuracy = solved_count / total if total else 0
        exec_rate = executable_count / total if total else 0

        print(
            f"Result {split}: Solved {solved_count}/{total} ({accuracy:.2%}) | Executable {executable_count}/{total} ({exec_rate:.2%})"
        )

        # Save
        os.makedirs(args.results_dir, exist_ok=True)
        tag_suffix = f"_{args.tag}" if getattr(args, "tag", "") else ""
        out_file = os.path.join(
            args.results_dir,
            f"{args.domain}_{args.encoding}_{split}{tag_suffix}_results.json",
        )
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoding", required=True, choices=["graphs", "fsf"])
    parser.add_argument("--pddl_dir", default="data/pddl")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--beam_width", type=int, default=3, help="Search beam width")
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

    HOME = os.path.expanduser("~")
    ROOT_DIR = f"{HOME}/planning/"
    parser.add_argument(
        "--val_path",
        default=os.environ.get("VAL_PATH", f"{ROOT_DIR}VAL/bin/Validate"),
        help="Path to VAL binary",
    )

    args = parser.parse_args()

    run_inference(args)
