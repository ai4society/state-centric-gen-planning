import argparse
import json
import os
import pickle
from code.common.fsf_wrapper import FSFEncoder
from code.common.utils import set_seed, validate_plan

import numpy as np
import xgboost as xgb
from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from scipy.spatial.distance import cosine, euclidean
from torch import cuda
from tqdm import tqdm

# WL Imports
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import load_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem


def embed_state_wl(atoms_set, feature_gen, wl_domain, wl_prob, pred_map):
    """
    Helper to convert a set of atoms (strings) into a Numpy Array [1, D] using WL.
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


def embed_state_fsf(atoms_set, encoder, objects, obj_map):
    """
    Helper to convert a set of atoms (strings) into a Numpy Array [1, D] using FSF.
    """
    # Convert set of strings to list of tuples: "(on a b)" -> ("on", "a", "b")
    atom_tuples = []
    for a in atoms_set:
        content = a.replace("(", "").replace(")", "").lower()
        parts = content.split()
        if parts:
            atom_tuples.append(tuple(parts))

    # Use the encoder's internal logic
    vec = encoder._state_to_vector(atom_tuples, objects, obj_map)

    # Return [1, D]
    return vec.reshape(1, -1)


def solve_problem(
    prob_file,
    domain_path,
    prob_path,
    model,
    max_steps,
    delta,
    encoding_type,
    feature_encoder=None,  # WL Generator or FSFEncoder
    wl_domain=None,
    pred_map=None,
    beam_width=3,
):
    """
    Runs Latent Space Search using Beam Search (XGBoost version).
    Supports both WL and FSF encodings.
    """
    print(
        f"Inference using {'Delta Prediction' if delta else 'State Prediction'} for {prob_file}"
    )

    # 1. Pyperplan Parsing (Ground Truth Physics)
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

    # 2. Embedding Setup (Goal & Init)
    if encoding_type == "fsf":
        # FSF Setup
        encoder = feature_encoder
        # FSF requires problem-specific object mapping
        objects = encoder._get_sorted_objects(prob_path)
        obj_map = encoder._get_object_indices(objects)

        # Embed Goal
        goal_vec_1d = encoder.embed_goal(prob_path)
        goal_vec = goal_vec_1d.reshape(1, -1)  # [1, D]

        # Embed Init
        init_vec = embed_state_fsf(initial_atoms, encoder, objects, obj_map)
    else:
        # WL Setup
        feature_gen = feature_encoder
        wl_prob = parse_problem(domain_path, prob_path)

        # Embed Goal
        goal_atoms = list(wl_prob.positive_goals)
        goal_state = State(goal_atoms)
        goal_ds = DomainDataset(wl_domain, [ProblemDataset(wl_prob, [goal_state])])
        goal_embs = feature_gen.embed(goal_ds)
        goal_vec = np.array(goal_embs[0], dtype=np.float32).reshape(1, -1)

        # Embed Init
        init_vec = embed_state_wl(
            initial_atoms, feature_gen, wl_domain, wl_prob, pred_map
        )

    # 3. Initialize Beam
    # Beam Element: (score, current_vec, atoms, plan, visited_hashes)
    # Note: XGBoost is stateless (no hidden state), unlike LSTM.
    initial_hash = frozenset(initial_atoms)

    # Explicitly initialize set to avoid dict confusion
    visited_set = set()
    visited_set.add(initial_hash)

    beam = [(0.0, init_vec, initial_atoms, [], visited_set)]

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

            # Reshape is crucial: XGBoost might return (D,) or (1, D)
            pred = pred.reshape(1, -1)

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

            # Sort successors by name to ensure processing order is deterministic
            successors.sort(key=lambda x: x[0])

            # C. Score Successors
            for op_name, next_atoms in successors:
                next_hash = frozenset(next_atoms)
                if next_hash in visited:
                    continue

                # Embed Candidate
                if encoding_type == "fsf":
                    cand_vec = embed_state_fsf(next_atoms, encoder, objects, obj_map)
                else:
                    cand_vec = embed_state_wl(
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

            # Stable Sort:
            # Primary Key: Score (float)
            # Secondary Key: String representation of the plan (deterministic tie-breaker)
            candidates.sort(key=lambda x: (x[0], str(x[3])))
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

    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 0. Load Metadata to determine encoding
    meta_path = os.path.join(
        args.checkpoint_dir, f"{args.domain}_xgb_meta_seed{args.seed}.pkl"
    )
    if not os.path.exists(meta_path):
        print(f"Error: Metadata not found at {meta_path}. Cannot determine encoding.")
        return

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    encoding_type = meta.get("encoding", "graphs")
    # Override delta with what the model was actually trained on
    trained_delta = meta.get("delta", args.delta)
    if trained_delta != args.delta:
        print(
            f"Warning: Argument --delta={args.delta} but model was trained with delta={trained_delta}. Using model setting."
        )

    print(f"Detected Encoding: {encoding_type} | Delta Mode: {trained_delta}")

    # 1. Load Encoders
    feature_encoder = None
    wl_domain = None
    pred_map = None

    if encoding_type == "fsf":
        # Load FSF Config
        config_path = os.path.join(
            args.data_dir, "encodings", "models", f"{args.domain}_fsf_config.json"
        )
        if not os.path.exists(config_path):
            print(f"Error: FSF Config not found at {config_path}")
            return

        with open(config_path, "r") as f:
            config = json.load(f)
            max_objects = config["max_objects"]

        domain_pddl = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
        feature_encoder = FSFEncoder(args.domain, domain_pddl, max_objects)
        print(f"Initialized FSF Encoder with Max Objects: {max_objects}")

    else:
        # Load WL Generator
        model_path = os.path.join(
            args.data_dir, "encodings", "models", f"{args.domain}_wl.json"
        )
        if not os.path.exists(model_path):
            print(f"Error: WL Model not found at {model_path}")
            return

        print(f"Loading WL Model from {model_path}...")
        feature_encoder = load_feature_generator(model_path)

        # Load Domain for WL parsing
        domain_pddl = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
        wl_domain = parse_domain(domain_pddl)
        pred_map = {p.name: p for p in wl_domain.predicates}

    # 2. Load XGBoost
    xgb_path = os.path.join(
        args.checkpoint_dir, f"{args.domain}_xgb_seed{args.seed}.ubj"
    )
    print(f"Loading XGBoost from {xgb_path}...")

    model = xgb.XGBRegressor(device=device)
    model.load_model(xgb_path)

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
                res = solve_problem(
                    prob_file=prob_file,
                    domain_path=domain_pddl,
                    prob_path=prob_path,
                    model=model,
                    max_steps=args.max_steps,
                    delta=trained_delta,
                    encoding_type=encoding_type,
                    feature_encoder=feature_encoder,
                    wl_domain=wl_domain,
                    pred_map=pred_map,
                    beam_width=args.beam_width,
                )

                # Validate Plan with VAL (External Verification)
                print(" Validating plan with VAL...")
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
                import traceback

                traceback.print_exc()
                results.append({"problem": prob_file, "solved": False, "error": str(e)})

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
            f"{args.domain}_{encoding_type}_{split}{tag_suffix}_seed{args.seed}_results.json",
        )
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--pddl_dir", default="data/pddl")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--beam_width", type=int, default=3, help="Search beam width")
    parser.add_argument("--delta", action="store_true")
    parser.add_argument("--tag", default="state")
    parser.add_argument("--seed", type=int, default=15)

    HOME = os.path.expanduser("~")
    ROOT_DIR = f"{HOME}/planning/"
    parser.add_argument(
        "--val_path",
        default=os.environ.get("VAL_PATH", f"{ROOT_DIR}VAL/bin/Validate"),
        help="Path to VAL binary",
    )

    args = parser.parse_args()
    run_inference(args)
