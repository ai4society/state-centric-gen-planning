import argparse
import os
import re

import numpy as np
from tqdm import tqdm

# WLPlan Imports
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import init_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem

ALL_DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]


def parse_traj_line_to_state(line, pred_map):
    """
    Parses a line like "(on a b) (clear c)" into a wlplan State object.
    """
    line = line.strip()
    if not line:
        return State([])

    # Regex to find all (predicate arg1 arg2 ...) groups
    matches = re.findall(r"\(([\w-]+(?: [\w-]+)*)\)", line)
    atoms = []

    for m in matches:
        parts = m.split()
        pred_name = parts[0]
        objs = parts[1:]

        if pred_name in pred_map:
            # Create Atom: (Predicate, [objects])
            atoms.append(Atom(pred_map[pred_name], objs))

    return State(atoms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="data/encodings/graphs")
    parser.add_argument("--model_dir", default="data/encodings/models")
    parser.add_argument("--iterations", type=int, default=2, help="WL iterations")
    parser.add_argument("--domain", type=str, default=None, help="Specific domain")
    args = parser.parse_args()

    # Make sure directories exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "pddl"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "states"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "plans"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "encodings"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "encodings", "graphs"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "encodings", "models"), exist_ok=True)

    domains_to_run = [args.domain] if args.domain else ALL_DOMAINS

    for domain_name in domains_to_run:
        print(f"\n=== Processing Domain: {domain_name} ===")

        # Paths
        domain_pddl = os.path.join(args.data_dir, "pddl", domain_name, "domain.pddl")
        train_states_dir = os.path.join(args.data_dir, "states", domain_name, "train")

        if not os.path.exists(domain_pddl):
            print(f"  [Error] Domain PDDL not found: {domain_pddl}")
            continue

        # 1. Parse Domain
        try:
            wl_domain = parse_domain(domain_pddl)
        except Exception as e:
            print(f"  [Error] Failed to parse domain: {e}")
            continue

        pred_map = {p.name: p for p in wl_domain.predicates}

        # 2. Initialize Feature Generator (ILG = Instance Learning Graph)
        feature_gen = init_feature_generator(
            feature_algorithm="wl",
            domain=wl_domain,
            graph_representation="ilg",
            iterations=args.iterations,
            pruning="none",
            multiset_hash=True,
        )

        # 3. Collect Vocabulary (Train Split Only)
        print("  [WL] Collecting vocabulary from training data...")

        # We need to load a subset of training data to build the vocabulary.
        # Loading ALL training states might be slow, but it ensures full coverage.
        train_files = sorted(
            [f for f in os.listdir(train_states_dir) if f.endswith(".traj")]
        )
        pddl_train_dir = os.path.join(args.data_dir, "pddl", domain_name, "train")

        wl_problems = []

        for t_file in tqdm(train_files, desc="Parsing Train"):
            prob_name = t_file.replace(".traj", "")
            prob_pddl = os.path.join(pddl_train_dir, f"{prob_name}.pddl")
            traj_path = os.path.join(train_states_dir, t_file)

            if not os.path.exists(prob_pddl):
                continue

            try:
                wl_prob = parse_problem(domain_pddl, prob_pddl)

                # Read trajectory
                with open(traj_path, "r") as f:
                    lines = f.readlines()

                # Parse states
                states = [parse_traj_line_to_state(line, pred_map) for line in lines]

                wl_problems.append(ProblemDataset(wl_prob, states))
            except Exception:
                continue

        # Collect
        if not wl_problems:
            print("  [Error] No valid training data found.")
            continue

        full_train_ds = DomainDataset(wl_domain, wl_problems)
        feature_gen.collect(full_train_ds)
        print(f"  [WL] Vocabulary Size: {feature_gen.get_n_features()}")

        # 4. Save Feature Generator (JSON)
        os.makedirs(args.model_dir, exist_ok=True)
        save_path = os.path.join(args.model_dir, f"{domain_name}_wl.json")
        feature_gen.save(save_path)
        print(f"  [WL] Saved model to {save_path}")

        # 5. Embed All Splits
        for split in SPLITS:
            print(f"  [WL] Embedding split: {split}")
            split_state_dir = os.path.join(args.data_dir, "states", domain_name, split)
            split_pddl_dir = os.path.join(args.data_dir, "pddl", domain_name, split)
            split_out_dir = os.path.join(args.output_dir, domain_name, split)
            os.makedirs(split_out_dir, exist_ok=True)

            if not os.path.exists(split_state_dir):
                continue

            traj_files = sorted([f for f in os.listdir(split_state_dir) if f.endswith(".traj")])

            for t_file in tqdm(traj_files, desc=f"Embedding {split}"):
                prob_name = t_file.replace(".traj", "")
                prob_pddl = os.path.join(split_pddl_dir, f"{prob_name}.pddl")
                traj_path = os.path.join(split_state_dir, t_file)
                out_traj_path = os.path.join(split_out_dir, f"{prob_name}.npy")
                out_goal_path = os.path.join(split_out_dir, f"{prob_name}_goal.npy")

                if not os.path.exists(prob_pddl):
                    continue

                try:
                    # Parse Problem & States
                    wl_prob = parse_problem(domain_pddl, prob_pddl)
                    with open(traj_path, "r") as f:
                        lines = f.readlines()
                    states = [parse_traj_line_to_state(l, pred_map) for l in lines]

                    # Embed Trajectory
                    # feature_gen.embed returns a flattened list of vectors [v_s0, v_s1, ...]
                    mini_ds = DomainDataset(
                        wl_domain, [ProblemDataset(wl_prob, states)]
                    )
                    embs = feature_gen.embed(mini_ds)

                    # Use 'embs' directly, not 'embs[0]'
                    traj_matrix = np.array(embs, dtype=np.float32)  # [T, D]

                    # Embed Goal
                    # We create a dummy state containing only the goal atoms
                    goal_atoms = list(wl_prob.positive_goals)
                    goal_state = State(goal_atoms)

                    goal_ds = DomainDataset(
                        wl_domain, [ProblemDataset(wl_prob, [goal_state])]
                    )
                    goal_embs = feature_gen.embed(goal_ds)

                    # Use 'goal_embs[0]' (the first and only vector)
                    goal_vec = np.array(goal_embs[0], dtype=np.float32)  # [D]

                    # Save
                    np.save(out_traj_path, traj_matrix)
                    np.save(out_goal_path, goal_vec)

                except Exception as e:
                    print(f"    Error embedding {prob_name}: {e}")


if __name__ == "__main__":
    main()
