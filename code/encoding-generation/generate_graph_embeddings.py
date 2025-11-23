import argparse
import glob
import os
import re

import numpy as np
from tqdm import tqdm
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import init_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem

# Configuration
DATA_DIR = "./data"
OUTPUT_DIR = os.path.join(DATA_DIR, "encodings", "graphs")
DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]

# Regex to parse "(on a b)" -> "on a b"
PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def parse_trajectory_to_states(traj_filepath, wlplan_domain):
    """
    Reads a .traj file and converts it into a list of wlplan.planning.State objects.
    """
    states = []

    # Pre-fetch predicate objects for faster lookups
    # wlplan_domain.predicates is a list of Predicate objects
    name_to_predicate = {p.name: p for p in wlplan_domain.predicates}

    with open(traj_filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract atoms strings: "(on a b) (clear c)" -> ["on a b", "clear c"]
            atom_strings = PREDICATE_REGEX.findall(line)

            atoms = []
            for atom_str in atom_strings:
                parts = atom_str.split()
                pred_name = parts[0]
                obj_names = parts[1:]

                if pred_name in name_to_predicate:
                    # Construct wlplan Atom
                    atom = Atom(
                        predicate=name_to_predicate[pred_name], objects=obj_names
                    )
                    atoms.append(atom)

            # Create State object
            states.append(State(atoms))

    return states


def get_pddl_paths(domain, split, problem_name):
    domain_path = os.path.join(DATA_DIR, "pddl", domain, "domain.pddl")
    # Handle potential naming mismatches if necessary, assuming standard structure here
    problem_path = os.path.join(DATA_DIR, "pddl", domain, split, f"{problem_name}.pddl")
    return domain_path, problem_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations", type=int, default=2, help="WL iterations (k-hop)"
    )
    args = parser.parse_args()

    for domain_name in DOMAINS:
        print(f"\n=== Processing Domain: {domain_name} ===")

        domain_pddl_path = os.path.join(DATA_DIR, "pddl", domain_name, "domain.pddl")
        if not os.path.exists(domain_pddl_path):
            print(f"  [Warn] Domain PDDL not found at {domain_pddl_path}. Skipping.")
            continue

        # 1. Parse Domain using wlplan
        try:
            wlplan_domain = parse_domain(domain_pddl_path)
        except Exception as e:
            print(f"  [Error] Failed to parse domain {domain_name}: {e}")
            continue

        # 2. Initialize Feature Generator
        # We use 'ilg' (Instance Learning Graph) as per your proposal
        print("  Initializing Feature Generator...")
        generator = init_feature_generator(
            feature_algorithm="wl",
            domain=wlplan_domain,
            graph_representation="ilg",
            iterations=args.iterations,
            pruning="none",
            multiset_hash=True,
        )

        # PHASE 1: COLLECT VOCABULARY (TRAIN SET ONLY)
        print("  Phase 1: Collecting vocabulary from Training set...")
        train_split_dir = os.path.join(DATA_DIR, "states", domain_name, "train")
        train_files = glob.glob(os.path.join(train_split_dir, "*.traj"))

        if not train_files:
            print("  [Error] No training files found. Cannot build vocabulary.")
            continue

        # We need to accumulate all training data into one DomainDataset for 'collect'
        train_problem_datasets = []

        for traj_file in tqdm(train_files, desc="  Parsing Train"):
            prob_name = os.path.splitext(os.path.basename(traj_file))[0]
            _, prob_pddl = get_pddl_paths(domain_name, "train", prob_name)

            if not os.path.exists(prob_pddl):
                continue

            try:
                # Parse Problem
                wlplan_prob = parse_problem(domain_pddl_path, prob_pddl)
                # Parse States
                wlplan_states = parse_trajectory_to_states(traj_file, wlplan_domain)

                if wlplan_states:
                    train_problem_datasets.append(
                        ProblemDataset(problem=wlplan_prob, states=wlplan_states)
                    )
            except Exception:
                # print(f"    Failed to parse {prob_name}: {e}")
                pass

        if not train_problem_datasets:
            print("  [Error] No valid training data parsed.")
            continue

        # Create the master training dataset and collect colors
        full_train_dataset = DomainDataset(
            domain=wlplan_domain, data=train_problem_datasets
        )
        generator.collect(full_train_dataset)
        print("  Vocabulary collected.")

        # PHASE 2: EMBED ALL SPLITS
        print("  Phase 2: Embedding all splits...")

        for split in SPLITS:
            split_input_dir = os.path.join(DATA_DIR, "states", domain_name, split)
            split_output_dir = os.path.join(OUTPUT_DIR, domain_name, split)
            os.makedirs(split_output_dir, exist_ok=True)

            traj_files = glob.glob(os.path.join(split_input_dir, "*.traj"))

            for traj_file in tqdm(traj_files, desc=f"  Embedding {split}"):
                prob_name = os.path.splitext(os.path.basename(traj_file))[0]
                _, prob_pddl = get_pddl_paths(domain_name, split, prob_name)

                if not os.path.exists(prob_pddl):
                    continue

                try:
                    # 1. Create a mini dataset for just this file
                    wlplan_prob = parse_problem(domain_pddl_path, prob_pddl)
                    wlplan_states = parse_trajectory_to_states(traj_file, wlplan_domain)

                    if not wlplan_states:
                        continue

                    single_file_dataset = DomainDataset(
                        domain=wlplan_domain,
                        data=[
                            ProblemDataset(problem=wlplan_prob, states=wlplan_states)
                        ],
                    )

                    # 2. Embed
                    # Returns a list of lists (or numpy array depending on version)
                    embeddings = generator.embed(single_file_dataset)

                    # 3. Save
                    embeddings_np = np.array(embeddings, dtype=np.float32)

                    # Flatten: embed returns [ [problem_1_states] ], we want [states]
                    if len(embeddings_np.shape) == 3:
                        embeddings_np = embeddings_np[0]

                    save_path = os.path.join(split_output_dir, f"{prob_name}.npy")
                    np.save(save_path, embeddings_np)

                except Exception as e:
                    print(f"Error embedding {prob_name}: {e}")


if __name__ == "__main__":
    main()
