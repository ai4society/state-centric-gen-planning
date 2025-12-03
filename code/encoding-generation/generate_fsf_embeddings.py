import argparse
import json
import os
from code.common.fsf_wrapper import FSFEncoder

import numpy as np
import pddl
from tqdm import tqdm

ALL_DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]


def scan_max_objects(domain_pddl_path, domain_root_dir):
    """
    Scans all PDDL files in the domain directory to find the maximum number of objects.
    """
    print("  [Scan] Scanning all problems to determine Max Objects...")

    # Parse domain to get constants
    dom = pddl.parse_domain(domain_pddl_path)
    num_constants = len(dom.constants)

    max_objs = 0

    # Walk through all splits
    for root, _, files in os.walk(domain_root_dir):
        for file in files:
            if file.endswith(".pddl") and file != "domain.pddl":
                try:
                    path = os.path.join(root, file)
                    prob = pddl.parse_problem(path)
                    # Total objects = Problem Objects + Domain Constants
                    count = len(prob.objects) + num_constants
                    if count > max_objs:
                        max_objs = count
                except Exception:
                    pass

    print(f"  [Scan] Max Objects found: {max_objs}")
    return max_objs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="data/encodings/fsf")
    parser.add_argument("--model_dir", default="data/encodings/models")
    parser.add_argument("--domain", type=str, default=None)
    args = parser.parse_args()

    domains_to_run = [args.domain] if args.domain else ALL_DOMAINS

    for domain_name in domains_to_run:
        print(f"\n*** Processing Domain: {domain_name} (FSF) ***")

        # Paths
        domain_pddl = os.path.join(args.data_dir, "pddl", domain_name, "domain.pddl")
        domain_root = os.path.join(args.data_dir, "pddl", domain_name)

        if not os.path.exists(domain_pddl):
            print(f"  [Error] Domain PDDL not found: {domain_pddl}")
            continue

        # 1. SCAN for Max Objects
        max_objects = scan_max_objects(domain_pddl, domain_root)

        # 2. Save Config
        os.makedirs(args.model_dir, exist_ok=True)
        config_path = os.path.join(args.model_dir, f"{domain_name}_fsf_config.json")
        with open(config_path, "w") as f:
            json.dump({"max_objects": max_objects}, f)
        print(f"  [Config] Saved max_objects to {config_path}")

        # 3. Initialize Encoder
        encoder = FSFEncoder(domain_name, domain_pddl, max_objects)

        # 4. Embed
        for split in SPLITS:
            print(f"  Processing split: {split}")
            split_state_dir = os.path.join(args.data_dir, "states", domain_name, split)
            split_pddl_dir = os.path.join(args.data_dir, "pddl", domain_name, split)
            split_out_dir = os.path.join(args.output_dir, domain_name, split)

            os.makedirs(split_out_dir, exist_ok=True)

            if not os.path.exists(split_state_dir):
                continue

            traj_files = sorted(
                [f for f in os.listdir(split_state_dir) if f.endswith(".traj")]
            )

            for i, t_file in enumerate(tqdm(traj_files, desc=f"Embedding {split}")):
                prob_name = t_file.replace(".traj", "")
                prob_pddl = os.path.join(split_pddl_dir, f"{prob_name}.pddl")
                traj_path = os.path.join(split_state_dir, t_file)

                out_traj_path = os.path.join(split_out_dir, f"{prob_name}.npy")
                out_goal_path = os.path.join(split_out_dir, f"{prob_name}_goal.npy")

                if not os.path.exists(prob_pddl):
                    continue

                try:
                    # Verbose on first file
                    is_verbose = i == 0
                    traj_matrix = encoder.embed_trajectory(
                        prob_pddl, traj_path, verbose=is_verbose
                    )
                    goal_vec = encoder.embed_goal(prob_pddl)

                    # 3. Save
                    np.save(out_traj_path, traj_matrix)
                    np.save(out_goal_path, goal_vec)

                except Exception as e:
                    print(f"    Error embedding {prob_name}: {e}")
                    import traceback

                    traceback.print_exc()


if __name__ == "__main__":
    main()
