import argparse
import os
import re
from code.common.wl_wrapper import WLEncoder

import numpy as np

# Configuration
DATA_DIR = "./data"
OUTPUT_DIR = os.path.join(DATA_DIR, "encodings", "graphs")
DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]

# Regex to parse "(on a b)" -> "on a b"
PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations", type=int, default=2, help="WL iterations (k-hop)"
    )
    args = parser.parse_args()

    for domain in DOMAINS:
        print(f"\n=== Processing Domain: {domain} ===")
        domain_pddl = os.path.join(DATA_DIR, "pddl", domain, "domain.pddl")
        if not os.path.exists(domain_pddl):
            print(f"  [Warn] Domain PDDL not found at {domain_pddl}. Skipping.")
            continue

        # 1. Initialize and Collect
        encoder = WLEncoder(domain_pddl, iterations=args.iterations)
        train_states_dir = os.path.join(DATA_DIR, "states", domain, "train")

        try:
            encoder.collect_vocabulary(train_states_dir)
        except Exception as e:
            print(f"Skipping {domain}: {e}")
            continue

        # 2. Embed All Splits
        for split in SPLITS:
            split_state_dir = os.path.join(DATA_DIR, "states", domain, split)
            split_pddl_dir = os.path.join(DATA_DIR, "pddl", domain, split)
            split_out_dir = os.path.join(OUTPUT_DIR, domain, split)
            os.makedirs(split_out_dir, exist_ok=True)

            if not os.path.exists(split_state_dir):
                continue

            # Iterate files
            for traj_file in os.listdir(split_state_dir):
                if not traj_file.endswith(".traj"):
                    continue

                prob_name = traj_file.replace(".traj", "")
                prob_pddl = os.path.join(split_pddl_dir, f"{prob_name}.pddl")
                traj_path = os.path.join(split_state_dir, traj_file)

                if not os.path.exists(prob_pddl):
                    continue

                try:
                    # A. Embed Trajectory
                    with open(traj_path, "r") as f:
                        lines = f.read().strip().split("\n")

                    # Batch embed is not easily exposed in wrapper for simplicity,
                    # but we can do loop (or optimize wrapper later).
                    # For data gen, speed is less critical than correctness.
                    traj_embs = []
                    for line in lines:
                        s = encoder.parse_state_string_to_wl_state(line)
                        emb = encoder.embed_state(s, prob_pddl)
                        traj_embs.append(emb)

                    np.save(
                        os.path.join(split_out_dir, f"{prob_name}.npy"),
                        np.array(traj_embs),
                    )

                    # B. Embed Goal
                    g_state = encoder.parse_pddl_goal_to_wl_state(prob_pddl)
                    g_emb = encoder.embed_state(g_state, prob_pddl)
                    np.save(
                        os.path.join(split_out_dir, f"{prob_name}_goal.npy"),
                        np.array(g_emb),
                    )

                except Exception as e:
                    print(f"Error {prob_name}: {e}")


if __name__ == "__main__":
    main()
