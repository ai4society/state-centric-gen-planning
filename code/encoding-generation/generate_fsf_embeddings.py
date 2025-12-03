import argparse
import os
from code.common.fsf_wrapper import FSFEncoder

import numpy as np
from tqdm import tqdm

ALL_DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="data/encodings/fsf")
    parser.add_argument("--domain", type=str, default=None)
    args = parser.parse_args()

    domains_to_run = [args.domain] if args.domain else ALL_DOMAINS

    for domain_name in domains_to_run:
        print(f"\n*** Processing Domain: {domain_name} (FSF) ***")

        # Paths
        domain_pddl = os.path.join(args.data_dir, "pddl", domain_name, "domain.pddl")
        if not os.path.exists(domain_pddl):
            print(f"  [Error] Domain PDDL not found: {domain_pddl}")
            continue

        # Initialize Encoder
        encoder = FSFEncoder(domain_name, domain_pddl)

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

            for t_file in tqdm(traj_files, desc=f"Embedding {split}"):
                prob_name = t_file.replace(".traj", "")
                prob_pddl = os.path.join(split_pddl_dir, f"{prob_name}.pddl")
                traj_path = os.path.join(split_state_dir, t_file)

                out_traj_path = os.path.join(split_out_dir, f"{prob_name}.npy")
                out_goal_path = os.path.join(split_out_dir, f"{prob_name}_goal.npy")

                if not os.path.exists(prob_pddl):
                    continue

                try:
                    # 1. Embed Trajectory [T, MAX_OBJS]
                    traj_matrix = encoder.embed_trajectory(prob_pddl, traj_path)

                    # 2. Embed Goal [MAX_OBJS]
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
