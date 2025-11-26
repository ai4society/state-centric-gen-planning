import argparse
import os
import numpy as np
from tqdm import tqdm
from code.common.wl_wrapper import WLEncoder

# Configuration
DATA_DIR = "./data"
OUTPUT_DIR = os.path.join(DATA_DIR, "encodings", "graphs")
ALL_DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]


def process_single_file(task_args, encoder):
    """
    Worker function for embedding.
    """
    traj_path, prob_pddl, out_traj_path, out_goal_path = task_args

    try:
        # A. Embed Trajectory
        with open(traj_path, "r") as f:
            lines = f.read().strip().split("\n")

        traj_embs = []
        for line in lines:
            s = encoder.parse_state_string_to_wl_state(line)
            emb = encoder.embed_state(s, prob_pddl)
            traj_embs.append(emb)

        np.save(out_traj_path, np.array(traj_embs))

        # B. Embed Goal
        g_state = encoder.parse_pddl_goal_to_wl_state(prob_pddl)
        g_emb = encoder.embed_state(g_state, prob_pddl)
        np.save(out_goal_path, np.array(g_emb))

        return True
    except Exception as e:
        print(f"Error processing {traj_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations", type=int, default=2, help="WL iterations (k-hop)"
    )
    parser.add_argument(
        "--domain", type=str, default=None, help="Specific domain to run (e.g., blocks)"
    )
    args = parser.parse_args()

    domains_to_run = [args.domain] if args.domain else ALL_DOMAINS

    for domain in domains_to_run:
        if domain not in ALL_DOMAINS:
            print(f"Warning: {domain} is not in the standard list, but trying anyway.")

        print(f"\n=== Processing Domain: {domain} ===")
        domain_pddl = os.path.join(DATA_DIR, "pddl", domain, "domain.pddl")
        if not os.path.exists(domain_pddl):
            print(f"  [Warn] Domain PDDL not found at {domain_pddl}. Skipping.")
            continue

        # 1. Initialize and Collect (MUST BE SEQUENTIAL)
        # The vocabulary hash map is built here.
        encoder = WLEncoder(domain_pddl, iterations=args.iterations)
        train_states_dir = os.path.join(DATA_DIR, "states", domain, "train")

        try:
            encoder.collect_vocabulary(train_states_dir)
        except Exception as e:
            print(f"Skipping {domain} due to collection error: {e}")
            continue

        # 2. Prepare Tasks
        tasks = []
        for split in SPLITS:
            split_state_dir = os.path.join(DATA_DIR, "states", domain, split)
            split_pddl_dir = os.path.join(DATA_DIR, "pddl", domain, split)
            split_out_dir = os.path.join(OUTPUT_DIR, domain, split)
            os.makedirs(split_out_dir, exist_ok=True)

            if not os.path.exists(split_state_dir):
                continue

            for traj_file in os.listdir(split_state_dir):
                if not traj_file.endswith(".traj"):
                    continue

                prob_name = traj_file.replace(".traj", "")
                prob_pddl = os.path.join(split_pddl_dir, f"{prob_name}.pddl")
                traj_path = os.path.join(split_state_dir, traj_file)
                out_traj = os.path.join(split_out_dir, f"{prob_name}.npy")
                out_goal = os.path.join(split_out_dir, f"{prob_name}_goal.npy")

                if os.path.exists(prob_pddl):
                    tasks.append((traj_path, prob_pddl, out_traj, out_goal))

        # 3. Run Embedding (SEQUENTIAL)
        print(f"  [Embedding] Processing {len(tasks)} files sequentially...")

        success_count = 0
        for task in tqdm(tasks, desc="Embedding"):
            if process_single_file(task, encoder):
                success_count += 1

        print(f"  [Embedding] Completed. Success rate: {success_count}/{len(tasks)}")

        # 4. VERIFICATION CHECK
        print(f"  [Check] Verifying output dimensions for {domain}...")
        if len(tasks) > 0:
            check_file = tasks[0][2]  # The output .npy file
            if os.path.exists(check_file):
                data = np.load(check_file)
                dims = data.shape
                print(f"    File: {os.path.basename(check_file)}")
                print(f"    Shape: {dims}")

                # Check if the second dimension (features) is > 1
                if len(dims) > 1 and dims[1] > 1:
                    print("    STATUS: SUCCESS (Vector embedding generated)")
                else:
                    print("    STATUS: FAILURE (Scalar embedding detected)")
            else:
                print("    STATUS: FAILURE (Output file not created)")


if __name__ == "__main__":
    main()
