import argparse
import csv
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

from .utils.pddl_utils import get_initial_state, parse_val_output_to_trajectory

HOME = os.path.expanduser("~")

# CONFIGURATION (for local)
# ROOT_DIR = f"{HOME}/usc/ai4s/libraries/planning/"
# VAL_PATH = os.environ.get("VAL_PATH", f"{ROOT_DIR}VAL/validate")

# CONFIGURATION (for Anvil)
ROOT_DIR = f"{HOME}/planning/"
VAL_PATH = os.environ.get("VAL_PATH", f"{ROOT_DIR}VAL/bin/Validate")

def generate_state_trajectory(args):
    plan_file, domain_file, problem_file, output_state_file = args
    problem_id = str(problem_file)

    if output_state_file.exists():
        return {
            "problem": problem_id,
            "status": "exists",
            "reason": "State file already exists",
        }

    output_state_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Run VAL in verbose mode
    val_log_file = output_state_file.with_suffix(".val.log")
    cmd = [VAL_PATH, "-v", str(domain_file), str(problem_file), str(plan_file)]

    try:
        with open(val_log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        # VAL failed. Read the log to find out why.
        reason = "VAL execution failed"
        if val_log_file.exists():
            try:
                with open(val_log_file, "r") as f:
                    lines = f.readlines()
                    # Grab the last non-empty lines which usually contain the error
                    tail = [lin.strip() for lin in lines if lin.strip()][-3:]
                    reason = f"VAL Error: {'; '.join(tail)}"
            except Exception:
                pass
        return {"problem": problem_id, "status": "val_failed", "reason": reason}

    # 2. Parse Initial State from PDDL
    try:
        initial_state = get_initial_state(str(domain_file), str(problem_file))
        if not initial_state:
            return {
                "problem": problem_id,
                "status": "pddl_parse_failed",
                "reason": "pddlpy returned empty initial state",
            }
    except Exception as e:
        return {"problem": problem_id, "status": "pddl_parse_failed", "reason": str(e)}

    # 3. Parse VAL output to get trajectory
    try:
        trajectory = parse_val_output_to_trajectory(val_log_file, initial_state)
        if not trajectory:
            return {
                "problem": problem_id,
                "status": "traj_parse_failed",
                "reason": "Reconstructed trajectory is empty (0 states)",
            }
    except Exception as e:
        return {"problem": problem_id, "status": "traj_parse_failed", "reason": str(e)}

    # 4. Save Trajectory
    try:
        with open(output_state_file, "w") as f:
            for state_set in trajectory:
                sorted_preds = sorted(list(state_set))
                f.write(" ".join(sorted_preds) + "\n")
    except Exception as e:
        return {"problem": problem_id, "status": "write_failed", "reason": str(e)}

    # Cleanup log file only on success (keep it on failure for manual inspection if needed?
    # Actually, let's delete it to save space, the CSV reason should be enough).
    val_log_file.unlink(missing_ok=True)

    return {"problem": problem_id, "status": "success", "reason": "OK"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pddl_dir", default="data/pddl")
    parser.add_argument("--plans_dir", default="data/plans")
    parser.add_argument("--output_dir", default="data/states")
    parser.add_argument("--report_path", default="code/data-processing/logs/")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = []
    plans_path = Path(args.plans_dir)
    pddl_path = Path(args.pddl_dir)
    out_path = Path(args.output_dir)

    # Walk through plans directory
    for domain_dir in plans_path.iterdir():
        if not domain_dir.is_dir():
            continue

        domain_file = pddl_path / domain_dir.name / "domain.pddl"
        if not domain_file.exists():
            print(f"Warning: Domain file not found for {domain_dir.name}")
            continue

        for root, _, files in os.walk(domain_dir):
            for file in files:
                if file.endswith(".plan"):
                    plan_file = Path(root) / file

                    # Reconstruct path to original problem file
                    rel_path = plan_file.relative_to(plans_path)
                    problem_file = pddl_path / rel_path.with_suffix(".pddl")

                    if not problem_file.exists():
                        continue

                    # Output path
                    state_file = out_path / rel_path.with_suffix(".traj")

                    tasks.append((plan_file, domain_file, problem_file, state_file))

    print(f"Found {len(tasks)} plans to convert. Starting conversion...")

    results = []
    stats = {
        "success": 0,
        "exists": 0,
        "val_failed": 0,
        "pddl_parse_failed": 0,
        "traj_parse_failed": 0,
        "write_failed": 0,
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in tqdm(
            executor.map(generate_state_trajectory, tasks), total=len(tasks)
        ):
            results.append(res)
            stats[res["status"]] += 1

    # Save Detailed Report
    report_path = args.report_path + "state_generation_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "status", "reason"])
        writer.writeheader()
        writer.writerows(results)

    print("\nConversion Complete.")
    print(f"Detailed report saved to: {report_path}")
    pprint(stats)


if __name__ == "__main__":
    main()
