import argparse
import csv
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

HOME = os.path.expanduser("~")

# CONFIGURATION (for local)
# ROOT_DIR = f"{HOME}/usc/ai4s/libraries/planning/"

# CONFIGURATION (for Anvil)
ROOT_DIR = f"{HOME}/planning/"

FD_PATH = os.environ.get("FD_PATH", f"{ROOT_DIR}downward/fast-downward.py")
TIMEOUT_BASELINE = "60s"
TIMEOUT_FALLBACK = "300s"

# 1. Baseline: Strict A* (for comparison)
CONFIG_BASELINE = ["--search", "astar(lmcut())"]

# 2. Fallback: Greedy Best First (for finding ANY plan for large problems)
CONFIG_FALLBACK = [
    "--evaluator",
    "h=ff()",
    "--search",
    "lazy_greedy([h], preferred=[h])",
]


def solve_problem(domain_path, problem_path, plan_output_path):
    """Runs FD. Returns dict with status and detailed reason."""

    # Use absolute paths because we will change CWD
    abs_domain = Path(domain_path).resolve()
    abs_problem = Path(problem_path).resolve()
    abs_plan_out = Path(plan_output_path).resolve()

    # Create a temporary directory for this specific process
    # FD writes 'output.sas' to CWD. This prevents race conditions.
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Try Baseline
        cmd_base = [
            "timeout",
            TIMEOUT_BASELINE,
            FD_PATH,
            "--plan-file",
            str(abs_plan_out),
            str(abs_domain),
            str(abs_problem),
        ] + CONFIG_BASELINE

        try:
            subprocess.run(
                cmd_base,
                cwd=tmp_dir,  # CRITICAL: Run in temp dir
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            if abs_plan_out.exists() and abs_plan_out.stat().st_size > 0:
                return {"status": "baseline", "reason": "Solved by baseline"}
        except subprocess.CalledProcessError:
            pass

        # 2. Try Fallback
        cmd_fall = [
            "timeout",
            TIMEOUT_FALLBACK,
            FD_PATH,
            "--plan-file",
            str(abs_plan_out),
            str(abs_domain),
            str(abs_problem),
        ] + CONFIG_FALLBACK

        try:
            subprocess.run(
                cmd_fall,
                cwd=tmp_dir,  # CRITICAL: Run in temp dir
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            if abs_plan_out.exists() and abs_plan_out.stat().st_size > 0:
                return {
                    "status": "fallback",
                    "reason": "Solved by fallback (Baseline failed)",
                }
            else:
                return {
                    "status": "failed",
                    "reason": "Fallback finished but generated empty plan file",
                }

        except subprocess.CalledProcessError as e:
            if e.returncode == 124:
                return {
                    "status": "failed",
                    "reason": f"Timeout in fallback ({TIMEOUT_FALLBACK})",
                }

            error_msg = (
                e.stderr.strip().splitlines()[-1] if e.stderr else "Unknown FD error"
            )
            return {"status": "failed", "reason": f"FD Error: {error_msg}"}


def process_file(args):
    pddl_file, domain_file, output_plan_file = args

    # Store relative path for cleaner reporting
    problem_id = str(pddl_file)

    if output_plan_file.exists():
        return {
            "problem": problem_id,
            "status": "exists",
            "reason": "Plan file already exists",
        }

    output_plan_file.parent.mkdir(parents=True, exist_ok=True)

    result = solve_problem(domain_file, pddl_file, output_plan_file)
    result["problem"] = problem_id
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/pddl")
    parser.add_argument("--output_dir", default="data/plans")
    parser.add_argument("--report_path", default="code/data-processing/logs/")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tasks = []
    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)

    # Walk through domains
    for domain_dir in data_path.iterdir():
        if not domain_dir.is_dir():
            continue

        domain_file = domain_dir / "domain.pddl"
        if not domain_file.exists():
            print(f"Skipping {domain_dir}, no domain.pddl found.")
            continue

        # Walk through subfolders (train, test, etc)
        for root, _, files in os.walk(domain_dir):
            for file in files:
                if file.endswith(".pddl") and file != "domain.pddl":
                    pddl_file = Path(root) / file

                    # Replicate structure
                    rel_path = pddl_file.relative_to(data_path)
                    plan_file = out_path / rel_path.with_suffix(".plan")

                    tasks.append((pddl_file, domain_file, plan_file))

    print(f"Found {len(tasks)} problems. Starting generation...")

    results = []
    stats = {"baseline": 0, "fallback": 0, "failed": 0, "exists": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Use list() to force execution and show progress bar
        for res in tqdm(executor.map(process_file, tasks), total=len(tasks)):
            results.append(res)
            stats[res["status"]] += 1

    # Save Detailed Report
    report_path = args.report_path + "plan_generation_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "status", "reason"])
        writer.writeheader()
        writer.writerows(results)

    print("\nGeneration Complete.")
    print(f"Detailed report saved to: {report_path}")
    pprint(stats)


if __name__ == "__main__":
    main()
