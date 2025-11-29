import math
import os
import random
import argparse
import math
from tqdm import tqdm
from .generators import *

# Configuration with reduced counts for testing
# "complexity" can be a list [1, 2, 3] or a range object range(start, stop, step)
CONFIG = {
    "blocks": {
        "splits": {
            "train": {"count": 15, "complexity": [4, 6, 7]},  # 4, 6, 7 blocks
            "validation": {"count": 5, "complexity": [8]},  # Val size 8
            "test-interpolation": {
                "count": 5,
                "complexity": [5],
            },  # Inter. size 5 blocks
            "test-extrapolation": {
                "count": 45,
                "complexity": range(9, 18),
            },  # 9-17 blocks
        },
    },
    "gripper": {
        "splits": {
            "train": {"count": 20, "complexity": [2, 4, 6, 8]},  # 2, 4, 6, or 8 balls
            "validation": {
                "count": 10,
                "complexity": [9, 10],
            },  # Val. size 9 and 10 balls
            "test-interpolation": {
                "count": 15,
                "complexity": [3, 5, 7],
            },  # 3, 5, or 7 balls
            "test-extrapolation": {
                "count": 80,
                "complexity": range(12, 43, 2),
            },  # 12-42 balls (even numbers)
        },
    },
    "visitall-from-everywhere": {
        "splits": {
            # Complexity = Total Cells
            "train": {
                "count": 45,
                "complexity": [1, 3, 4, 6, 10, 11, 12, 14, 16],
            },  # cells ∈ {1,3,4,6,10,11,12,14,16}
            "validation": {"count": 10, "complexity": [18, 20]},  # 18 and 20 cells
            "test-interpolation": {
                "count": 25,
                "complexity": [2, 5, 8, 9, 15],
            },  # sizes {2,5,8,9,15} cells
            "test-extrapolation": {
                "count": 490,
                "complexity": range(24, 122),
            },  # sizes 24,25,…,121 cells
        },
    },
    "logistics": {
        "splits": {
            # Complexity = Number of Goals
            "train": {"count": 15, "complexity": [1, 3, 5]},  # goals ∈ {1,3,5}
            "validation": {"count": 5, "complexity": [6]},  # 6 goals
            "test-interpolation": {"count": 10, "complexity": [2, 4]},  # goals ∈ {2,4}
            "test-extrapolation": {
                "count": 45,
                "complexity": range(7, 16),
            },  # goals ∈ {7,8,…,15}
        },
    },
}


def get_grid_dimensions(n_cells):
    """Finds width and height such that w * h = n_cells."""
    # Find all factors
    factors = []
    for i in range(1, int(n_cells**0.5) + 1):
        if n_cells % i == 0:
            factors.append((i, n_cells // i))

    # Pick a random factor pair
    w, h = random.choice(factors)
    # Randomly swap width/height for variety
    if random.random() < 0.5:
        w, h = h, w
    return w, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pddl_dir", default="data/pddl")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)

    for domain, config in CONFIG.items():
        print(f"*** Generating Domain: {domain} ***")

        domain_dir = os.path.join(args.pddl_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)

        # 2. Generate Splits
        for split, split_cfg in config["splits"].items():
            count = split_cfg["count"]
            complexity_opts = split_cfg["complexity"]

            # Convert range to list if necessary
            possible_complexities = list(complexity_opts)

            # Build the list of target complexities to ensure coverage
            target_complexities = []

            # Priority 1: Add one of each available complexity
            if count >= len(possible_complexities):
                target_complexities.extend(possible_complexities)
                # Priority 2: Fill the rest with random choices
                remaining = count - len(possible_complexities)
                for _ in range(remaining):
                    target_complexities.append(random.choice(possible_complexities))
            else:
                # If count is too small to cover everything, take the first 'count' unique ones
                # (In a real run, count should be >> len(possible_complexities))
                print(
                    f"  [Warning] Count {count} is smaller than complexity options {len(possible_complexities)} for {split}. Truncating."
                )
                target_complexities = possible_complexities[:count]

            # Shuffle so the file indices don't correlate strictly with size
            random.shuffle(target_complexities)

            split_dir = os.path.join(domain_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            print(f"  Generating {split} ({len(target_complexities)} problems)...")

            for i, c in enumerate(tqdm(target_complexities)):
                problem_name = f"{domain}-{split}-{i}"
                filename = os.path.join(split_dir, f"{problem_name}.pddl")

                pddl_content = ""

                if domain == "blocks":
                    # c = number of blocks
                    pddl_content = generate_problem_blocks(c, problem_name)

                elif domain == "gripper":
                    # c = number of balls
                    pddl_content = generate_problem_gripper(c, problem_name)

                elif domain == "visitall-from-everywhere":
                    # c = total cells. Need to factorize into w * h
                    w, h = get_grid_dimensions(c)
                    pddl_content = generate_problem_visitall(w, h, problem_name)

                elif domain == "logistics":
                    # c = number of goals
                    n_goals = c

                    # Heuristics to scale problem size based on goals
                    # Ensure we have at least as many packages as goals
                    n_pkgs = n_goals + random.randint(0, 2)

                    # Scale cities roughly by packages (e.g., 1 city per 2-3 packages, min 2)
                    n_cities = max(2, math.ceil(n_pkgs / 2.5))

                    # Scale airplanes (1 per 2 cities, min 1)
                    n_planes = max(1, math.ceil(n_cities / 2))

                    # Locations per city (2 to 4)
                    n_locs = random.randint(2, 4)

                    pddl_content = generate_problem_logistics(
                        num_packages=n_pkgs,
                        num_cities=n_cities,
                        num_airplanes=n_planes,
                        num_locations_per_city=n_locs,
                        problem_name=problem_name,
                    )

                with open(filename, "w") as f:
                    f.write(pddl_content)


if __name__ == "__main__":
    main()
