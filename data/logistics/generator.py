import random
import math
import os
import argparse
import textwrap
import sys # For exit

def generate_logistics_problem(num_packages: int,
                               num_cities: int,
                               num_locations: int,
                               num_airplanes: int,
                               num_goals: int,
                               problem_id: str,
                               seed: int | None = None, # Add seed for reproducibility reporting
                               args: argparse.Namespace | None = None # Add args to include in comment
                              ) -> str:
    """
    Generates a PDDL logistics problem definition string with descriptive package names.

    Args:
        num_packages: Total number of packages.
        num_cities: Number of cities.
        num_locations: Total number of locations (must be >= num_cities).
        num_airplanes: Number of airplanes.
        num_goals: Number of packages that should have a goal state.
        problem_id: A unique identifier string for the problem name (base).
        seed: The random seed used for generation (for logging in ID).
        args: The argparse.Namespace object containing the command-line arguments.

    Returns:
        A string containing the PDDL problem definition.

    Raises:
        ValueError: If parameters are invalid (e.g., num_cities < 1,
                    num_locations < num_cities, num_goals > num_packages).
    """
    # --- Input Validation ---
    if num_cities < 1:
        raise ValueError("Must have at least one city.")
    if num_locations < num_cities:
        raise ValueError(f"Number of locations ({num_locations}) must be at least the number of cities ({num_cities}) "
                         "to ensure each city gets at least one location.")
    if num_packages < 0:
        raise ValueError("Number of packages cannot be negative.")
    if num_airplanes < 1:
        raise ValueError("Must have at least one airplane.")
    if num_goals < 0 or (num_packages > 0 and num_goals > num_packages):
        raise ValueError("Number of goals must be between 0 and the total number of packages.")
    # This check is redundant due to the previous one, but kept for clarity if num_packages is 0
    if num_packages == 0 and num_goals > 0:
         raise ValueError("Cannot have goals if there are no packages.")


    # --- Object Naming and Creation (Non-Package) ---
    cities = [f"cit{i}" for i in range(1, num_cities + 1)]
    trucks = [f"tru{i}" for i in range(1, num_cities + 1)] # One truck per city
    airplanes = [f"apn{i}" for i in range(1, num_airplanes + 1)]
    # Packages list will be populated dynamically later

    # --- Location Generation and Distribution ---
    all_locations = [] # Final list of location names
    city_locations = {city: [] for city in cities} # Map city -> final location names

    # Generate all potential location names first
    # We need a way to map back to cities for distribution
    potential_locations_with_city_hint = []
    loc_counter_per_city = {city: 0 for city in cities}

    # Ensure at least one location per city by pre-assigning
    city_indices = list(range(num_cities))
    random.shuffle(city_indices)
    for i in range(num_cities):
        city_idx = city_indices[i]
        city = cities[city_idx]
        loc_counter_per_city[city] += 1
        loc_name = f"loc_c{city_idx+1}_{loc_counter_per_city[city]}"
        potential_locations_with_city_hint.append((loc_name, city))

    # Generate remaining locations and assign randomly
    remaining_locations_to_generate = num_locations - num_cities
    for _ in range(remaining_locations_to_generate):
        target_city = random.choice(cities)
        city_idx = cities.index(target_city) # Find index to use in naming
        loc_counter_per_city[target_city] += 1
        loc_name = f"loc_c{city_idx+1}_{loc_counter_per_city[target_city]}"
        potential_locations_with_city_hint.append((loc_name, target_city))

    # Shuffle the generated locations and assign to city_locations map
    random.shuffle(potential_locations_with_city_hint)
    for loc_name, city in potential_locations_with_city_hint:
         city_locations[city].append(loc_name)
         all_locations.append(loc_name) # Populate the flat list

    # --- Airport Assignment ---
    airports = []
    city_airport = {}
    city_non_airport_locations = {city: [] for city in cities} # Not strictly needed for PDDL, but useful internally

    for city, final_locs in city_locations.items():
        if not final_locs:
             # This should not happen with the new distribution logic if num_cities >= 1
             raise RuntimeError(f"City {city} has no locations assigned after distribution.")
        apt = random.choice(final_locs)
        airports.append(apt)
        city_airport[city] = apt
        for loc in final_locs:
            if loc != apt:
                 city_non_airport_locations[city].append(loc) # Populate non-airport list


    # --- Initial State Predicates (Static Objects) ---
    init_predicates = []
    # Type declarations for non-packages
    init_predicates.extend([f"(truck {t})" for t in trucks])
    init_predicates.extend([f"(airplane {a})" for a in airplanes])
    init_predicates.extend([f"(city {c})" for c in cities])
    init_predicates.extend([f"(location {l})" for l in all_locations])
    init_predicates.extend([f"(airport {a})" for a in airports])

    # Airplane initial locations
    for apn in airplanes:
        start_apt = random.choice(airports)
        init_predicates.append(f"(at {apn} {start_apt})")

    # Truck initial locations (one truck per city, starting at a random location in that city)
    for i, city in enumerate(cities):
        truck = trucks[i]
        start_loc = random.choice(city_locations[city])
        init_predicates.append(f"(at {truck} {start_loc})")

    # Location in-city predicates
    for city, final_locs in city_locations.items():
        for loc in final_locs:
            init_predicates.append(f"(in-city {loc} {city})")

    # --- Package Generation and Initial State ---
    packages = [] # Final list of package names
    package_start_locations = {} # Map package name -> start loc
    packages_at_location_counter = {loc: 0 for loc in all_locations} # Counter for naming

    for _ in range(num_packages):
        # 1. Choose random start location
        pkg_start_loc = random.choice(all_locations)

        # 2. Increment counter for this location and create unique name
        packages_at_location_counter[pkg_start_loc] += 1
        count = packages_at_location_counter[pkg_start_loc]

        # 3. Create descriptive name (e.g., pkg_loc_c1_2_1)
        pkg_name = f"pkg_{pkg_start_loc}_{count}"
        packages.append(pkg_name)

        # 4. Store start location for goal generation
        package_start_locations[pkg_name] = pkg_start_loc

        # 5. Add initial state predicates for this package
        init_predicates.append(f"(package {pkg_name})") # Type declaration
        init_predicates.append(f"(at {pkg_name} {pkg_start_loc})") # Initial location

    # --- Goal State ---
    goal_predicates = []
    # Select packages that will have goals
    goal_packages = random.sample(packages, num_goals)

    for pkg in goal_packages:
        start_loc = package_start_locations[pkg]
        # Find possible goal locations (any location different from the start)
        possible_goal_locs = [loc for loc in all_locations if loc != start_loc]

        if not possible_goal_locs:
                # This case happens if there's only one location total in the domain
                if num_locations == 1:
                    # Cannot set a different goal, skip this package's goal
                    print(f"Warning: Only one location exists. Cannot set a different goal for {pkg}. Skipping goal.")
                    continue
                else:
                    # This should ideally not happen if num_locations > 1, indicates a logic error
                    # Or perhaps an edge case where all packages start at the single non-airport in a city?
                    # Re-checking logic: possible_goal_locs is based on all_locations, so this error
                    # implies num_locations <= 1. The check above handles num_locations == 1.
                    # So this else should technically not be reachable if num_locations > 1 and num_cities >= 1.
                    # Keeping for robustness.
                    print(f"Error: Could not find a different goal location for {pkg} starting at {start_loc} "
                                        f"with {num_locations} total locations. This indicates a potential issue.")
                    # Optionally, raise an error here instead of just printing
                    # raise RuntimeError(...)
                    continue # Skip goal for this package if no valid goal location found

        # Choose a random goal location from the possible ones
        goal_loc = random.choice(possible_goal_locs)
        goal_predicates.append(f"(at {pkg} {goal_loc})")

    # --- Assemble PDDL String ---
    final_problem_id = f"{problem_id}-s{seed}" if seed is not None else problem_id

    # Add comment with command-line arguments if available
    args_comment = ""
    if args:
        # Reconstruct the command string. sys.argv[0] is the script name.
        command_string = " ".join(sys.argv[1:])
        args_comment = f"; Generated with arguments: {command_string}\n"


    pddl = f"{args_comment}(define (problem {final_problem_id})\n"
    pddl += f"\t(:domain logistics)\n"

    # Objects (Combine all object lists)
    # Use filter(None, ...) just in case any list is empty and join results in leading/trailing space
    all_objects_list = packages + trucks + airplanes + cities + all_locations
    all_objects_str = ' '.join(filter(None, all_objects_list))
    pddl += f"\t(:objects {all_objects_str})\n"

    # Initial state (one predicate per line)
    pddl += "\t(:init\n"
    for pred in init_predicates:
        pddl += f"\t\t{pred}\n"
    pddl += "\t)\n" # End :init

    # Goal state (one predicate per line)
    pddl += "\t(:goal (and\n"
    if goal_predicates:
        for pred in goal_predicates:
            pddl += f"\t\t{pred}\n"
    else:
        raise ValueError(f"Requested {num_goals} goals, but none could be generated. "
                        f"Ensure num_packages > 0 and num_locations > 1. "
                        f"Requested: \n\t- {num_cities} cities \n\t- {num_locations} locations \n\t- {num_airplanes} airplanes \n\t- {num_packages} packages, \n\t- {num_goals} goals.")

    pddl += "\t))\n" # End (and and :goal

    pddl += ")\n" # End define

    print(f"  Problem generated with: \n\t-{num_cities} cities (and trucks), \n\t-{num_locations} locations, \n\t-{num_airplanes} airplanes, \n\t-{num_packages} packages, \n\t-{num_goals} goals ({num_goals / num_packages:.2f}%).")

    return pddl

# ------------------------------------------
# Automatic Problem Generation (Batch Mode)
# ------------------------------------------

def generate_problems_batch(target_objects_list: list[int],
                             output_dir: str,
                             seed: int | None = None,
                             args: argparse.Namespace | None = None # Add args to pass to problem generator
                             ):
    """
    Generates a batch of logistics problems, one for each target object count in the list.

    Args:
        target_objects_list: A list of desired approximate total object counts.
        output_dir: The directory where PDDL files will be saved.
        seed: The random seed to use for generation (optional).
        args: The argparse.Namespace object containing the command-line arguments.
    """
    num_problems = len(target_objects_list)
    print(f"Generating {num_problems} problems for target counts {target_objects_list} in '{output_dir}'...")
    if seed is not None:
        print(f"Using random seed: {seed}")

    os.makedirs(output_dir, exist_ok=True)

    for i, target_num_objects in enumerate(target_objects_list):
        problem_index = i + 1 # 1-based index for filenames/IDs
        print(f" Generating problem {problem_index}/{num_problems} (Target Objects: {target_num_objects})...")

        if target_num_objects < 6:
             print(f"  Warning: Target objects ({target_num_objects}) is very low. May result in trivial problem or failure to generate.")

        # --- Heuristics to determine parameters ---
        # Total = C (cities) + C (trucks) + L (locations) + A (airplanes) + P (packages)
        # Assuming Trucks = Cities
        # Total = 2C + L + A + P
        avg_cities_per_airplane = random.uniform(2.5, 4.5)
        avg_locations_per_city = random.uniform(1.2, 3.0)
        goal_percentage = random.uniform(0.8, 1.0) # Percentage of packages with goals

        # Estimate C (number of cities) based on target total objects
        # Estimate non-package objects as a fraction of the total. The more objects, the larger the fraction can be packages.
        non_pkg_fraction_guess = random.uniform(max(0.3, 0.6 - target_num_objects/100), 0.65)
        # The denominator represents the number of non-package objects per city, roughly:
        # 1 city + 1 truck + avg_locations_per_city locations + (1/avg_cities_per_airplane) airplanes
        denominator = (2.0 + avg_locations_per_city + (1.0 / avg_cities_per_airplane))

        # Estimated number of cities
        estimated_c = (target_num_objects * non_pkg_fraction_guess) / denominator

        # Calculate parameters, ensuring minimums and some randomness. Proritize more cities rather than fewer. 
        num_cities = max(1, round(estimated_c * random.uniform(0.85, 1.3)))
        # Ensure at least num_cities locations, and at least 2 total locations if possible for goals
        num_locations = max(max(num_cities,2), round(num_cities * avg_locations_per_city * random.uniform(0.85, 1.15)))
        num_airplanes = max(1, round(num_cities / avg_cities_per_airplane * random.uniform(0.9, 1.1)))

        # Calculate deterministic number of packages to meet the target object count
        objects_so_far = num_cities + num_cities + num_locations + num_airplanes # Cities, Trucks, Locations, Airplanes
        num_packages = max(0, target_num_objects - objects_so_far)

        # Calculate goals: simplified calculation, ensuring goals are only attempted if possible
        num_goals = 0 # Default to 0 goals
        # Only calculate goals if there are packages and more than one location to move them to
        if num_packages > 0 and num_locations > 1:
             # Calculate target number of goals based on percentage, ensure at least 1 if possible
             # and not more than the total number of packages
             num_goals = max(1, min(num_packages, round(num_packages * goal_percentage)))
        # If num_packages is 0 or num_locations is 1, num_goals remains 0, which is handled by generate_logistics_problem

        problem_id_base = f"logistics-auto-t{target_num_objects}-i{problem_index}"

        try:
            print(f"  Parameters: C={num_cities}, L={num_locations}, Pkg={num_packages}, A={num_airplanes}, Goals={num_goals} ({goal_percentage*100:.1f})")

            pddl_content = generate_logistics_problem(
                num_packages=num_packages,
                num_cities=num_cities,
                num_locations=num_locations,
                num_airplanes=num_airplanes,
                num_goals=num_goals,
                problem_id=problem_id_base,
                seed=seed,
                args=args # Pass args to the problem generator
            )


            # Construct final filename based on user input and seed
            final_problem_id = f"{problem_id_base}-s{seed}" if seed is not None else problem_id_base
            file_name = f"{final_problem_id}.pddl"
            output_path = os.path.join(output_dir, file_name)

            # Ensure output directory exists
            if output_dir and not os.path.exists(output_dir):
                 print(f"Creating output directory: {output_dir}")
                 os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(pddl_content)
            print(f"  Successfully generated {file_name}")

        except ValueError as e:
            print(f"  Skipping problem {problem_index} due to parameter error: {e}", file=sys.stderr)
        except Exception as e:
             print(f"  Failed to generate problem {problem_index}: {e}", file=sys.stderr)

    print(f"Batch generation complete.")

# ------------------------------------------
# Command Line Interface
# ------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate PDDL Logistics Problems.")
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for deterministic generation.')

    subparsers = parser.add_subparsers(dest='mode', required=True, help='Generation mode')

    # --- Single Problem Generation ---
    parser_single = subparsers.add_parser('single', help='Generate a single problem with specific parameters.')
    parser_single.add_argument('--cities', type=int, required=True, help='Number of cities.')
    parser_single.add_argument('--locations', type=int, required=True, help='Total number of locations (must be >= cities).')
    parser_single.add_argument('--packages', type=int, required=True, help='Total number of packages.')
    parser_single.add_argument('--airplanes', type=int, required=True, help='Number of airplanes.')
    parser_single.add_argument('--goals', type=int, required=True, help='Number of packages with goal conditions.')
    parser_single.add_argument('--id', type=str, default='logistics-problem', help='Base Problem ID (name) for the PDDL file (seed will be appended).')
    parser_single.add_argument('--output_file', type=str, required=True, help='Path to save the generated PDDL file (e.g., ./problems/myprob.pddl). Seed marker is added automatically if seed is provided.')

    # --- Batch Problem Generation ---
    parser_batch = subparsers.add_parser('batch', help='Generate multiple problems based on a list of target object counts.')
    parser_batch.add_argument('--target_objects_list', type=int, nargs='+', required=True,
                              help='List of approximate total object counts for the problems to generate (e.g., 10 15 20).')
    parser_batch.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated PDDL files.')

    args = parser.parse_args()

    if args.seed is not None:
        print(f"Using global random seed: {args.seed}")
        random.seed(args.seed)

    if args.mode == 'single':
        try:
            print(f"Generating single problem '{args.id}'...")
            pddl_content = generate_logistics_problem(
                num_packages=args.packages,
                num_cities=args.cities,
                num_locations=args.locations,
                num_airplanes=args.airplanes,
                num_goals=args.goals,
                problem_id=args.id,
                seed=args.seed,
                args=args # Pass args to the problem generator
            )

            # Construct final filename based on user input and seed
            output_dir = os.path.dirname(args.output_file)
            base_filename = os.path.basename(args.output_file)
            filename_base, filename_ext = os.path.splitext(base_filename)

            # Append seed marker if seed is used and not already present (basic check)
            seed_marker = f"-s{args.seed}"
            # Check if the base filename already ends with a seed marker pattern to avoid duplicates
            # This is a basic check and might not catch all cases, but covers the standard format.
            if args.seed is not None and not filename_base.endswith(f"-s{args.seed}"):
                 final_filename_base = f"{filename_base}{seed_marker}"
            else:
                 final_filename_base = filename_base

            # Ensure extension is .pddl
            if not filename_ext:
                 filename_ext = ".pddl"
            elif filename_ext != ".pddl":
                 print(f"Warning: Output file extension '{filename_ext}' is not '.pddl'. Using '.pddl'.")
                 filename_ext = ".pddl"


            final_filename = f"{final_filename_base}{filename_ext}"
            output_path = os.path.join(output_dir, final_filename)


            if output_dir and not os.path.exists(output_dir):
                 print(f"Creating output directory: {output_dir}")
                 os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(pddl_content)
            print(f"Successfully generated problem file: {output_path}")

        except ValueError as e:
            print(f"Error generating single problem: {e}", file=sys.stderr)
            exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            exit(1)

    elif args.mode == 'batch':
        try:
            generate_problems_batch(
                target_objects_list=args.target_objects_list,
                output_dir=args.output_dir,
                seed=args.seed,
                args=args # Pass args to the batch generator
            )
        except Exception as e:
            print(f"An unexpected error occurred during batch generation: {e}", file=sys.stderr)
            exit(1)

if __name__ == "__main__":
    # Store the original command line arguments before argparse consumes them
    original_command_line_args = sys.argv[:]
    main()
