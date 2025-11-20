import argparse
import random
import sys
import time

def generate_locations(num_cities, city_size, num_packages, num_airplanes, num_trucks):
    """Generates random initial and goal locations for objects."""
    truck_locs = []  # List of (city_index, location_index_in_city)
    package_origins = [] # List of (city_index, location_index_in_city)
    package_destins = [] # List of (city_index, location_index_in_city)
    airplane_locs = []   # List of city_index (airplanes are always at location 0 within the city)

    # Generate truck locations
    # The C code puts the first 'num_cities' trucks one in each city, then others randomly.
    # Let's replicate that.
    for i in range(num_cities):
        loc_in_city = random.randrange(city_size)
        truck_locs.append((i, loc_in_city))
    for i in range(num_cities, num_trucks):
        city_index = random.randrange(num_cities)
        loc_in_city = random.randrange(city_size)
        truck_locs.append((city_index, loc_in_city))

    # Generate package origin and destination locations
    # Note: The C code generates ALL package destinations here,
    #       and we will select a subset for the goal later.
    for i in range(num_packages):
        origin_city = random.randrange(num_cities)
        origin_loc = random.randrange(city_size)
        dest_city = random.randrange(num_cities)
        dest_loc = random.randrange(city_size)
        package_origins.append((origin_city, origin_loc))
        package_destins.append((dest_city, dest_loc)) # Store all destinations

    # Generate airplane locations (always at the airport, location 0)
    for i in range(num_airplanes):
        city_index = random.randrange(num_cities)
        airplane_locs.append(city_index)

    return {
        'truck_locs': truck_locs,
        'package_origins': package_origins,
        'package_destins': package_destins, # All destinations are stored
        'airplane_locs': airplane_locs
    }

def print_pddl_problem(num_cities, city_size, num_packages, num_airplanes, num_trucks, locations_data, num_goal_atoms):
    """Prints the generated PDDL problem file to stdout with lowercase keywords."""
    print(f"(define (problem logistics-c{num_cities}-s{city_size}-p{num_packages}-a{num_airplanes})")
    print("  (:domain logistics)") # Lowercase domain keyword

    # Print objects
    print("  (:objects") # Lowercase objects keyword
    print("           ", end=" ")
    for i in range(num_airplanes):
        print(f"a{i}", end=" ")
    print()

    print("           ", end=" ")
    for i in range(num_cities):
        print(f"c{i}", end=" ")
    print()

    print("           ", end=" ")
    for i in range(num_trucks):
        print(f"t{i}", end=" ")
    print()

    print("           ", end=" ")
    for city_idx in range(num_cities):
        for loc_idx in range(city_size):
            print(f"l{city_idx}-{loc_idx}", end=" ")
    print()

    print("           ", end=" ")
    for i in range(num_packages):
        print(f"p{i}", end=" ")
    print()
    print("  )") # End objects

    # Print initial state
    print("  (:init") # Lowercase init keyword

    # Static predicates (lowercase)
    for i in range(num_airplanes):
        print(f"    (airplane a{i})")
    for i in range(num_cities):
        print(f"    (city c{i})")
    for i in range(num_trucks):
        print(f"    (truck t{i})")
    for city_idx in range(num_cities):
        for loc_idx in range(city_size):
            print(f"    (location l{city_idx}-{loc_idx})")
            print(f"    (in-city l{city_idx}-{loc_idx} c{city_idx})")
    for city_idx in range(num_cities):
        print(f"    (airport l{city_idx}-0)") # Location 0 is always the airport
    for i in range(num_packages):
        print(f"    (package p{i})")

    # Initial object locations (from generated data - lowercase predicate)
    for i, (city, loc) in enumerate(locations_data['truck_locs']):
        print(f"    (at t{i} l{city}-{loc})")
    for i, (city, loc) in enumerate(locations_data['package_origins']):
        print(f"    (at p{i} l{city}-{loc})")
    for i, city in enumerate(locations_data['airplane_locs']):
         print(f"    (at a{i} l{city}-0)") # Airplanes start at the airport of their random city

    print("  )") # End init

    # Print goal state
    print("  (:goal") # Lowercase goal keyword
    print("    (and") # Lowercase and keyword

    # Select a random subset of package destinations for the goal
    # Each item in package_destins is (city, loc) for package i (index matches)
    # We need to select num_goal_atoms *indices* or *items* from package_destins
    if num_goal_atoms > 0 and num_packages > 0:
        # Create a list of (package_index, (dest_city, dest_loc)) tuples
        package_dest_items = list(enumerate(locations_data['package_destins']))
        # Randomly sample 'num_goal_atoms' such items
        selected_goals = random.sample(package_dest_items, num_goal_atoms)

        # Print the selected goal locations
        for pkg_idx, (city, loc) in selected_goals:
            print(f"        (at p{pkg_idx} l{city}-{loc})")
    elif num_goal_atoms == 0 and num_packages > 0:
         # If 0 goal atoms requested but there are packages, the goal is (and) which is trivially true.
         # No goal atoms are printed inside the (and).
         pass # Do nothing
    elif num_packages == 0 and num_goal_atoms > 0:
         # This case is technically invalid because packages are required (-p >= 1),
         # but good to consider. With no packages, there can be no package goals.
         # The validation in main should prevent this if num_goal_atoms > 0.
         pass # Do nothing
    # If num_packages is 0 and num_goal_atoms is 0 (or default), also do nothing.


    print("    )") # End and
    print("  )") # End goal

    print(")") # End define

def main():
    parser = argparse.ArgumentParser(
        description="Generate random PDDL logistics problems.",
        # Use RawTextHelpFormatter to preserve newlines in help messages
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-a', '--airplanes', type=int, required=True,
        help='number of airplanes (minimal 0)'
    )
    parser.add_argument(
        '-c', '--cities', type=int, required=True,
        help='number of cities (minimal 1)'
    )
    parser.add_argument(
        '-s', '--city-size', type=int, required=True,
        help='city size (minimal 1)'
    )
    parser.add_argument(
        '-p', '--packages', type=int, required=True,
        help='number of packages (minimal 1)'
    )
    parser.add_argument(
        '-t', '--trucks', type=int,
        help='number of trucks (optional, default: same as number of cities;\n'
             'there will be at least one truck per city)'
    )
    parser.add_argument(
        '-r', '--random-seed', type=int,
        help='random seed (minimal 1, optional)'
    )
    parser.add_argument(
        '-g', '--goal-atoms', type=int,
        help='number of goal atoms (optional, default: same as number of packages;\n'
             'must be between 0 and number of packages inclusive)'
    )

    args = parser.parse_args()

    # Validate base arguments
    if args.cities < 1 or args.city_size < 1 or args.packages < 1:
        print("Error: Number of cities (--cities/-c), city size (--city-size/-s), and number of packages (--packages/-p) must be at least 1.", file=sys.stderr)
        sys.exit(1)

    # Default trucks to number of cities if not specified
    num_trucks = args.trucks if args.trucks is not None else args.cities

    # Validate truck count minimum
    if num_trucks < args.cities:
         print(f"Error: Number of trucks ({num_trucks}) must be at least the number of cities ({args.cities}).", file=sys.stderr)
         sys.exit(1)

    # Default goal atoms to number of packages if not specified
    num_goal_atoms = args.goal_atoms if args.goal_atoms is not None else args.packages

    # Validate goal atom count
    if not (0 <= num_goal_atoms <= args.packages):
        print(f"Error: Number of goal atoms ({num_goal_atoms}) must be between 0 and the number of packages ({args.packages}) inclusive.", file=sys.stderr)
        sys.exit(1)
    # Although packages >= 1, allowing num_goal_atoms = 0 is harmless (empty goal)

    # Set random seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
    else:
        random.seed(int(time.time()))

    # Generate random locations
    locations_data = generate_locations(
        args.cities, args.city_size, args.packages, args.airplanes, num_trucks
    )

    # Print the PDDL problem
    print_pddl_problem(
        args.cities, args.city_size, args.packages, args.airplanes, num_trucks, locations_data, num_goal_atoms
    )

if __name__ == "__main__":
    main()
