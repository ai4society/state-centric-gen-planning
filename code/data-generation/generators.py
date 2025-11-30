import random


def generate_problem_blocks(num_blocks, problem_name):
    blocks = [f"b{i + 1}" for i in range(num_blocks)]

    def generate_state(objs):
        shuffled = objs[:]
        random.shuffle(shuffled)
        towers = []
        current_tower = []
        for b in shuffled:
            # 30% chance to break tower, or if it's the first block
            if not current_tower or random.random() < 0.3:
                if current_tower:
                    towers.append(current_tower)
                current_tower = []
            current_tower.append(b)
        if current_tower:
            towers.append(current_tower)

        preds = ["(handempty)"]
        for t in towers:
            preds.append(f"(ontable {t[0]})")
            preds.append(f"(clear {t[-1]})")
            for i in range(len(t) - 1):
                preds.append(f"(on {t[i + 1]} {t[i]})")
        return preds

    init_preds = generate_state(blocks)
    # Ensure goal is different by reshuffling until different
    goal_preds = generate_state(blocks)

    # Simple PDDL construction
    pddl = f"(define (problem {problem_name})\n"
    pddl += "  (:domain blocks)\n"
    pddl += f"  (:objects {' '.join(blocks)})\n"
    pddl += "  (:init\n    " + "\n    ".join(init_preds) + "\n  )\n"
    pddl += "  (:goal (and\n    " + "\n    ".join(goal_preds) + "\n  ))\n)"
    return pddl


def generate_problem_gripper(num_balls, problem_name):
    balls = [f"ball{i + 1}" for i in range(num_balls)]

    # Init: All balls in rooma
    init_preds = [
        "(room rooma)",
        "(room roomb)",
        "(gripper left)",
        "(gripper right)",
        "(at-robby rooma)",
        "(free left)",
        "(free right)",
    ]
    for b in balls:
        init_preds.append(f"(ball {b})")
        init_preds.append(f"(at {b} rooma)")

    # Goal: All balls in roomb
    goal_preds = [f"(at {b} roomb)" for b in balls]

    pddl = f"(define (problem {problem_name})\n"
    pddl += "  (:domain gripper-strips)\n"
    pddl += f"  (:objects rooma roomb left right {' '.join(balls)})\n"
    pddl += "  (:init\n    " + "\n    ".join(init_preds) + "\n  )\n"
    pddl += "  (:goal (and\n    " + "\n    ".join(goal_preds) + "\n  ))\n)"
    return pddl


def generate_problem_logistics(
    num_packages, num_cities, num_airplanes, num_locations_per_city, problem_name
):
    # Objects
    cities = [f"c{i + 1}" for i in range(num_cities)]
    trucks = [f"t{i + 1}" for i in range(num_cities)]  # 1 truck per city
    airplanes = [f"apn{i + 1}" for i in range(num_airplanes)]
    packages = [f"p{i + 1}" for i in range(num_packages)]

    locations = []
    airports = []

    init_preds = []

    # Setup Cities and Locations
    for i, city in enumerate(cities):
        init_preds.append(f"(city {city})")
        t = trucks[i]
        init_preds.append(f"(truck {t})")
        
        # Generate locations for this city
        city_locs = []
        for j in range(num_locations_per_city):
            loc_name = f"l-{city}-{j + 1}"
            city_locs.append(loc_name)
            locations.append(loc_name)
            init_preds.append(f"(location {loc_name})")
            init_preds.append(f"(in-city {loc_name} {city})")

        # First location is airport
        airport = city_locs[0]
        airports.append(airport)
        init_preds.append(f"(airport {airport})")

        # Place Truck at random location in city
        t_loc = random.choice(city_locs)
        init_preds.append(f"(at {t} {t_loc})")

    # Setup Airplanes (at random airports)
    for apn in airplanes:
        init_preds.append(f"(airplane {apn})")
        start = random.choice(airports)
        init_preds.append(f"(at {apn} {start})")

    # Setup Packages
    pkg_locs = {}
    for p in packages:
        init_preds.append(f"(package {p})")
        # Random start location
        start = random.choice(locations)
        pkg_locs[p] = start
        init_preds.append(f"(at {p} {start})")

    # Goals (Move package to a different location)
    goal_preds = []
    for p in packages:
        start = pkg_locs[p]
        # Pick a destination that is NOT the start
        possible = [l for l in locations if l != start]
        if possible:
            dest = random.choice(possible)
            goal_preds.append(f"(at {p} {dest})")

    all_objs = cities + trucks + airplanes + packages + locations

    pddl = f"(define (problem {problem_name})\n"
    pddl += "  (:domain logistics)\n"
    pddl += f"  (:objects {' '.join(all_objs)})\n"
    pddl += "  (:init\n    " + "\n    ".join(init_preds) + "\n  )\n"
    pddl += "  (:goal (and\n    " + "\n    ".join(goal_preds) + "\n  ))\n)"
    return pddl


def generate_problem_visitall(width, height, problem_name):
    # Generate objects
    locs = [f"loc-x{x}-y{y}" for x in range(width) for y in range(height)]

    # Random start
    start_x = random.randint(0, width - 1)
    start_y = random.randint(0, height - 1)
    start_loc = f"loc-x{start_x}-y{start_y}"

    init_preds = [f"(at-robot {start_loc})", f"(visited {start_loc})"]

    # Connectivity
    for x in range(width):
        for y in range(height):
            curr = f"loc-x{x}-y{y}"
            init_preds.append(f"(place {curr})")

            # Right
            if x + 1 < width:
                right = f"loc-x{x + 1}-y{y}"
                init_preds.append(f"(connected {curr} {right})")
                init_preds.append(f"(connected {right} {curr})")
            # Down
            if y + 1 < height:
                down = f"loc-x{x}-y{y + 1}"
                init_preds.append(f"(connected {curr} {down})")
                init_preds.append(f"(connected {down} {curr})")

    # Goal: Visit all
    goal_preds = [f"(visited {l})" for l in locs]

    pddl = f"(define (problem {problem_name})\n"
    pddl += "  (:domain grid-visit-all)\n"
    pddl += f"  (:objects {' '.join(locs)})\n"
    pddl += "  (:init\n    " + "\n    ".join(init_preds) + "\n  )\n"
    pddl += "  (:goal (and\n    " + "\n    ".join(goal_preds) + "\n  ))\n)"
    return pddl
