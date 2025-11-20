import os
import random


def generate_visitall_task(width, height, start_x, start_y):
    """
    Generates a PDDL Visitall task for a grid of the given width and height
    with a specified starting position.

    Args:
        width: The width of the grid (integer).
        height: The height of the grid (integer).
        start_x: The x-coordinate of the robot's starting position (integer).
        start_y: The y-coordinate of the robot's starting position (integer).

    Returns:
        A string containing the PDDL task definition.
    """
    task_name = f"grid-{width}-{height}-start-{start_x}-{start_y}" # More descriptive task name
    domain_name = "grid-visit-all" # Assuming the domain is named grid-visit-all

    # Generate objects (locations)
    objects = []
    for y in range(height):
        for x in range(width):
            objects.append(f"loc-x{x}-y{y}")
    objects_str = " ".join(objects)

    initial_position = f"loc-x{start_x}-y{start_y}"

    # Generate initial state predicates
    init_state = [f"(at-robot {initial_position})", f"(visited {initial_position})"]

    # Add connected predicates for adjacent cells (up, down, left, right)
    for y in range(height):
        for x in range(width):
            current_loc = f"loc-x{x}-y{y}"
            # Connect to the right
            if x + 1 < width:
                right_loc = f"loc-x{x+1}-y{y}"
                init_state.append(f"(connected {current_loc} {right_loc})")
                init_state.append(f"(connected {right_loc} {current_loc})")
            # Connect downwards
            if y + 1 < height:
                down_loc = f"loc-x{x}-y{y+1}"
                init_state.append(f"(connected {current_loc} {down_loc})")
                init_state.append(f"(connected {down_loc} {current_loc})")

    # Add place predicates for all locations
    for loc in objects:
        init_state.append(f"(place {loc})")

    init_state_str = "\n    ".join(init_state)

    # Generate goal state predicates (all locations visited)
    goal_state = []
    for loc in objects:
        goal_state.append(f"(visited {loc})")
    goal_state_str = "\n      ".join(goal_state)

    # Construct the full PDDL task string
    pddl_task = f"""(define (problem {task_name})
  (:domain {domain_name})
  (:objects {objects_str})
  (:init
    {init_state_str}
  )
  (:goal
    (and
      {goal_state_str}
    )
  )
)"""

    return pddl_task

if __name__ == "__main__":
    # Define the range for width and height
    min_width = 1
    max_width = 11
    min_height = 1
    max_height = 11

    widths = range(min_width, max_width + 1)
    heights = range(min_height, max_height + 1)

    # Define the output directory
    size_to_output_dir = {
        1: "train",  # 1 tasks
        2: "test-interpolation",  # 4 tasks
        3: "train",  # 6 tasks
        4: "train",  # 12 tasks
        5: "test-interpolation",  # 10 tasks
        6: "train",  # 24 tasks
        7: "train",  # 14 tasks
        8: "test-interpolation",  # 32 tasks
        9: "test-interpolation",  # 27 tasks
        10: "train",  # 40 tasks
        11: "train",  # 22 tasks
        12: "train",  # 48 tasks
        14: "train",  # 28 tasks
        15: "test-interpolation",  # 30 tasks
        16: "train",  # 48 tasks
        18: "validation",  # 72 tasks
        20: "validation",  # 80 tasks
        21: "test-extrapolation",  # 42 tasks
        22: "test-extrapolation",  # 44 tasks
        24: "test-extrapolation",  # 96 tasks
        25: "test-extrapolation",  # 25 tasks
        27: "test-extrapolation",  # 54 tasks
        28: "test-extrapolation",  # 56 tasks
        30: "test-extrapolation",  # 120 tasks
        32: "test-extrapolation",  # 64 tasks
        33: "test-extrapolation",  # 66 tasks
        35: "test-extrapolation",  # 70 tasks
        36: "test-extrapolation",  # 108 tasks
        40: "test-extrapolation",  # 160 tasks
        42: "test-extrapolation",  # 84 tasks
        44: "test-extrapolation",  # 88 tasks
        45: "test-extrapolation",  # 90 tasks
        48: "test-extrapolation",  # 96 tasks
        49: "test-extrapolation",  # 49 tasks
        50: "test-extrapolation",  # 100 tasks
        54: "test-extrapolation",  # 108 tasks
        55: "test-extrapolation",  # 110 tasks
        56: "test-extrapolation",  # 112 tasks
        60: "test-extrapolation",  # 120 tasks
        63: "test-extrapolation",  # 126 tasks
        64: "test-extrapolation",  # 64 tasks
        66: "test-extrapolation",  # 132 tasks
        70: "test-extrapolation",  # 140 tasks
        72: "test-extrapolation",  # 144 tasks
        77: "test-extrapolation",  # 154 tasks
        80: "test-extrapolation",  # 160 tasks
        81: "test-extrapolation",  # 81 tasks
        88: "test-extrapolation",  # 176 tasks
        90: "test-extrapolation",  # 180 tasks
        99: "test-extrapolation",  # 198 tasks
        100: "test-extrapolation",  # 100 tasks
        110: "test-extrapolation",  # 220 tasks
        121: "test-extrapolation",  # 121 tasks
    }

    print(f"Generating tasks for grids {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)} with all possible start positions...")

    # Iterate through all combinations of width and height
    for width in widths:
        for height in heights:
            i = 1

            # Determine the output directory based on the grid size
            output_dir = size_to_output_dir[width * height]
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Sample 10 start positions for each grid size
            cells = [(x, y) for x in range(width) for y in range(height)]
            selected_cells = random.sample(cells, min(10, len(cells)))

            for start_x, start_y in selected_cells:
                # Define the output filename using the actual start coordinates
                output_filename = os.path.join(output_dir, f"w{width:02}h{height:02}-{i:02}.pddl")

                print(f"  Generating task: {output_filename} (width={width}, height={height}, start={start_x},{start_y})")

                # Removed try...except blocks as requested

                # Generate the PDDL content using the actual start coordinates
                pddl_content = generate_visitall_task(width, height, start_x=start_x, start_y=start_y)

                # Write the content to the file
                with open(output_filename, "w") as f:
                    f.write(pddl_content)

                i += 1

    print("\nFinished generating all tasks.")
