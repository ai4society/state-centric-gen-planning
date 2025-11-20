import argparse
import sys
import re

def read_and_preprocess(pddl_file_path):
    """Reads the PDDL file, removes comments, and standardizes whitespace/parentheses."""
    try:
        with open(pddl_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {pddl_file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {pddl_file_path}: {e}")
        return None

    # Remove comments (lines starting with ;)
    lines = content.splitlines()
    content_without_comments = "\n".join(line for line in lines if not line.strip().startswith(';'))

    # Standardize whitespace around parentheses
    content = content_without_comments.replace('(', ' ( ').replace(')', ' ) ')

    # Replace multiple whitespace characters with a single space
    content = re.sub(r'\s+', ' ', content).strip()

    return content

def tokenize(text):
    """Splits the preprocessed text into tokens."""
    return text.split()

def parse_expression(tokens, index):
    """
    Recursively parses a PDDL expression from a list of tokens.
    Returns the parsed expression (as a tuple or atom) and the next index in tokens.
    """
    if index >= len(tokens):
        raise ValueError("Unexpected end of tokens while parsing.")

    token = tokens[index]

    if token == '(':
        # Start of a list/expression
        expression_list = []
        index += 1 # Consume '('
        while index < len(tokens) and tokens[index] != ')':
            sub_expression, index = parse_expression(tokens, index)
            expression_list.append(sub_expression)
        if index >= len(tokens) or tokens[index] != ')':
            raise ValueError(f"Mismatched parentheses: Expected ')' at index {index}")
        index += 1 # Consume ')'
        # Convert the list to a tuple for immutability (represents a predicate or structure)
        return tuple(expression_list), index
    elif token == ')':
        # Unexpected end of a list
        raise ValueError(f"Unexpected ')' at index {index}")
    else:
        # It's an atom (symbol, keyword, number)
        return token, index + 1

def parse_pddl_content(text):
    """Parses the preprocessed PDDL text into a nested structure."""
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("No tokens found after preprocessing.")

    # A PDDL file is expected to have a single top-level define block
    try:
        parsed_structure, final_index = parse_expression(tokens, 0)
    except ValueError as e:
        print(f"Parsing error: {e}")
        return None

    if final_index != len(tokens):
         # This indicates there might be unparsed tokens at the end
         print(f"Warning: Not all tokens were consumed during parsing. Remaining: {tokens[final_index:]}")

    return parsed_structure

def extract_problem_data_from_structure(parsed_structure):
    """Extracts relevant problem data from the parsed nested structure."""
    data = {
        'problem_name': None,
        'domain_name': None,
        'objects': [], # List of object names
        'init': [],    # List of predicate tuples (e.g., ('at', 'obj1', 'locA'))
        'goal': None   # Goal structure tuple
    }

    if not isinstance(parsed_structure, tuple) or len(parsed_structure) == 0 or parsed_structure[0] != 'define':
        print("Error: Top-level structure is not a 'define' block.")
        return None

    # The structure should be (define (problem problem-name) ...sections...)
    if len(parsed_structure) < 2 or not isinstance(parsed_structure[1], tuple) or len(parsed_structure[1]) < 2 or parsed_structure[1][0] != 'problem':
         print("Error: Second element is not a '(problem ...)' block.")
         return None

    data['problem_name'] = parsed_structure[1][1]

    # Iterate through the sections within the define block
    # We expect sections like (:domain ...), (:objects ...), (:init ...), (:goal ...)
    i = 2 # Start after the problem name tuple
    while i < len(parsed_structure):
        section = parsed_structure[i]
        if isinstance(section, tuple) and len(section) > 0 and section[0].startswith(':'):
            section_keyword = section[0]
            if section_keyword == ':domain' and len(section) > 1:
                data['domain_name'] = section[1]
            elif section_keyword == ':objects':
                 # Objects are listed as parameters of the :objects tuple
                data['objects'] = list(section[1:])
            elif section_keyword == ':init':
                 # Init predicates are parameters of the :init tuple
                data['init'] = list(section[1:])
            elif section_keyword == ':goal' and len(section) > 1:
                 # The goal structure is the single parameter of the :goal tuple
                data['goal'] = section[1]
            # Add handling for other sections like :requirements if needed
        else:
            # This might be unexpected structure within the define block
            print(f"Warning: Skipping unexpected element in define block: {section}")

        i += 1

    return data


def process_initial_state(extracted_data):
    """Processes the extracted data to structure it for visualization."""
    initial_state_data = {
        'cities': {},
        'locations': {}, # {loc_name: {type: 'Airport'|'Location', city: 'city_name', contents: {trucks: [], packages: [], airplanes: []}}}
        'objects': {}, # {obj_name: type} # Store actual object names here
        'goal': []
    }

    # 1. Identify object types
    # Iterate through init predicates to find objects declared with their type predicates
    for predicate in extracted_data.get('init', []):
        if isinstance(predicate, tuple) and len(predicate) > 1:
            pred_name = predicate[0]
            obj_name = predicate[1]
            if pred_name in ['truck', 'airplane', 'package', 'city', 'location', 'airport']:
                 initial_state_data['objects'][obj_name] = pred_name # Store the type

    # Fallback: If objects weren't fully typed in init, use the :objects list
    if not initial_state_data['objects'] and extracted_data.get('objects'):
         print("Warning: Object types not explicitly defined in :init.")
         for obj_name in extracted_data['objects']:
             # Just add them as objects without inferred type if not specified
             if obj_name not in initial_state_data['objects']:
                initial_state_data['objects'][obj_name] = 'unknown'


    # 2. Identify cities and locations within them, and location types
    for predicate in extracted_data.get('init', []):
         if isinstance(predicate, tuple) and len(predicate) > 1:
            pred_name = predicate[0]
            if pred_name == 'city' and len(predicate) > 1:
                 city_name = predicate[1]
                 initial_state_data['cities'][city_name] = [] # List of locations in this city
            elif pred_name == 'in-city' and len(predicate) > 2:
                loc_name = predicate[1]
                city_name = predicate[2]
                if city_name in initial_state_data['cities']:
                    initial_state_data['cities'][city_name].append(loc_name)
                if loc_name not in initial_state_data['locations']:
                     initial_state_data['locations'][loc_name] = {'type': 'Location', 'city': city_name, 'contents': {'trucks': [], 'packages': [], 'airplanes': []}}
                initial_state_data['locations'][loc_name]['city'] = city_name # Ensure city is recorded
            elif pred_name == 'airport' and len(predicate) > 1:
                loc_name = predicate[1]
                if loc_name not in initial_state_data['locations']:
                     # Location wasn't in in-city yet, add placeholder
                     initial_state_data['locations'][loc_name] = {'type': 'Airport', 'city': None, 'contents': {'trucks': [], 'packages': [], 'airplanes': []}}
                else:
                     initial_state_data['locations'][loc_name]['type'] = 'Airport'

    # Ensure all locations mentioned in 'at' predicates are initialized, even if no 'in-city'/'airport' predicate seen
    for predicate in extracted_data.get('init', []):
         if isinstance(predicate, tuple) and len(predicate) > 2 and predicate[0] == 'at':
             obj_name = predicate[1]
             loc_name = predicate[2]
             if loc_name not in initial_state_data['locations']:
                  initial_state_data['locations'][loc_name] = {'type': 'Location', 'city': None, 'contents': {'trucks': [], 'packages': [], 'airplanes': []}}


    # Sort locations within cities for consistent visualization
    for city in initial_state_data['cities']:
        initial_state_data['cities'][city].sort() # Alphabetical sorting of locations


    # 3. Populate contents at locations using real object names
    for predicate in extracted_data.get('init', []):
        if isinstance(predicate, tuple) and len(predicate) > 2 and predicate[0] == 'at':
            obj_name = predicate[1]
            loc_name = predicate[2]
            obj_type = initial_state_data['objects'].get(obj_name, 'unknown')

            if loc_name in initial_state_data['locations']:
                if obj_type == 'truck':
                    initial_state_data['locations'][loc_name]['contents']['trucks'].append(obj_name)
                elif obj_type == 'airplane':
                    initial_state_data['locations'][loc_name]['contents']['airplanes'].append(obj_name)
                elif obj_type == 'package':
                    initial_state_data['locations'][loc_name]['contents']['packages'].append(obj_name)
                else:
                    # For unknown types, just add to a general list or pick a category
                    # Let's add them to packages for simplicity in this limited visualization
                    initial_state_data['locations'][loc_name]['contents']['packages'].append(obj_name)


    # Sort contents within locations for consistent visualization
    for loc_data in initial_state_data['locations'].values():
        loc_data['contents']['trucks'].sort()
        loc_data['contents']['packages'].sort()
        loc_data['contents']['airplanes'].sort()


    # 4. Extract Goal State using real object names
    goal_structure = extracted_data.get('goal')
    if goal_structure:
        # Assuming goal is (and (at ...) (at ...)) or just (at ...)
        if isinstance(goal_structure, tuple):
            if goal_structure[0] == 'and':
                for goal_pred in goal_structure[1:]:
                    if isinstance(goal_pred, tuple) and len(goal_pred) > 2 and goal_pred[0] == 'at':
                        obj, loc = goal_pred[1], goal_pred[2]
                        # Use the real object name directly
                        initial_state_data['goal'].append(('at', obj, loc)) # Store as tuple ('at', obj_name, loc)
                    else:
                         initial_state_data['goal'].append(str(goal_pred)) # Add unsupported goals as strings
            elif goal_structure[0] == 'at' and len(goal_structure) > 2:
                 obj, loc = goal_structure[1], goal_structure[2]
                 # Use the real object name directly
                 initial_state_data['goal'].append(('at', obj, loc)) # Store as tuple ('at', obj_name, loc)
            else:
                 initial_state_data['goal'].append(str(goal_structure)) # Add unsupported top-level goal as string
        else:
            initial_state_data['goal'].append(str(goal_structure)) # Add non-tuple goal as string


    # No object_symbols needed in the output data structure anymore
    # initial_state_data['object_symbols'] = object_symbols # Removed

    return initial_state_data

def generate_ascii_art(data):
    """Generates ASCII art visualization from processed initial state data using real object names."""

    cities = sorted(data['cities'].keys()) # Sort cities alphabetically
    if not cities:
        # Attempt to visualize locations without city info if cities weren't defined
        if data['locations']:
             print("Warning: No cities found, attempting to visualize locations without city boxes.")
             # A simpler visualization without city boxes would be needed here
             # For now, we'll return a message indicating this limitation
             return "Could not visualize: No cities defined in PDDL problem."
        return "No cities or locations found in the PDDL problem."

    # Prepare lines for each city block
    city_lines = {city: [] for city in cities}
    header_height = 3 # For top border, city name, and separator line

    # Build content lines for each city
    for city in cities:
        lines = []
        # Add location information
        locations_in_city = data['cities'][city]
        if not locations_in_city:
             lines.append(" No locations in city") # No border/padding yet
        else:
            for loc in locations_in_city:
                loc_data = data['locations'].get(loc) # Use .get in case of inconsistent data
                if not loc_data: continue # Skip if location data is missing

                loc_line = f"  [{loc}] ({loc_data['type']})"
                lines.append(loc_line)

                contents = loc_data['contents']
                # Use real object names directly from the contents lists
                if contents['trucks']:
                    lines.append(f"    Trucks: {' '.join(contents['trucks'])}")
                if contents['airplanes']:
                    lines.append(f"    Airplanes: {' '.join(contents['airplanes'])}")
                if contents['packages']:
                    lines.append(f"    Packages: {' '.join(contents['packages'])}")

                if not any(contents.values()):
                    lines.append("    (Empty)")

                lines.append("") # Blank line after each location block

        city_lines[city] = lines


    # Calculate required width for city blocks
    # Find the widest line in any city block's content
    max_content_width = 0
    for city in cities:
        for line in city_lines[city]:
            max_content_width = max(max_content_width, len(line))

    # Add padding for borders and internal spacing
    internal_padding = 4 # e.g., "  [" and " ] " spacing
    border_width = 2 # '|' and ' |'
    min_city_name_width = len(max(cities, key=len)) if cities else 0

    # City name width needs to be at least as wide as the name, centered
    # Also needs to accommodate the widest content line plus padding/borders
    city_block_width = max(max_content_width + internal_padding + border_width,
                           min_city_name_width + border_width + 2) # +2 for space around name

    # Re-pad lines and add borders
    final_city_lines = {city: [] for city in cities}
    max_content_lines = 0
    for city in cities:
        # Add top border
        final_city_lines[city].append("+" + "-" * (city_block_width - 2) + "+")
        # Add city name line
        final_city_lines[city].append(f"| {city.center(city_block_width - 4)} |")
        # Add separator line
        final_city_lines[city].append("|" + "-" * (city_block_width - 2) + "|")

        # Add padded content lines
        for line in city_lines[city]:
             padded_line = f"| {line.ljust(city_block_width - 4)} |"
             final_city_lines[city].append(padded_line)

        max_content_lines = max(max_content_lines, len(city_lines[city]))

    # Add blank lines to make all city content blocks the same height before adding bottom border
    for city in cities:
         current_content_height = len(final_city_lines[city]) - header_height
         while current_content_height < max_content_lines:
              final_city_lines[city].append(f"|{' ' * (city_block_width - 4)}|")
              current_content_height += 1

         # Add bottom border
         final_city_lines[city].append("+" + "-" * (city_block_width - 2) + "+")


    # Combine city blocks side-by-side
    combined_art = []
    separator = "     " # Space between cities

    # Combine lines from all city blocks
    if cities:
        max_lines_total = max(len(lines) for lines in final_city_lines.values())
        for i in range(max_lines_total):
            current_line_parts = []
            for city in cities:
                if i < len(final_city_lines[city]):
                    current_line_parts.append(final_city_lines[city][i])
                else:
                    # Should not happen if padding works, but as fallback
                    current_line_parts.append(" " * city_block_width)
            combined_art.append(separator.join(current_line_parts))


    # Add Goal State using real object names
    combined_art.append("\nGoal State:")
    if data['goal']:
        for item in data['goal']:
            # Item is now a tuple ('at', obj_name, loc) or a raw string
            if isinstance(item, tuple) and len(item) > 2 and item[0] == 'at':
                 obj_name, loc = item[1], item[2]
                 combined_art.append(f"  - ({obj_name} at [{loc}])")
            else:
                 # Fallback for unsupported goal structures
                 combined_art.append(f"  - {str(item)}")
    else:
        combined_art.append("  No specific goal defined or goal not in supported format.")


    return "\n".join(combined_art)


def main():
    argparser = argparse.ArgumentParser(description="Visualize the initial state of a PDDL logistics problem using a custom parser.")
    argparser.add_argument("pddl_file", help="Path to the PDDL problem file.")

    args = argparser.parse_args()

    preprocessed_content = read_and_preprocess(args.pddl_file)
    if preprocessed_content is None:
        return # Error already printed

    parsed_structure = parse_pddl_content(preprocessed_content)
    if parsed_structure is None:
        return # Error already printed

    extracted_data = extract_problem_data_from_structure(parsed_structure)
    if extracted_data is None:
        return # Error already printed

    initial_state_data = process_initial_state(extracted_data)

    ascii_art = generate_ascii_art(initial_state_data)
    print(ascii_art)

if __name__ == "__main__":
    main()
