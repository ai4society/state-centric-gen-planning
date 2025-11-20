import argparse
import re
import sys

def parse_pddl(file_path):
    """
    Parses a PDDL file to count total objects and objects per type.

    Args:
        file_path (str): The path to the PDDL file.

    Returns:
        tuple: A tuple containing the total number of objects and a dictionary
               with counts of objects per type.
    """
    total_objects = 0
    objects_by_type = {}
    objects_list = [] # Initialize objects_list

    try:
        lines = []
        with open(file_path, 'r') as f:
            for line in f:
                if ":goal" in line:
                    # Stop reading lines after the goal section
                    break
                lines.append(line)
        content = ''.join(lines)

        # Parse objects section
        # Regex to match (:objects ...)
        objects_match = re.search(r'\(:objects\s+([^)]+)\)', content, re.DOTALL)
        if objects_match:
            objects_string = objects_match.group(1)
            # Split objects string by whitespace and filter out empty strings
            objects_list = objects_string.split()
            total_objects = len(objects_list)

        # Parse single-argument predicates for object types
        # This assumes single-argument predicates like (type object_name)
        # and captures the predicate name (type) and the object name (object_name)
        type_predicate_matches = re.findall(r'\((\w+)\s+([\w-]+)\)', content)
        for pred_name, obj_name in type_predicate_matches:
             # Simple heuristic: if the predicate is not a common PDDL keyword
             # and the object was found in the objects list, consider it a type predicate
             if pred_name not in ['define', 'domain', 'problem', ':', 'objects',
                                  'init', 'goal', 'action', 'parameters', 'precondition',
                                  'effect', 'types', 'constants', 'predicates', 'functions'] and obj_name in objects_list:
                 objects_by_type[pred_name] = objects_by_type.get(pred_name, 0) + 1


    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

    return total_objects, objects_by_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a PDDL file to count objects and objects by type.")
    parser.add_argument("pddl_file", help="The path to the PDDL file.")

    args = parser.parse_args()

    total, objects_by_type = parse_pddl(args.pddl_file)

    if total is not None:
        print(f"Total number of objects: {total}")
        sys.exit()
        print("Number of objects per type:")
        if objects_by_type:
            for obj_type, count in objects_by_type.items():
                print(f"  {obj_type}: {count}")
        else:
            print("  No single-argument type predicates found or objects were not found in the objects list.")
