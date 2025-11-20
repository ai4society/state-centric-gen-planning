import sys
from collections import deque, defaultdict

class PDDLParser:
    """
    Parses a PDDL-like file to extract problem information relevant for a grid visiting task.
    Specifically, it extracts objects (locations), robot's starting position,
    connections between locations, and the set of 'place' locations to be visited.
    """
    def __init__(self):
        self.problem_name: str | None = None
        self.domain_name: str | None = None
        self.objects: set[str] = set()
        self.robot_start_loc: str | None = None
        self.initially_visited_raw: set[str] = set()  # Locations from (visited ...) in :init
        self.connections: defaultdict[str, list[str]] = defaultdict(list)
        self.places: set[str] = set()  # All locations designated as 'place'

    def _tokenize(self, text: str) -> list[str]:
        """Converts PDDL text into a list of tokens."""
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        return text.split()

    def _parse_list_recursive(self, tokens_list: list[str]):
        """Recursively parses a list structure from tokens."""
        lst = []
        if not tokens_list:
            raise ValueError("Unexpected EOF while expecting list elements.")

        while tokens_list and tokens_list[0] != ')':
            lst.append(self._parse_atom_or_list_recursive(tokens_list))

        if not tokens_list or tokens_list[0] != ')':
             raise ValueError("Expected ')' to close a list, but not found or EOF.")
        tokens_list.pop(0)  # Consume ')'
        return lst

    def _parse_atom_or_list_recursive(self, tokens_list: list[str]):
        """Parses the next element which can be an atom or a new list."""
        if not tokens_list:
            raise ValueError("File ended unexpectedly. Expected an atom or a list.")

        token = tokens_list.pop(0)
        if token == '(':
            return self._parse_list_recursive(tokens_list)
        elif token == ')':
            raise ValueError("Unexpected ')' found at an invalid position.")
        else:
            return token # Atom

    def parse_file(self, file_path: str):
        """Reads and parses the PDDL file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}", file=sys.stderr)
            sys.exit(1)

        tokens = self._tokenize(content)
        if not tokens:
            raise ValueError("Empty or invalid PDDL file content.")

        if tokens[0] != '(': # Basic check for S-expression start
            raise ValueError("PDDL file must start with '('.")

        parsed_sexp = self._parse_atom_or_list_recursive(tokens)

        if tokens: # Check for unconsumed tokens
           remaining_str = " ".join(tokens).strip()
           if remaining_str and not remaining_str.startswith(';'): # Allow comments at end of file
            print(f"Warning: Extra tokens left after parsing: {' '.join(tokens[:5])}...", file=sys.stderr)

        if not isinstance(parsed_sexp, list) or not parsed_sexp or parsed_sexp[0] != 'define':
            raise ValueError("Expected PDDL structure to be '(define ...)' at the top level.")

        for item in parsed_sexp[1:]:
            if not isinstance(item, list) or not item: continue

            key = item[0]
            if not isinstance(key, str): continue

            if key == 'problem' and len(item) > 1:
                self.problem_name = item[1]
            elif key == ':domain' and len(item) > 1:
                self.domain_name = item[1]
            elif key == ':objects':
                obj_list = []
                for obj_token_idx in range(1, len(item)):
                    obj_token = item[obj_token_idx]
                    if obj_token == '-':
                        break
                    obj_list.append(obj_token)
                self.objects.update(obj_list)
            elif key == ':init':
                self._parse_init(item[1:])
            elif key == ':goal':
                pass # Goal section not strictly used if targets are all 'places'

    def _parse_init(self, init_preds: list):
        """Parses the ':init' section of the PDDL."""
        for pred_item in init_preds:
            if not isinstance(pred_item, list) or not pred_item: continue

            pred_name = pred_item[0]
            args = pred_item[1:]

            if pred_name == 'at-robot':
                if len(args) == 1:
                    self.robot_start_loc = args[0]
                elif len(args) == 2:
                    self.robot_start_loc = args[1]
                else:
                    print(f"Warning: Unhandled 'at-robot' format: {pred_item}", file=sys.stderr)
            elif pred_name == 'visited':
                if args: self.initially_visited_raw.add(args[0])
            elif pred_name == 'connected':
                if len(args) == 2:
                    loc1, loc2 = args[0], args[1]
                    self.connections[loc1].append(loc2)
                else:
                    print(f"Warning: Unhandled 'connected' format: {pred_item}", file=sys.stderr)
            elif pred_name == 'place':
                if args: self.places.add(args[0])


def solve_planning_problem(pddl_data: PDDLParser) -> list[str] | None:
    """
    Finds the cheapest plan (minimum number of moves) to visit all 'place' locations.
    Uses Breadth-First Search (BFS) on a state space of (current_location, visited_places_mask).
    Returns the list of move actions, or None if no solution is found.
    """

    all_target_places = frozenset(pddl_data.places)
    if not all_target_places:
        print("Error: No 'place' locations defined in PDDL :init. Cannot determine cells to visit.", file=sys.stderr)
        return None

    if not pddl_data.robot_start_loc:
        print("Error: Robot start location not defined in PDDL :init (at-robot ...).", file=sys.stderr)
        return None

    initial_visited_targets = {loc for loc in pddl_data.initially_visited_raw if loc in all_target_places}
    if pddl_data.robot_start_loc in all_target_places:
        initial_visited_targets.add(pddl_data.robot_start_loc)

    frozen_initial_visited_targets = frozenset(initial_visited_targets)

    queue = deque()
    initial_bfs_state = (pddl_data.robot_start_loc, frozen_initial_visited_targets, [])
    queue.append(initial_bfs_state)

    visited_bfs_states = set()
    visited_bfs_states.add((pddl_data.robot_start_loc, frozen_initial_visited_targets))

    while queue:
        current_loc, current_visited_mask, current_path = queue.popleft()

        if current_visited_mask == all_target_places:
            return current_path

        for neighbor_loc in pddl_data.connections.get(current_loc, []):
            new_path = current_path + [f"(move {current_loc} {neighbor_loc})"]

            new_visited_mask_set = set(current_visited_mask)
            if neighbor_loc in all_target_places:
                new_visited_mask_set.add(neighbor_loc)

            frozen_new_visited_mask = frozenset(new_visited_mask_set)

            next_bfs_state_key = (neighbor_loc, frozen_new_visited_mask)

            if next_bfs_state_key not in visited_bfs_states:
                visited_bfs_states.add(next_bfs_state_key)
                queue.append((neighbor_loc, frozen_new_visited_mask, new_path))

    return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <pddl_file_path>", file=sys.stderr)
        sys.exit(1)

    pddl_file_path = sys.argv[1]

    parser = PDDLParser()
    try:
        parser.parse_file(pddl_file_path)
    except ValueError as e:
        print(f"Error parsing PDDL file '{pddl_file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during parsing of '{pddl_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not parser.robot_start_loc:
        print("Critical Error: Robot start location could not be determined from PDDL :init.", file=sys.stderr)
        sys.exit(1)
    if not parser.places:
        print("Critical Error: No 'place' locations defined in PDDL :init.", file=sys.stderr)
        sys.exit(1)

    plan = solve_planning_problem(parser)

    if plan is not None:
        if not plan:
            # This case (empty plan) means the goal was satisfied at the start.
            # Depending on expected output, either print nothing or a comment.
            # For this problem, an empty output for an empty plan is acceptable.
            pass
        for action_str in plan:
            print(action_str)
    else:
        # Print to stderr as it's an indication of failure to find a plan.
        print("; No solution found to visit all specified places.", file=sys.stderr)
