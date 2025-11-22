import re

import pddlpy
from pddlpy.pddl import Atom


def normalize_predicate_string(p_str_unnormalized):
    """
    Normalizes a predicate string to: "(predicate arg1 arg2)"
    Lowercases everything, ensures single spaces.
    """
    content = p_str_unnormalized.strip()

    # Handle cases where VAL outputs "handempty" without parens
    if not content.startswith("("):
        content = f"({content})"

    # Strip outer parens
    inner = content[1:-1].strip()
    if not inner:
        return "()"

    parts = inner.lower().split()
    return f"({' '.join(parts)})"


def pddlpy_atom_to_string(atom):
    """Converts pddlpy Atom object to normalized string."""
    if isinstance(atom, Atom):
        parts = atom.predicate
    elif isinstance(atom, tuple):
        parts = list(atom)
    else:
        return str(atom)

    return f"({' '.join(str(p).lower() for p in parts)})"


def parse_initial_state_regex(problem_file_path):
    """
    Fallback method: Extracts initial state using Regex if pddlpy fails.
    Finds the (:init ...) block and extracts predicates.
    """
    try:
        with open(problem_file_path, "r") as f:
            content = f.read().lower()  # Lowercase for easier matching

        # Remove comments
        content = re.sub(r";.*", "", content)

        # Find the :init block
        # This is a naive bracket counter to find the content of (:init ...)
        init_match = re.search(r"\(:init", content)
        if not init_match:
            return set()

        start_idx = init_match.end()
        balance = 1
        end_idx = start_idx

        for i, char in enumerate(content[start_idx:]):
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1

            if balance == 0:
                end_idx = start_idx + i
                break

        init_block = content[start_idx:end_idx]

        # Extract predicates: (name arg1 arg2)
        # We look for patterns starting with ( and ending with )
        predicates = set()
        matches = re.findall(r"\([^\)]+\)", init_block)
        for m in matches:
            # Clean up newlines and extra spaces
            clean_pred = " ".join(m.split())
            predicates.add(clean_pred)

        return predicates
    except Exception as e:
        print(f"Regex fallback failed for {problem_file_path}: {e}")
        return set()


def get_initial_state(domain_file, problem_file):
    """Parses PDDL to get the initial state as a set of strings."""
    # 1. Try pddlpy (The "Correct" way)
    try:
        domprob = pddlpy.DomainProblem(domain_file, problem_file)
        initial_atoms = domprob.initialstate()
        if initial_atoms:
            return {pddlpy_atom_to_string(atom) for atom in initial_atoms}
    except Exception:
        pass  # Fall through to regex

    # 2. Try Regex (The "Robust" way)
    print(f"Notice: pddlpy failed for {problem_file}, switching to regex fallback.")
    return parse_initial_state_regex(problem_file)


def parse_val_output_to_trajectory(val_output_path, initial_state_set):
    """
    Parses verbose VAL output to reconstruct the sequence of states.
    Returns a list of sets, where each set contains the predicates true in that state.
    """
    trajectory = []
    current_state = set(initial_state_set)

    # Add S0
    trajectory.append(set(current_state))

    # Regex for VAL output lines
    # Matches: "Deleting (at truck1 loc1)" or "Adding (at truck1 loc2)"
    delta_regex = re.compile(r"^(?:Deleting|Adding)\s+(.*?)\s*$", re.IGNORECASE)

    try:
        with open(val_output_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    plan_started = False

    for line in lines:
        line = line.strip()

        # Detect start of validation
        if "Plan Validation details" in line or "Checking instance" in line:
            plan_started = True

        if not plan_started:
            continue

        # Detect time step boundaries
        # "Checking next happening" implies the previous action is done and state is settled
        if "Checking next happening" in line or "Plan executed successfully" in line:
            # Only append if the state has actually changed (or it's a distinct time step)
            # We append a COPY of the current state
            if len(trajectory) == 0 or current_state != trajectory[-1]:
                trajectory.append(set(current_state))

            if "Plan executed successfully" in line:
                break

        # Apply effects
        match = delta_regex.match(line)
        if match:
            pred_str = match.group(1)
            norm_pred = normalize_predicate_string(pred_str)

            if line.lower().startswith("deleting"):
                if norm_pred in current_state:
                    current_state.remove(norm_pred)
            elif line.lower().startswith("adding"):
                current_state.add(norm_pred)

    return trajectory
