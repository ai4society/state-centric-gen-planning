import glob
import os
import re

import pddlpy
from tqdm import tqdm
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import init_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem

# Regex to parse "(on a b)" -> "on a b"
PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


class WLEncoder:
    def __init__(self, domain_pddl_path, iterations=2):
        self.domain_pddl_path = domain_pddl_path
        self.iterations = iterations
        self.wlplan_domain = parse_domain(domain_pddl_path)
        self.generator = init_feature_generator(
            feature_algorithm="wl",
            domain=self.wlplan_domain,
            graph_representation="ilg",
            iterations=self.iterations,
            pruning="none",
            multiset_hash=True,
        )
        self.name_to_predicate = {p.name: p for p in self.wlplan_domain.predicates}
        self.is_collected = False

    def parse_state_string_to_wl_state(self, state_str_or_list):
        """
        Converts a list of predicate strings ["(on a b)", "(clear a)"]
        or a raw string to a wlplan.State object.
        """
        if isinstance(state_str_or_list, str):
            atom_strings = PREDICATE_REGEX.findall(state_str_or_list)
        else:
            # Assume list of strings like "(on a b)"
            atom_strings = []
            for s in state_str_or_list:
                # clean parens if present
                clean = s.replace("(", "").replace(")", "")
                atom_strings.append(clean)

        atoms = []
        for atom_str in atom_strings:
            parts = atom_str.split()
            pred_name = parts[0]
            obj_names = parts[1:]

            if pred_name in self.name_to_predicate:
                atom = Atom(
                    predicate=self.name_to_predicate[pred_name], objects=obj_names
                )
                atoms.append(atom)

        return State(atoms)

    def parse_pddl_goal_to_wl_state(self, problem_pddl_path):
        """Extracts goal from PDDL and converts to wlplan.State"""
        domprob = pddlpy.DomainProblem(self.domain_pddl_path, problem_pddl_path)
        pddl_goals = domprob.goals()

        atoms = []
        for g in pddl_goals:
            # Handle pddlpy variations
            if hasattr(g, "predicate"):
                parts = g.predicate
            else:
                parts = g

            pred_name = parts[0]
            obj_names = list(parts[1:])

            if pred_name in self.name_to_predicate:
                atom = Atom(self.name_to_predicate[pred_name], obj_names)
                atoms.append(atom)

        return State(atoms)

    def collect_vocabulary(self, train_states_dir):
        """
        Iterates over all .traj files in the training directory to build the WL hash map.
        This MUST be called before embedding anything.
        """
        print(f"  [WL-Wrapper] Collecting vocabulary from {train_states_dir}...")
        train_files = glob.glob(os.path.join(train_states_dir, "*.traj"))

        if not train_files:
            raise ValueError(f"No training files found in {train_states_dir}")

        problem_datasets = []

        # We need the corresponding PDDL for every traj to create a ProblemDataset
        # Standard structure: data/states/<dom>/train/X.traj -> data/pddl/<dom>/train/X.pddl

        # Heuristic to find PDDL dir based on states dir structure
        # .../data/states/blocks/train -> .../data/pddl/blocks/train
        pddl_train_dir = train_states_dir.replace("states", "pddl")

        for traj_file in tqdm(train_files, desc="  Parsing Train Data"):
            prob_name = os.path.splitext(os.path.basename(traj_file))[0]
            prob_pddl = os.path.join(pddl_train_dir, f"{prob_name}.pddl")

            if not os.path.exists(prob_pddl):
                continue

            try:
                wlplan_prob = parse_problem(self.domain_pddl_path, prob_pddl)

                # Parse Trajectory
                with open(traj_file, "r") as f:
                    content = f.read()
                # Split by newlines to get states
                lines = content.strip().split("\n")
                states = [self.parse_state_string_to_wl_state(line) for line in lines]

                # Parse Goal (Important! Goals contain atoms that might not be in trajectories)
                goal_state = self.parse_pddl_goal_to_wl_state(prob_pddl)
                states.append(goal_state)

                problem_datasets.append(
                    ProblemDataset(problem=wlplan_prob, states=states)
                )
            except Exception:
                pass

        full_dataset = DomainDataset(domain=self.wlplan_domain, data=problem_datasets)
        self.generator.collect(full_dataset)
        self.is_collected = True
        print("  [WL-Wrapper] Vocabulary collected.")

    def embed_state(self, wl_state, problem_pddl_path):
        """
        Embeds a single wlplan.State object.
        Requires the problem PDDL to define objects/types context.
        """
        if not self.is_collected:
            raise RuntimeError("Must call collect_vocabulary before embedding.")

        wlplan_prob = parse_problem(self.domain_pddl_path, problem_pddl_path)
        dataset = DomainDataset(
            domain=self.wlplan_domain,
            data=[ProblemDataset(problem=wlplan_prob, states=[wl_state])],
        )
        # Returns [1, 1, D]
        emb = self.generator.embed(dataset)
        return emb[0][0]
