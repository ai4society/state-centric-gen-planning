import numpy as np
import pddl

# CONFIGURATION
# Max objects to track.
# 50 is chosen to safely cover OOD problems (e.g., 20 blocks + table + hand).
MAX_OBJECTS = 50

# Special Values
VAL_PAD = -99.0  # Slot is unused (outside problem size)
VAL_DONTCARE = -10.0  # Goal value when variable is not specified
VAL_UNKNOWN = -5.0  # Error case


class FSFEncoder:
    def __init__(self, domain_name, domain_pddl_path):
        self.domain_name = domain_name
        self.domain_pddl = pddl.parse_domain(domain_pddl_path)
        self.max_objects = MAX_OBJECTS

    def _get_sorted_objects(self, problem_path):
        """
        Parses problem to get all objects, including domain constants.
        Returns sorted list of object names.
        """
        problem = pddl.parse_problem(problem_path)

        objs = set()
        for o in problem.objects:
            objs.add(o.name)
        for o in self.domain_pddl.constants:
            objs.add(o.name)

        return sorted(list(objs))

    def _get_object_indices(self, objects):
        """Returns dict {obj_name: 1-based_index}"""
        return {o: i + 1 for i, o in enumerate(objects)}

    def parse_state_atoms(self, state_lines):
        """Parses list of strings ['(on a b)', ...] into tuples ('on', 'a', 'b')"""
        atoms = []
        for line in state_lines:
            line = line.strip().replace("(", "").replace(")", "").lower()
            parts = line.split()
            if parts:
                atoms.append(tuple(parts))
        return atoms

    def embed_trajectory(self, problem_path, trajectory_file):
        """
        Reads a .traj file and returns [T, MAX_OBJECTS] matrix.
        """
        objects = self._get_sorted_objects(problem_path)
        obj_map = self._get_object_indices(objects)

        with open(trajectory_file, "r") as f:
            lines = f.readlines()

        vectors = []
        for line in lines:
            atoms = self.parse_state_atoms([line])
            vec = self._state_to_vector(atoms, objects, obj_map)
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)

    def embed_goal(self, problem_path):
        """
        Parses problem goal and returns [MAX_OBJECTS] vector.
        Unspecified variables are set to VAL_DONTCARE.
        """
        objects = self._get_sorted_objects(problem_path)
        obj_map = self._get_object_indices(objects)

        problem = pddl.parse_problem(problem_path)

        # Extract goal atoms
        goal_atoms = []

        def visit(node):
            if hasattr(node, "name") and hasattr(node, "terms"):
                # Predicate
                args = [t.name if hasattr(t, "name") else str(t) for t in node.terms]
                goal_atoms.append(tuple([node.name] + args))
            elif hasattr(node, "operands"):
                for op in node.operands:
                    visit(op)

        visit(problem.goal)

        # Generate vector with DONTCARE as default
        return self._state_to_vector(
            goal_atoms, objects, obj_map, default_val=VAL_DONTCARE
        )

    def _state_to_vector(self, atoms, objects, obj_map, default_val=0.0):
        """
        Core Logic: Maps atoms to vector based on domain rules.
        """
        # Initialize with Padding
        vec = np.full(self.max_objects, VAL_PAD, dtype=np.float32)

        # Set active object slots to default_val (e.g., 0 for Table, or -10 for Goal)
        for i in range(len(objects)):
            if i < self.max_objects:
                vec[i] = default_val

        # Helper to get index safely
        def idx(name):
            return float(obj_map.get(name, 0))

        # DOMAIN SPECIFIC LOGIC
        if "blocks" in self.domain_name:
            # Slot i = Block i
            # Values: 0 (Table), -1 (Held), k (On block k)

            # First pass: Set held
            for pred in atoms:
                name = pred[0]
                args = pred[1:]

                if name == "holding":
                    # (holding a) -> V[a] = -1
                    if args[0] in obj_map:
                        vec[obj_map[args[0]] - 1] = -1.0

                elif name == "on":
                    # (on a b) -> V[a] = Index(b)
                    if args[0] in obj_map and args[1] in obj_map:
                        vec[obj_map[args[0]] - 1] = idx(args[1])

                elif name == "ontable":
                    # (ontable a) -> V[a] = 0
                    if args[0] in obj_map:
                        vec[obj_map[args[0]] - 1] = 0.0

        elif "gripper" in self.domain_name:
            # Slot i:
            #  - If Robby: Room Index
            #  - If Ball: Room Index OR -1 * Gripper Index (if carried)
            #  - If Gripper: 0 (Free) or Ball Index (Holding)

            # We need to know types. PDDL parser gives types, but let's infer from predicates for robustness
            # or just map everything.

            for pred in atoms:
                name = pred[0]
                args = pred[1:]

                if name == "at-robby":
                    # (at-robby room) -> Find robby object?
                    # Usually 'robby' is a constant or implicit.
                    # If implicit, we can't map it to a slot unless 'robby' is in objects.
                    # In standard gripper, 'at-robby' is unary. We assume 'robby' is not in objects list?
                    # If robby is not in objects, we can't assign a slot.
                    # Hack: Use the LAST available slot for global variables if needed.
                    # But usually 'gripper' domain has no explicit robot object, just grippers.
                    pass

                elif name == "at":
                    # (at ball room) -> V[ball] = Index(room)
                    obj, room = args
                    if obj in obj_map:
                        vec[obj_map[obj] - 1] = idx(room)

                elif name == "carry":
                    # (carry ball gripper)
                    # V[ball] = -1 * Index(gripper)
                    # V[gripper] = Index(ball)
                    obj, gripper = args
                    if obj in obj_map:
                        vec[obj_map[obj] - 1] = -1.0 * idx(gripper)
                    if gripper in obj_map:
                        vec[obj_map[gripper] - 1] = idx(obj)

                elif name == "free":
                    # (free gripper) -> V[gripper] = 0
                    grp = args[0]
                    if grp in obj_map:
                        vec[obj_map[grp] - 1] = 0.0

        elif "logistics" in self.domain_name:
            # Slot i:
            #  - Package/Truck/Plane: Location Index
            #  - Package (in vehicle): -1 * Vehicle Index

            for pred in atoms:
                name = pred[0]
                args = pred[1:]

                if name == "at":
                    # (at obj loc)
                    obj, loc = args
                    if obj in obj_map:
                        vec[obj_map[obj] - 1] = idx(loc)

                elif name == "in":
                    # (in pkg vehicle)
                    pkg, veh = args
                    if pkg in obj_map:
                        vec[obj_map[pkg] - 1] = -1.0 * idx(veh)

        elif "visit" in self.domain_name:  # visitall
            # Slot 0: Robot Location
            # Slot 1..N: Cells (0=Unvisited, 1=Visited)

            # We need to identify the robot. Usually implicit or 'robot'.
            # VisitAll usually has (at-robot x).
            # We will use a fixed convention:
            # If 'robot' is an object, it gets a slot.
            # If not, we might lose the robot pos if we only track objects.
            # However, in standard visitall, cells are objects.

            for pred in atoms:
                name = pred[0]
                args = pred[1:]

                if name == "at-robot":
                    # (at-robot cell)
                    # We need a place to store this.
                    # Let's assume the FIRST slot (index 0) is reserved for global state if needed,
                    # OR we just look for the object 'robot'.
                    # If 'robot' isn't in objects, we can't encode it in FSF easily without a dedicated 'globals' slot.
                    # Let's assume we map the CELL's value to a special code?
                    # No, better: V[cell] = 2 (Robot is here).
                    cell = args[0]
                    if cell in obj_map:
                        vec[obj_map[cell] - 1] = 2.0

                elif name == "visited":
                    # (visited cell) -> V[cell] = 1
                    cell = args[0]
                    if cell in obj_map:
                        # If robot is also there (2.0), we might overwrite.
                        # Let's use bitmask logic or priority?
                        # If robot (2) is there, it implies visited (1). So 2 is fine.
                        if vec[obj_map[cell] - 1] != 2.0:
                            vec[obj_map[cell] - 1] = 1.0

        return vec
