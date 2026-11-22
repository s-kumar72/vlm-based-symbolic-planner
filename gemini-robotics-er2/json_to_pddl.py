"""States JSON (from scene_to_goal.py, 2d mode) -> PDDL -> plan.

The same facts are used two ways:
  - written out as standard PDDL text (domain.pddl / problem.pddl), and
  - planned on directly with the A* from Lab 3 SymPlanner (built into this file)

Domain: tokens sit in places (grid cells, plus optional buffer spots beside the grid).
One token per place. A single action moves a token to an empty place.

    python json_to_pddl.py states.json                   # print the plan
    python json_to_pddl.py states.json --out-dir pddl/   # also write domain.pddl + problem.pddl
    python json_to_pddl.py --labels A C                  # use two hand-labelled test photos instead of a JSON
"""
import argparse
import json
import heapq
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

HERE = Path(__file__).parent
GRID_CONFIG = HERE / "grid_config.toml"
LABELS = HERE / "test_images" / "labels.toml"


# ================================================================ names

def token_id(name: str) -> str:
    """'lime green token' -> 'lime_green_token' (PDDL names can't have spaces)."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def cell_id(row: int, col: int) -> str:
    return f"r{row}c{col}"


def buffer_id(i: int) -> str:
    return f"buffer{i}"


def cell_ids(rows: int, cols: int) -> list:
    return [cell_id(r, c) for r in range(rows) for c in range(cols)]


def parse_place(place: str):
    """'r1c2' -> ("cell", 1, 2); 'buffer0' -> ("buffer", 0)."""
    m = re.fullmatch(r"r(\d+)c(\d+)", place)
    if m:
        return ("cell", int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"buffer(\d+)", place)
    if m:
        return ("buffer", int(m.group(1)))
    raise ValueError(f"unknown place {place!r}")


# ================================================================ problem

class ProblemError(ValueError):
    pass


@dataclass
class Problem:
    tokens: dict          # token id -> original name
    places: list          # every place id (grid cells, then buffers)
    init: dict            # token id -> place id
    goal: dict            # token id -> place id

    def init_facts(self) -> set:
        facts = {("at", (t, p)) for t, p in self.init.items()}
        occupied = set(self.init.values())
        facts |= {("empty", (p,)) for p in self.places if p not in occupied}
        return facts

    def goal_facts(self) -> set:
        return {("at", (t, p)) for t, p in self.goal.items()}


def problem_from_states(states: dict, rows: int, cols: int, n_buffers: int = 1) -> Problem:
    """Build the planning problem from scene_to_goal's 2d output."""
    if states.get("mode", "2d") != "2d":
        raise ProblemError("only 2d states can be turned into a plan so far (3d goals are shapes, "
                           "so the planner would first need to pick where to build them)")
    current, goal = states["current"], states["goal"]
    if set(current) != set(goal):
        raise ProblemError(f"current and goal have different tokens: "
                           f"{sorted(set(current) ^ set(goal))}")

    def to_places(state, label):
        placed, used = {}, {}
        for name, entry in state.items():
            r, c = entry["cell"]
            if not (0 <= r < rows and 0 <= c < cols):
                raise ProblemError(f"{label}: {name} at {[r, c]} is outside the {rows}x{cols} grid")
            p = cell_id(r, c)
            if p in used:
                raise ProblemError(f"{label}: {name} and {used[p]} are both in cell {[r, c]}; "
                                   "the robot plan needs one token per cell")
            used[p] = name
            placed[token_id(name)] = p
        return placed

    cells = cell_ids(rows, cols)
    buffers = [buffer_id(i) for i in range(n_buffers)]
    return Problem(tokens={token_id(n): n for n in current}, places=cells + buffers,
                   init=to_places(current, "current"), goal=to_places(goal, "goal"))


# ================================================================ PDDL text

DOMAIN_PDDL = """\
(define (domain grid-tokens)
  (:requirements :strips :typing)
  (:types token place)
  (:predicates
    (at ?t - token ?p - place)
    (empty ?p - place))

  ;; Pick a token up from one place and put it down in an empty place.
  (:action move
    :parameters (?t - token ?from - place ?to - place)
    :precondition (and (at ?t ?from) (empty ?to))
    :effect (and (at ?t ?to) (empty ?from)
                 (not (at ?t ?from)) (not (empty ?to)))))
"""


def fact_pddl(fact) -> str:
    name, args = fact
    return f"({name} {' '.join(args)})"


def problem_pddl(problem: Problem, name: str = "rearrange") -> str:
    nl = "\n    "
    init = sorted(problem.init_facts())
    goal = sorted(problem.goal_facts())
    return f"""(define (problem {name})
  (:domain grid-tokens)
  (:objects
    {' '.join(problem.tokens)} - token
    {' '.join(problem.places)} - place)
  (:init
    {nl.join(fact_pddl(f) for f in init)})
  (:goal (and
    {nl.join(fact_pddl(f) for f in goal)})))
"""


# ================================================================ planning (Lab 3's SymPlanner A*)
# Facts are (predicate, (args...)) tuples, e.g. ("at", ("pink_token", "r0c1")).

@dataclass(frozen=True)
class GroundAction:
    name: str
    args: tuple
    add: tuple
    delete: tuple


@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameters: tuple       # variables like ?t
    preconds: tuple         # positive literals only (STRIPS)
    add_effects: tuple
    del_effects: tuple = ()

    def ground(self, subst: dict) -> GroundAction:
        sub = lambda pred: (pred[0], tuple(subst.get(a, a) for a in pred[1]))
        return GroundAction(self.name, tuple(subst.get(v, v) for v in self.parameters),
                            tuple(sub(p) for p in self.add_effects),
                            tuple(sub(p) for p in self.del_effects))


# Same action as the move in DOMAIN_PDDL above.
MOVE = ActionSchema(
    name="move",
    parameters=("?t", "?from", "?to"),
    preconds=(("at", ("?t", "?from")), ("empty", ("?to",))),
    add_effects=(("at", ("?t", "?to")), ("empty", ("?from",))),
    del_effects=(("at", ("?t", "?from")), ("empty", ("?to",))),
)


def applicable_actions(state) -> list:
    """Every legal move: any token to any empty place."""
    empties = sorted(args[0] for name, args in state if name == "empty")
    out = []
    for _, (t, frm) in sorted((n, a) for n, a in state if n == "at"):
        for to in empties:
            out.append(MOVE.ground({"?t": t, "?from": frm, "?to": to}))
    return out


def apply(state, action: GroundAction) -> frozenset:
    return frozenset((set(state) - set(action.delete)) | set(action.add))


def heuristic(state, goal: set) -> int:
    """Goal facts not true yet. Each move makes at most one 'at' goal true, so this never overestimates."""
    return sum(1 for g in goal if g not in state)


def astar(init, goal, max_expansions: int = 50000):
    """A* over symbolic states with unit action cost (from Lab 3's SymPlanner).
    Returns a list of GroundActions, or None if no plan was found."""
    init_state, goal_set = frozenset(init), set(goal)
    counter = 0   # tie-breaker so the heap never compares frozensets
    frontier = [(heuristic(init_state, goal_set), 0, counter, init_state)]
    g_cost = {init_state: 0}
    parent = {init_state: (None, None)}
    closed, expansions = set(), 0

    while frontier:
        _, g, _, state = heapq.heappop(frontier)
        if state in closed:
            continue
        if goal_set <= state:
            steps = []
            while parent[state][0] is not None:
                prev, act = parent[state]
                steps.append(act)
                state = prev
            return steps[::-1]
        closed.add(state)
        expansions += 1
        if expansions > max_expansions:
            return None
        for act in applicable_actions(state):
            child = apply(state, act)
            if child not in g_cost or g + 1 < g_cost[child]:
                g_cost[child] = g + 1
                parent[child] = (state, act)
                counter += 1
                heapq.heappush(frontier, (g + 1 + heuristic(child, goal_set), g + 1, counter, child))
    return None


def plan(problem: Problem, max_expansions: int = 50000) -> list:
    """Returns [(token id, from place, to place), ...]; raises if no plan is found."""
    actions = astar(problem.init_facts(), problem.goal_facts(), max_expansions)
    if actions is None:
        raise ProblemError("no plan found (if the grid is full, add a buffer spot with --buffers)")
    return [tuple(a.args) for a in actions]


def write_pddl(problem: Problem, steps: list, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "domain.pddl").write_text(DOMAIN_PDDL)
    (out / "problem.pddl").write_text(problem_pddl(problem))
    (out / "plan.txt").write_text("\n".join(f"(move {t} {a} {b})" for t, a, b in steps) + "\n")
    print(f"wrote {out}/domain.pddl, problem.pddl, plan.txt")


# ================================================================ inputs

def load_grid_size(path=GRID_CONFIG):
    with open(path, "rb") as f:
        g = tomllib.load(f)["grid"]
    return g["rows"], g["cols"]


def states_from_labels(cur_key: str, goal_key: str, path=LABELS) -> dict:
    """Build a states dict from two hand-labelled test photos (no VLM needed)."""
    with open(path, "rb") as f:
        labels = tomllib.load(f)
    to_state = lambda key: {n: {"cell": list(c)} for n, c in labels[key]["cells"].items()}
    return {"mode": "2d", "current": to_state(cur_key), "goal": to_state(goal_key)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="?", help="JSON saved by scene_to_goal.py -o")
    ap.add_argument("--labels", nargs=2, metavar=("CURRENT", "GOAL"), help="use labelled test photos, e.g. A C")
    ap.add_argument("--buffers", type=int, default=1, help="spare spots beside the grid (default 1)")
    ap.add_argument("--grid-config", default=GRID_CONFIG)
    ap.add_argument("--out-dir", help="write domain.pddl, problem.pddl and plan.txt here")
    args = ap.parse_args()

    if args.labels:
        states = states_from_labels(*args.labels)
    elif args.states:
        states = json.loads(Path(args.states).read_text())
    else:
        ap.error("give a states JSON or --labels CURRENT GOAL")

    rows, cols = load_grid_size(args.grid_config)
    try:
        problem = problem_from_states(states, rows, cols, args.buffers)
        steps = plan(problem)
    except ProblemError as e:
        raise SystemExit(f"error: {e}")
    lines = [f"{i:2d}. move {t}: {a} -> {b}" for i, (t, a, b) in enumerate(steps, 1)]
    print(f"plan: {len(steps)} moves\n" + "\n".join(lines))

    if args.out_dir:
        write_pddl(problem, steps, args.out_dir)


if __name__ == "__main__":
    main()