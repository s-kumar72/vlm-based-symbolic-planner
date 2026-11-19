'''
SETUP:
- create venv and run pip install -r requirements.txt from w/in this folder
- export GEMINI_API_KEY=<key> (aistudio.google.com)

Photo requirements:
- current photo: objects in a grid
- goal photo: either a 2d goal state (objects move locations in the grid) or a 3d goal state (cup stacking).
              for 3d goal state, only the target shape matters (not the location, i.e. it can build it anywhere on the grid).
              for cup stacking, each cup is read as either resting on the table or on 1 - 2 other cups, which is then turned into
              positions relative to the first bottom cup

Usage:
    python scene_to_goal.py current.jpg goal.jpg
    python scene_to_goal.py current.jpg goal.jpg -o states.json --config my_grid.toml
    python scene_to_goal.py a b --show-prompt (prints VLM prompt only)

Output will be (for example):
    {
      "current": {"red cup":   {"on": "table", "cell": [0, 1], "level": 0, "xyz_mm": [150.0, 50.0, 0.0]},
                  "green cup": {"on": ["red cup"], "level": 1, "xyz_mm": [150.0, 50.0, 110.0]}, ...},
      "goal_shape": {"red cup":  {"on": "table", "level": 0, "xyz_rel_mm": [0.0, 0.0, 0.0]},
                     "pink cup": {"on": ["red cup", "blue cup"], "level": 1, "xyz_rel_mm": [45.0, 0.0, 110.0]}, ...},
      "base_order": ["red cup", "blue cup", "green cup"],
      "build_order": ["red cup", "blue cup", "green cup", "pink cup", ...]
    }
'''
import argparse
import json
import math
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DEFAULT_CONFIG = Path(__file__).with_name("grid_config.toml")
TABLE = "table"
MAX_SUPPORTS = 2

# PARSE CONFIG AND CREATE PROMPT
def load_config(path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    required = ("rows", "cols", "cell_width_mm", "cell_height_mm", "origin_marker", "axis_hint", "cup_spacing_mm", "stack_height_mm")
    for key in required:
        if key not in cfg:
            raise KeyError(f"{path} is missing '{key}")
    cfg["objects"] = [o.strip().lower() for o in cfg.get("objects", [])]
    cfg.setdefault("grid_origin_in_robot_mm", [0.0, 0.0])
    cfg.setdefault("grid_rotation_deg", 0.0)
    cfg.setdefault("model_id", "gemini-robotics-er-2-preview")
    return cfg

def build_prompt(cfg: dict) -> str:
    if cfg["objects"]:
        objects = ("The only objects are: " + ", ".join(cfg["objects"]) + ". Use these names exactly.")
    else:
        objects = ("Give each object a short descriptive name (e.g. 'red cup'), "
                "and use the same name for the same object in both images.")
    return f"""You will see two photos of the same set of cups.
{objects}

IMAGE 1 is the CURRENT state: an overhead photo of a grid on a table.
The grid has {cfg['rows']} rows (numbered 0 to {cfg['rows'] - 1}) and {cfg['cols']} columns (numbered 0 to {cfg['cols'] - 1}).
Cell (row 0, col 0) is the origin cell, marked with {cfg['origin_marker']}.
{cfg['axis_hint']}
The origin marker is not an object. Some tokens may already be stacked on other tokens.
For every object in IMAGE 1:
- if it sits on the table, give its row and col;
- if it rests on other tokens, give "on": a list of the 1 or 2 tokens directly underneath it.

IMAGE 2 is the GOAL: the shape the tokens should be arranged in.
Ignore where the shape is on the table or grid; only the shape matters.
For every object in IMAGE 2:
- if it sits on the table, give "on": "table";
- if it rests on other tokens, give "on": a list of the 1 or 2 tokens directly underneath it.
Also give "base_order": the tokens that sit on the table in IMAGE 2, from left to right as seen in IMAGE 2.

Every object in IMAGE 1 must appear in IMAGE 2, and vice versa.

Return only JSON in this form:
{{"current": [{{"object": "<name>", "row": <int>, "col": <int>}},
            {{"object": "<name>", "on": ["<name>", "<name>"]}}, ...],
"goal": [{{"object": "<name>", "on": "table"}},
        {{"object": "<name>", "on": ["<name>", "<name>"]}}, ...],
"base_order": ["<name>", ...]}}
"""

# PARSE VLM OUTPUT
def _norm(s) -> str:
    return str(s).strip().lower()

def parse_state(items, cfg: dict, label: str, with_cells: bool):
    """Model's list -> ({name: {"on": "table" | [names], "cell": (r, c)}}, errors)."""
    state, errors = {}, []
    if not isinstance(items, list):
        return {}, [f"{label}: expected a list, got {type(items).__name__}"]
    for item in items:
        if not isinstance(item, dict) or "object" not in item:
            errors.append(f"{label}: malformed entry {item!r}")
            continue
        name = _norm(item["object"])
        if cfg["objects"] and name not in cfg["objects"]:
            errors.append(f"{label}: unknown object {name!r}")
        if name in state:
            errors.append(f"{label}: {name} listed twice")
            continue
        on = item.get("on", TABLE)
        if isinstance(on, str):
            on = TABLE if _norm(on) in (TABLE, "grid", "") else [_norm(on)]
        elif isinstance(on, list):
            on = [_norm(x) for x in on]
        else:
            errors.append(f"{label}: {name} has malformed 'on' {on!r}")
            continue
        entry = {"on": on}
        if on == TABLE and with_cells:
            try:
                r, c = int(item["row"]), int(item["col"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label}: {name} is on the table but has no valid row/col")
                continue
            if not (0 <= r < cfg["rows"] and 0 <= c < cfg["cols"]):
                errors.append(f"{label}: {name} at ({r},{c}) is outside the grid")
            entry["cell"] = (r, c)
        state[name] = entry

    if with_cells:
        cells = [e["cell"] for e in state.values() if "cell" in e]
        for cell in {c for c in cells if cells.count(c) > 1}:
            names = [n for n, e in state.items() if e.get("cell") == cell]
            errors.append(f"{label}: cell {cell} has more than one cup on the table {names}")
    return state, errors


# CHECK STRUCTURE

def structure_errors(state: dict, label: str) -> list:
    errors = []
    for name, e in state.items():
        on = e["on"]
        if on == TABLE:
            continue
        if not 1 <= len(on) <= MAX_SUPPORTS:
            errors.append(f"{label}: {name} rests on {len(on)} cups (must be 1 or {MAX_SUPPORTS})")
        if len(set(on)) != len(on):
            errors.append(f"{label}: {name} lists the same support twice {on}")
        if name in on:
            errors.append(f"{label}: {name} rests on itself")
        for s in on:
            if s not in state:
                errors.append(f"{label}: {name} rests on {s!r}, which isn't in the image")
    return errors


def compute_levels(state: dict, label: str):
    """Level 0 = on the table; otherwise one above its supports. Detects loops."""
    levels, errors, visiting = {}, [], set()

    def level(n):
        if n in levels:
            return levels[n]
        if n in visiting:
            raise ValueError(n)
        visiting.add(n)
        on = state[n]["on"]
        lv = 0 if on == TABLE else 1 + max(level(s) for s in on)
        visiting.discard(n)
        levels[n] = lv
        return lv

    for n in state:
        try:
            level(n)
        except ValueError as e:
            errors.append(f"{label}: stacking loop involving {e}")
            break
    return levels, errors


def physical_errors(state: dict, levels: dict, label: str) -> list:
    errors = []
    seen_supports = {}
    for name, e in state.items():
        on = e["on"]
        if on == TABLE:
            continue
        if len({levels[s] for s in on}) > 1:
            errors.append(f"{label}: {name} rests on cups at different heights {on}")
        key = tuple(sorted(on))
        if key in seen_supports:
            errors.append(f"{label}: {name} and {seen_supports[key]} both rest on exactly {list(key)}")
        seen_supports[key] = name
    return errors


# GET GEOMETRY

def cell_center_mm(cfg: dict, row: int, col: int):
    """Centre of a cell in the robot frame (mm). Grid frame: x along columns, y along rows,
    origin at the outer corner of cell (0,0)."""
    gx = (col + 0.5) * cfg["cell_width_mm"]
    gy = (row + 0.5) * cfg["cell_height_mm"]
    th = math.radians(cfg["grid_rotation_deg"])
    ox, oy = cfg["grid_origin_in_robot_mm"]
    return (ox + gx * math.cos(th) - gy * math.sin(th),
            oy + gx * math.sin(th) + gy * math.cos(th))


def positions(state: dict, levels: dict, base_xy: dict, cfg: dict) -> dict:
    """xyz for every cup: table cups from base_xy; stacked cups centred over their supports."""
    pos = {}
    for name in sorted(state, key=lambda n: levels[n]):
        on = state[name]["on"]
        if on == TABLE:
            x, y = base_xy[name]
        else:
            x = sum(pos[s][0] for s in on) / len(on)
            y = sum(pos[s][1] for s in on) / len(on)
        pos[name] = [round(x, 2), round(y, 2), round(levels[name] * cfg["stack_height_mm"], 2)]
    return pos


# VALIDATE & BUILD OUTPUT

def validate(raw: dict, cfg: dict):
    """Model's raw JSON -> (output dict, []) if valid, else (None, errors)."""
    if not isinstance(raw, dict):
        return None, [f"expected a JSON object, got {type(raw).__name__}"]
    cur, e1 = parse_state(raw.get("current"), cfg, "current", with_cells=True)
    goal, e2 = parse_state(raw.get("goal"), cfg, "goal", with_cells=False)
    errors = e1 + e2 + structure_errors(cur, "current") + structure_errors(goal, "goal")
    if not cur:
        errors.append("no objects found in the current image")
    missing, extra = set(cur) - set(goal), set(goal) - set(cur)
    if missing:
        errors.append(f"in current but not in goal: {sorted(missing)}")
    if extra:
        errors.append(f"in goal but not in current: {sorted(extra)}")
    if errors:
        return None, errors

    lv_cur, e1 = compute_levels(cur, "current")
    lv_goal, e2 = compute_levels(goal, "goal")
    errors = e1 + e2
    if errors:
        return None, errors
    errors = physical_errors(cur, lv_cur, "current") + physical_errors(goal, lv_goal, "goal")

    base_order = [_norm(n) for n in raw.get("base_order") or []]
    goal_table = {n for n, e in goal.items() if e["on"] == TABLE}
    if len(set(base_order)) != len(base_order):
        errors.append(f"base_order has duplicates {base_order}")
    if set(base_order) != goal_table:
        errors.append(f"base_order {base_order} doesn't match the goal's table cups {sorted(goal_table)}")
    if errors:
        return None, errors

    # Positions
    cur_xy = {n: cell_center_mm(cfg, *e["cell"]) for n, e in cur.items() if e["on"] == TABLE}
    cur_pos = positions(cur, lv_cur, cur_xy, cfg)
    spacing = cfg["cup_spacing_mm"]
    goal_xy = {n: (i * spacing, 0.0) for i, n in enumerate(base_order)}
    goal_pos = positions(goal, lv_goal, goal_xy, cfg)

    for name, e in cur.items():
        if e["on"] != TABLE and len(e["on"]) == 2:
            a, b = (cur[s] for s in e["on"])
            if "cell" in a and "cell" in b:
                (r1, c1), (r2, c2) = a["cell"], b["cell"]
                if abs(r1 - r2) + abs(c1 - c2) != 1:
                    errors.append(f"current: {name} rests on {e['on']}, which aren't in adjacent cells")
    for name, e in goal.items():
        if e["on"] != TABLE and len(e["on"]) == 2:
            (x1, y1, _), (x2, y2, _) = (goal_pos[s] for s in e["on"])
            if abs(math.hypot(x1 - x2, y1 - y2) - spacing) > 1e-6:
                errors.append(f"goal: {name} rests on {e['on']}, which aren't next to each other")
    if errors:
        return None, errors

    def fmt(state, levels, pos, xyz_key, cells):
        out = {}
        for n in sorted(state):
            entry = {"on": state[n]["on"]}
            if cells and "cell" in state[n]:
                entry["cell"] = list(state[n]["cell"])
            entry["level"] = levels[n]
            entry[xyz_key] = pos[n]
            out[n] = entry
        return out

    return {
        "current": fmt(cur, lv_cur, cur_pos, "xyz_mm", cells=True),
        "goal_shape": fmt(goal, lv_goal, goal_pos, "xyz_rel_mm", cells=False),
        "base_order": base_order,
        "build_order": sorted(goal, key=lambda n: (lv_goal[n], goal_pos[n][0], n)),
    }, []


# VLM 

def ask_once(client, cfg: dict, prompt: str, current_img, goal_img) -> dict:
    from google.genai import types
    resp = client.models.generate_content(
        model=cfg["model_id"],
        contents=["IMAGE 1 (CURRENT):", current_img, "IMAGE 2 (GOAL):", goal_img, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(resp.text)


def get_states(cfg: dict, current_path: str, goal_path: str,
               votes: int = 2, max_attempts: int = 5) -> dict:
    """Ask until `votes` valid answers agree, so a single misread can't slip through."""
    from google import genai
    from PIL import Image

    client = genai.Client()  # reads GEMINI_API_KEY
    prompt = build_prompt(cfg)
    current_img, goal_img = Image.open(current_path), Image.open(goal_path)
    counts: dict[str, int] = {}
    for attempt in range(1, max_attempts + 1):
        try:
            raw = ask_once(client, cfg, prompt, current_img, goal_img)
        except json.JSONDecodeError as e:
            print(f"[attempt {attempt}] model didn't return JSON: {e}", file=sys.stderr)
            continue
        result, errors = validate(raw, cfg)
        if errors:
            print(f"[attempt {attempt}] rejected: {errors}", file=sys.stderr)
            continue
        key = json.dumps(result, sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
        print(f"[attempt {attempt}] valid answer ({counts[key]}/{votes} matching)", file=sys.stderr)
        if counts[key] >= votes:
            return result
    raise RuntimeError(f"No {votes} matching valid answers after {max_attempts} attempts")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("current", help="photo of the grid as it is now")
    ap.add_argument("goal", help="photo of the target shape")
    ap.add_argument("-o", "--out", help="save the result JSON here")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="grid config (default grid_config.toml)")
    ap.add_argument("--votes", type=int, default=2, help="matching answers required (default 2)")
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.show_prompt:
        print(build_prompt(cfg))
        return
    result = get_states(cfg, args.current, args.goal, votes=args.votes)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
