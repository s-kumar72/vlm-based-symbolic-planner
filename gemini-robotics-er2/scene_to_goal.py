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

Modes:
- 2d: both photos show a grid & objects move between cells
- 3d: current photo shows the grid (objects may be stacked); goal photo shows a stacking shape and only the shape
        matters (not where it is on the grid)

Usage:
    python scene_to_goal.py current.jpg goal.jpg
    python scene_to_goal.py current.jpg goal.jpg --mode 3d -o states.json 
    python scene_to_goal.py --show-prompt [--mode 3d]

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
from pathlib import Path

from checks import cell_center_mm, same_objects_errors, stack_levels, stack_positions
from vlm import DEFAULT_CONFIG, grid_prompt, load_config, read_grid, read_shape, shape_prompt


class TaskError(ValueError):
    """The two readings are each valid but don't fit together."""


# 2d

def build_2d(cfg: dict, current: dict, goal: dict) -> dict:
    errors = same_objects_errors(current, goal)
    if errors:
        raise TaskError("; ".join(errors))
    fmt = lambda st: {n: {"cell": e["cell"], "xy_mm": cell_center_mm(cfg, *e["cell"])}
                      for n, e in sorted(st.items())}
    return {
        "mode": "2d",
        "current": fmt(current),
        "goal": fmt(goal),
        "changed": sorted(n for n in current if current[n]["cell"] != goal[n]["cell"]),
    }


# 3d

def build_3d(cfg: dict, current: dict, shape: dict) -> dict:
    errors = same_objects_errors(current, shape["on"])
    if errors:
        raise TaskError("; ".join(errors))
    st = cfg["stacking"]

    # Current
    cur_on = {n: e["on"] for n, e in current.items()}
    cur_lv, _ = stack_levels(cur_on, "current")
    cur_xy = {n: cell_center_mm(cfg, *e["cell"]) for n, e in current.items() if not e["on"]}
    cur_pos = stack_positions(cur_on, cur_lv, cur_xy, st["level_height_mm"])

    # Goal
    goal_on, base = shape["on"], shape["base_order"]
    goal_lv, _ = stack_levels(goal_on, "goal")
    goal_xy = {n: (i * st["spacing_mm"], 0.0) for i, n in enumerate(base)}
    goal_pos = stack_positions(goal_on, goal_lv, goal_xy, st["level_height_mm"])
    for name, below in goal_on.items():
        if len(below) == 2:
            (x1, y1, _), (x2, y2, _) = (goal_pos[s] for s in below)
            if abs(math.hypot(x1 - x2, y1 - y2) - st["spacing_mm"]) > 1e-6:
                raise TaskError(f"goal: {name} rests on {below}, which aren't next to each other")

    return {
        "mode": "3d",
        "current": {n: {"cell": current[n]["cell"], "on": cur_on[n], "level": cur_lv[n],
                        "xyz_mm": cur_pos[n]} for n in sorted(current)},
        "goal_shape": {n: {"on": goal_on[n], "level": goal_lv[n], "xyz_rel_mm": goal_pos[n]}
                       for n in sorted(goal_on)},
        "base_order": base,
        "build_order": sorted(goal_on, key=lambda n: (goal_lv[n], goal_pos[n][0], n)),
    }


# run

def run_task(vlm, cfg: dict, current_photo, goal_photo) -> dict:
    """Read both photos the way the mode needs, then combine them."""
    if cfg["task"]["mode"] == "2d":
        current = read_grid(vlm, cfg, current_photo, allow_stacks=False, label="current")
        goal = read_grid(vlm, cfg, goal_photo, allow_stacks=False, label="goal")
        return build_2d(cfg, current, goal)
    current = read_grid(vlm, cfg, current_photo, allow_stacks=True, label="current")
    shape = read_shape(vlm, cfg, goal_photo, label="goal")
    return build_3d(cfg, current, shape)


def show_prompts(cfg: dict) -> None:
    if cfg["task"]["mode"] == "2d":
        prompts = {"current + goal (grid)": grid_prompt(cfg, allow_stacks=False)}
    else:
        prompts = {"current (grid, stacks allowed)": grid_prompt(cfg, allow_stacks=True),
                   "goal (shape)": shape_prompt(cfg)}
    for title, (system, user) in prompts.items():
        print(f"===== {title} =====\n[system]\n{system}\n[user]\n{user}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("current", nargs="?", help="photo of the grid as it is now")
    ap.add_argument("goal", nargs="?", help="photo of the goal (grid in 2d, shape in 3d)")
    ap.add_argument("--mode", choices=["2d", "3d"], help="override [task] mode in the config")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="config file (default grid_config.toml)")
    ap.add_argument("-o", "--out", help="save the result JSON here")
    ap.add_argument("--show-prompt", action="store_true", help="print the prompts and exit")
    args = ap.parse_args()

    cfg = load_config(args.config, mode=args.mode)
    if args.show_prompt:
        show_prompts(cfg)
        return
    if not (args.current and args.goal):
        ap.error("give a current photo and a goal photo")

    from vlm import make_vlm
    result = run_task(make_vlm(cfg), cfg, args.current, args.goal)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
