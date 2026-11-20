"""
Validation and geometry (turns raw VLM JSON into clean states and lists errors).

Every parse_* function returns (result, errors); a non-empty error list means
the reading is rejected and the VLM is prompted again.

Grid state:  {name: {"cell": [row, col], "on": [names underneath]}}   ("on" empty = on the table)
Shape:       {"on": {name: [names underneath]}, "base_order": [names]}
"""
import math


def _norm(s) -> str:
    return str(s).strip().lower()


def _check_name(name: str, cfg: dict, label: str, errors: list) -> None:
    names = cfg["objects"]["names"]
    if names and name not in names:
        errors.append(f"{label}: unknown object {name!r}")


# STACKING RULES

def stacking_errors(on_map: dict, label: str, max_supports: int) -> list:
    errors = []
    for name, below in on_map.items():
        if not below:
            continue
        if len(below) > max_supports:
            errors.append(f"{label}: {name} rests on {len(below)} objects (max {max_supports})")
        if len(set(below)) != len(below):
            errors.append(f"{label}: {name} lists the same support twice {below}")
        if name in below:
            errors.append(f"{label}: {name} rests on itself")
        for s in below:
            if s not in on_map:
                errors.append(f"{label}: {name} rests on {s!r}, which isn't in the photo")
    return errors


def stack_levels(on_map: dict, label: str):
    levels, visiting = {}, set()

    def level(n):
        if n in levels:
            return levels[n]
        if n in visiting:
            raise ValueError(n)
        visiting.add(n)
        below = on_map[n]
        levels[n] = 0 if not below else 1 + max(level(s) for s in below)
        visiting.discard(n)
        return levels[n]

    try:
        for n in on_map:
            level(n)
    except ValueError as e:
        return {}, [f"{label}: stacking loop involving {e}"]
    return levels, []


def support_errors(on_map: dict, levels: dict, label: str) -> list:
    errors, seen = [], {}
    for name, below in on_map.items():
        if not below:
            continue
        if len({levels[s] for s in below}) > 1:
            errors.append(f"{label}: {name} rests on objects at different heights {below}")
        key = tuple(sorted(below))
        if key in seen:
            errors.append(f"{label}: {name} and {seen[key]} both rest on exactly {list(key)}")
        seen[key] = name
    return errors


# GRID READINGS

def parse_grid(raw: dict, cfg: dict, label: str, allow_stacks: bool):
    g = cfg["grid"]
    state, errors = {}, []
    items = raw.get("objects") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return {}, [f"{label}: expected an 'objects' list"]
    for item in items:
        try:
            name = _norm(item["object"])
            row, col = int(item["row"]), int(item["col"])
            below = [_norm(s) for s in (item.get("on") or [])] if allow_stacks else []
        except (KeyError, TypeError, ValueError, AttributeError):
            errors.append(f"{label}: malformed entry {item!r}")
            continue
        _check_name(name, cfg, label, errors)
        if not (0 <= row < g["rows"] and 0 <= col < g["cols"]):
            errors.append(f"{label}: {name} at ({row},{col}) is outside the grid")
        if name in state:
            errors.append(f"{label}: {name} listed twice")
        state[name] = {"cell": [row, col], "on": below}
    if not state:
        errors.append(f"{label}: no objects found")

    limit = g["max_objects_per_cell"]
    if limit:
        per_cell = {}
        for name, e in state.items():
            if not e["on"]:
                per_cell.setdefault(tuple(e["cell"]), []).append(name)
        for cell, names in per_cell.items():
            if len(names) > limit:
                errors.append(f"{label}: cell {cell} has {len(names)} objects on the table {names} (max {limit})")

    if allow_stacks and not errors:
        on_map = {n: e["on"] for n, e in state.items()}
        errors += stacking_errors(on_map, label, cfg["stacking"]["max_supports"])
        if not errors:
            levels, errs = stack_levels(on_map, label)
            errors += errs or support_errors(on_map, levels, label)
        for name, e in state.items():  
            for s in e["on"]:
                if s in state:
                    (r1, c1), (r2, c2) = e["cell"], state[s]["cell"]
                    if abs(r1 - r2) > 1 or abs(c1 - c2) > 1:
                        errors.append(f"{label}: {name} at {e['cell']} rests on {s} at {state[s]['cell']}, too far away")
    return state, errors


# SHAPE READINGS

def parse_shape(raw: dict, cfg: dict, label: str = "goal"):
    errors, on_map = [], {}
    items = raw.get("objects") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return {}, [f"{label}: expected an 'objects' list"]
    for item in items:
        try:
            name = _norm(item["object"])
            below = [_norm(s) for s in (item.get("on") or [])]
        except (KeyError, TypeError, AttributeError):
            errors.append(f"{label}: malformed entry {item!r}")
            continue
        _check_name(name, cfg, label, errors)
        if name in on_map:
            errors.append(f"{label}: {name} listed twice")
        on_map[name] = below
    if not on_map:
        errors.append(f"{label}: no objects found")
    errors += stacking_errors(on_map, label, cfg["stacking"]["max_supports"])
    if errors:
        return {}, errors

    levels, errs = stack_levels(on_map, label)
    errors += errs or support_errors(on_map, levels, label)

    base_order = [_norm(n) for n in raw.get("base_order") or []]
    on_table = {n for n, b in on_map.items() if not b}
    if len(set(base_order)) != len(base_order):
        errors.append(f"{label}: base_order has duplicates {base_order}")
    if set(base_order) != on_table:
        errors.append(f"{label}: base_order {base_order} doesn't match the objects on the table {sorted(on_table)}")
    return {"on": on_map, "base_order": base_order}, errors


# ACROSS TWO PHOTOS

def same_objects_errors(current_names, goal_names) -> list:
    errors = []
    missing, extra = set(current_names) - set(goal_names), set(goal_names) - set(current_names)
    if missing:
        errors.append(f"in current but not in goal: {sorted(missing)}")
    if extra:
        errors.append(f"in goal but not in current: {sorted(extra)}")
    return errors


# GEOMETRY (CELLS/ STACKS -> MM)

def cell_center_mm(cfg: dict, row: int, col: int) -> list:
    g, r = cfg["grid"], cfg["robot"]
    gx = (col + 0.5) * g["cell_width_mm"]
    gy = (row + 0.5) * g["cell_height_mm"]
    th = math.radians(r["grid_rotation_deg"])
    ox, oy = r["grid_origin_mm"]
    return [round(ox + gx * math.cos(th) - gy * math.sin(th), 2),
            round(oy + gx * math.sin(th) + gy * math.cos(th), 2)]


def stack_positions(on_map: dict, levels: dict, base_xy: dict, level_height_mm: float) -> dict:
    pos = {}
    for name in sorted(on_map, key=lambda n: levels[n]):
        below = on_map[name]
        if below:
            x = sum(pos[s][0] for s in below) / len(below)
            y = sum(pos[s][1] for s in below) / len(below)
        else:
            x, y = base_xy[name]
        pos[name] = [round(x, 2), round(y, 2), round(levels[name] * level_height_mm, 2)]
    return pos