"""
Check the VLM against the hand-labelled photos in test_images/.

    python test_vlm.py                  # random current/goal pair, compare both to labels
    python test_vlm.py --pair A C       # a specific pair
    python test_vlm.py --trials 3       # three random pairs
    python test_vlm.py --all            # read every labelled photo once, report accuracy
    python test_vlm.py --all --mode 3d  # same, also checking which tokens lie on others
"""
import argparse
import json
import random
from pathlib import Path

from scene_to_goal import TaskError, build_2d, build_3d
from vlm import DEFAULT_CONFIG, PerceptionError, load_config, make_vlm, read_grid, read_shape, tomllib

HERE = Path(__file__).parent


def score(reading: dict, label: dict, check_on: bool):
    expected, expected_on = label["cells"], label.get("on", {})
    correct, notes = 0, []
    for name, cell in expected.items():
        if name not in reading:
            notes.append(f"missing {name}")
            continue
        ok = reading[name]["cell"] == list(cell)
        if not ok:
            notes.append(f"{name}: expected {list(cell)}, got {reading[name]['cell']}")
        if check_on:
            want, got = sorted(expected_on.get(name, [])), sorted(reading[name]["on"])
            if want != got:
                ok = False
                notes.append(f"{name}: expected on {want}, got {got}")
        correct += ok
    notes += [f"extra {name}" for name in reading if name not in expected]
    return correct, len(expected), notes


def read_and_score(vlm, cfg, labels, key, images_dir, role):
    allow_stacks = cfg["task"]["mode"] == "3d"
    path = images_dir / labels[key]["file"]
    print(f"\n--- {role}: {key} ({path.name})")
    try:
        reading = read_grid(vlm, cfg, path, allow_stacks=allow_stacks, label=f"{role} {key}")
    except PerceptionError as e:
        print(f"FAILED: {e}")
        return None, (0, len(labels[key]["cells"]))
    correct, total, notes = score(reading, labels[key], check_on=allow_stacks)
    print(f"{correct}/{total} correct" + ("" if not notes else ":\n  " + "\n  ".join(notes)))
    return reading, (correct, total)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="read every labelled photo once")
    ap.add_argument("--pair", nargs=2, metavar=("CURRENT", "GOAL"), help="label keys, e.g. A C")
    ap.add_argument("--trials", type=int, default=1, help="number of random pairs")
    ap.add_argument("--seed", type=int, help="random seed for repeatable pairs")
    ap.add_argument("--mode", choices=["2d", "3d"], help="override [task] mode in the config")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--images", default=HERE / "test_images")
    args = ap.parse_args()

    cfg = load_config(args.config, mode=args.mode)
    images_dir = Path(args.images)
    with open(images_dir / "labels.toml", "rb") as f:
        labels = tomllib.load(f)
    vlm = make_vlm(cfg)
    totals = [0, 0]

    def add(result):
        totals[0] += result[0]
        totals[1] += result[1]

    if args.all:
        for key in labels:
            add(read_and_score(vlm, cfg, labels, key, images_dir, "photo")[1])
    else:
        rng = random.Random(args.seed)
        pairs = [args.pair] if args.pair else [rng.sample(sorted(labels), 2) for _ in range(args.trials)]
        for cur_key, goal_key in pairs:
            print(f"\n========== current {cur_key} -> goal {goal_key} ==========")
            current, res = read_and_score(vlm, cfg, labels, cur_key, images_dir, "current")
            add(res)
            if cfg["task"]["mode"] == "2d":
                goal, res = read_and_score(vlm, cfg, labels, goal_key, images_dir, "goal")
                add(res)
            else:
                try:
                    goal = read_shape(vlm, cfg, images_dir / labels[goal_key]["file"])
                except PerceptionError as e:
                    print(f"goal FAILED: {e}")
                    goal = None
            if current is None or goal is None:
                continue
            try:
                build = build_2d if cfg["task"]["mode"] == "2d" else build_3d
                print("\nresult:\n" + json.dumps(build(cfg, current, goal), indent=2))
            except TaskError as e:
                print(f"\nreadings are valid but don't fit together: {e}")

    if totals[1]:
        print(f"\n===== overall: {totals[0]}/{totals[1]} objects correct "
              f"({100 * totals[0] / totals[1]:.0f}%) =====")


if __name__ == "__main__":
    main()