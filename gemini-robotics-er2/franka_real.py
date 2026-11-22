"""Runs the plan on the REAL Franka through frankapy, the same way as Vamsi's Lab3_Robot.py:
plan in Python, then one goto_joints per joint waypoint and open/close_gripper. No waypoint files.

Open loop: the robot assumes every block is exactly where the plan says. Never run alone;
keep a hand on the e-stop. Grid position comes from [franka] in sim_config.toml (robot base
frame, z = 0 at the table), so measure the real grid and set it there first.

    python franka_real.py --labels A C --dry-run                       # no robot, print the commands
    python franka_real.py states.json --grid-config ../gemini-robotics-er2/grid_config_rel.toml --dry-run
    python franka_real.py --labels A C --sim-check                     # same commands, sent to the MuJoCo Franka
    python franka_real.py states.json --z-offset 0.05 --no-gripper --step   # no blocks, 5 cm high
    python franka_real.py states.json --step                           # blocks, Enter before each pick/place
    python franka_real.py states.json                                  # full run
"""
import argparse
import json
import math
import sys
from pathlib import Path

import mink
import mujoco
import numpy as np

import franka_sim as fs
import grid_sim as gs

FRANKAPY_HOME = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
MIN_MOVE_TIME = 1.5   # s, shortest goto_joints
LINE_TOL = 0.002      # m, allowed TCP drift from a straight approach when thinning
START_TOL = 0.05      # rad, robot must be this close to where the next move starts
IK_TOL = 0.003        # m
TRANSIT_SPEED, APPROACH_SPEED = 0.8, 0.25   # rad/s before --speed scaling (as in Vamsi's Lab 3)


class DryRunArm:
    def __init__(self, verbose=False):
        self.q, self.verbose = FRANKAPY_HOME.copy(), verbose
        self.n_moves = self.n_grips = 0
        self.total_time = 0.0

    def reset_joints(self):
        self.q = FRANKAPY_HOME.copy()

    def get_joints(self):
        return self.q.copy()

    def goto_joints(self, joints, duration=5):
        self.q = np.array(joints, float)
        self.n_moves += 1
        self.total_time += duration
        if self.verbose:
            print(f"      goto_joints({np.round(self.q, 3).tolist()}, duration={duration:.1f})")

    def open_gripper(self):
        self.n_grips += 1
        if self.verbose:
            print("      open_gripper()")

    def close_gripper(self):
        self.n_grips += 1
        if self.verbose:
            print("      close_gripper()")


class SimArm(DryRunArm):
    """Plays the exact frankapy commands on the MuJoCo Franka (min-jerk joint moves)."""

    def __init__(self, cfg, placement):
        super().__init__()
        self.sim = gs.GridSim(cfg, fs.FRANKA, placement)
        self.cfg = cfg

    def get_joints(self):
        return self.sim.d.qpos[:7].copy()

    def goto_joints(self, joints, duration=5):
        super().goto_joints(joints, duration)
        q0, q1 = np.array(self.sim.q_des), np.array(joints, float)
        n = max(1, round(duration / gs.CTRL_DT))
        for k in range(1, n + 1):
            s = k / n
            self.sim.q_des = q0 + (q1 - q0) * (10 * s**3 - 15 * s**4 + 6 * s**5)
            self.sim.sim()
        self.sim.sim(5)

    def reset_joints(self):
        self.goto_joints(FRANKAPY_HOME, 4.0)

    def open_gripper(self):
        super().open_gripper()
        self.sim.set_gripper(self.cfg["robot"]["gripper_open"])

    def close_gripper(self):
        super().close_gripper()
        self.sim.set_gripper(self.cfg["robot"]["gripper_closed"])


class MotionPlanner:
    """Straight tool-point lines -> joint paths with mink IK on Lab 3's Panda model."""

    def __init__(self, cfg, z_offset=0.0):
        self.cfg, self.r, self.mo = cfg, cfg["robot"], cfg["motion"]
        self.m = mujoco.MjModel.from_xml_path(str(fs.FRANKA["dir"] / fs.FRANKA["include"]))
        self.d = mujoco.MjData(self.m)
        self.ik = mink.Configuration(self.m)
        self.task = mink.FrameTask("hand", "body", position_cost=1.0, orientation_cost=0.5)
        self.posture = mink.PostureTask(self.m, cost=1e-2)
        self.posture.set_target(self._full(fs.HOME_Q))
        self.limits = [mink.ConfigurationLimit(self.m)]
        self.R = fs.rotation(math.radians(self.r["yaw_deg"]))
        self.offset = self.R @ np.array([0, 0, fs.FRANKA["tool_offset"]])
        top = self.r["table_top_mm"] / 1000
        self.transit = top + self.mo["transit_height_mm"] / 1000
        self.grasp = top + self.r["grasp_height_mm"] / 1000 + z_offset
        self.place = self.grasp + self.mo["place_drop_mm"] / 1000
        self.max_ik_err = self.max_line_dev = 0.0

    def _full(self, q):
        return np.concatenate([q, [0.04, 0.04]])

    def tcp(self, q):
        self.d.qpos[:] = self._full(q)
        mujoco.mj_kinematics(self.m, self.d)
        b = self.d.body("hand")
        return b.xpos + b.xmat.reshape(3, 3) @ [0, 0, fs.FRANKA["tool_offset"]]

    def line(self, q_start, target):
        start = self.tcp(q_start)
        self.ik.update(self._full(q_start))
        n = max(1, math.ceil(np.linalg.norm(target - start) * 1000 / self.mo["step_mm"]))
        path = [np.array(q_start, float)]
        for k in range(1, n + 1):
            p = start + (target - start) * k / n
            self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(self.R), p - self.offset))
            for _ in range(50):
                vel = mink.solve_ik(self.ik, [self.task, self.posture], 0.05, "daqp", 1e-3, limits=self.limits)
                self.ik.integrate_inplace(vel, 0.05)
            q = self.ik.q[:7].copy()
            err = np.linalg.norm(self.tcp(q) - p)
            self.max_ik_err = max(self.max_ik_err, err)
            if err > IK_TOL:
                raise RuntimeError(f"IK can't reach {np.round(p, 3)} (off by {err * 1000:.0f} mm)")
            path.append(q)
        return self._thin(path)

    def _deviation(self, qa, qb):
        pa, pb = self.tcp(qa), self.tcp(qb)
        d, dev = pb - pa, 0.0
        for s in np.linspace(0, 1, 7)[1:-1]:
            p = self.tcp(qa + s * (qb - qa)) - pa
            if np.linalg.norm(d) > 1e-9:
                p = p - (p @ d) / (d @ d) * d
            dev = max(dev, np.linalg.norm(p))
        return dev

    def _thin(self, path):
        """Keep as few waypoints as possible while goto_joints' joint-space motion stays near the line."""
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and self._deviation(path[i], path[j]) > LINE_TOL:
                j -= 1
            self.max_line_dev = max(self.max_line_dev, self._deviation(path[i], path[j]))
            out.append(path[j])
            i = j
        return out

    def pick_and_place(self, q, src, dst):
        """Returns (steps, q_end). Steps: ("move", path, speed) or ("grip", "open"/"close", label)."""
        (sx, sy), (dx, dy) = gs.place_xy(self.cfg, src), gs.place_xy(self.cfg, dst)
        steps = []

        def move(target, speed):
            nonlocal q
            path = self.line(q, np.array(target, float))
            steps.append(("move", path, speed))
            q = path[-1]

        here = self.tcp(q)
        move([here[0], here[1], max(here[2], self.transit)], TRANSIT_SPEED)
        move([sx, sy, self.transit], TRANSIT_SPEED)
        move([sx, sy, self.grasp], APPROACH_SPEED)
        steps.append(("grip", "close", f"pick {src}"))
        move([sx, sy, self.transit], APPROACH_SPEED)
        move([dx, dy, self.transit], TRANSIT_SPEED)
        move([dx, dy, self.place], APPROACH_SPEED)
        steps.append(("grip", "open", f"place {dst}"))
        move([dx, dy, self.transit], APPROACH_SPEED)
        return steps, q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="?", help="JSON saved by scene_to_goal.py -o")
    ap.add_argument("--labels", nargs=2, metavar=("CURRENT", "GOAL"), help="labelled test photos, e.g. A C")
    ap.add_argument("--dry-run", action="store_true", help="don't connect; print what would be sent")
    ap.add_argument("--sim-check", action="store_true", help="send the same commands to the MuJoCo Franka")
    ap.add_argument("--verbose", action="store_true", help="with --dry-run: print every command")
    ap.add_argument("--step", action="store_true", help="wait for Enter before every pick and place")
    ap.add_argument("--z-offset", type=float, default=0.0, help="raise grasp/place heights by this many metres")
    ap.add_argument("--speed", type=float, default=0.5, help="scale on joint speeds (default 0.5)")
    ap.add_argument("--no-gripper", action="store_true", help="never open/close the gripper")
    ap.add_argument("--yes", action="store_true", help="skip the 'go' confirmation")
    ap.add_argument("--grid-config", help="grid file used for the photos (default: sim_config.toml's)")
    args = ap.parse_args()

    cfg = gs.load_config("franka", grid_config=args.grid_config)
    if args.labels:
        states = gs.jp.states_from_labels(*args.labels)
    elif args.states:
        states = json.loads(Path(args.states).read_text())
    else:
        ap.error("give a states JSON or --labels CURRENT GOAL")
    try:
        problem = gs.jp.problem_from_states(states, cfg["grid"]["rows"], cfg["grid"]["cols"], len(cfg["robot"]["buffers_mm"]))
        plan = gs.jp.plan(problem)
    except gs.jp.ProblemError as e:
        sys.exit(f"error: {e}")
    names = problem.tokens
    print(f"plan: {len(plan)} moves")
    for i, (t, a, b) in enumerate(plan, 1):
        print(f"  {i:2d}. move {names[t]}: {a} -> {b}")

    g, r = cfg["grid"], cfg["robot"]
    print(f"\ngrid {g['rows']}x{g['cols']}, cells {g['cell_width_mm']:.0f} x {g['cell_height_mm']:.0f} mm, "
          f"grasp {r['grasp_height_mm']:.0f} mm above the table + z offset {args.z_offset * 1000:.0f} mm")
    print("where the robot thinks each spot is (x forward, y left, mm from the base):")
    for p in gs.all_places(cfg):
        x, y = gs.place_xy(cfg, p)
        print(f"  {p:8s} x {x * 1000:6.1f}  y {y * 1000:+6.1f}")

    # Plan every joint path before anything moves, so IK problems show up first.
    mp = MotionPlanner(cfg, args.z_offset)
    jobs, q = [], np.array(fs.HOME_Q, float)
    for t, a, b in plan:
        steps, q = mp.pick_and_place(q, a, b)
        jobs.append((names[t], a, b, steps))
    print(f"\nmotion planned: max IK error {mp.max_ik_err * 1000:.1f} mm, "
          f"max drift from straight lines {mp.max_line_dev * 1000:.1f} mm")

    if args.sim_check:
        arm = SimArm(cfg, {names[t]: p for t, p in problem.init.items()})
    elif args.dry_run:
        arm = DryRunArm(args.verbose)
    else:
        print("\nAbout to move the REAL robot.")
        print("  - a partner is present with a hand on the e-stop")
        print("  - blocks are where the current state says, on the grid set in sim_config.toml [franka]")
        print(f"  - z offset {args.z_offset:.3f} m, speed {args.speed:.2f}, gripper {'OFF' if args.no_gripper else 'on'}")
        if not args.yes and input("Type 'go' to start: ").strip().lower() != "go":
            sys.exit("Aborted.")
        from frankapy import FrankaArm
        arm = FrankaArm()
        if hasattr(arm, "is_joints_in_collision_with_boxes"):
            bad = [(n, a, b) for n, a, b, steps in jobs for s in steps if s[0] == "move"
                   for qq in s[1] if arm.is_joints_in_collision_with_boxes([float(v) for v in qq])]
            if bad:
                sys.exit(f"Nothing was run: moves {sorted(set(bad))} go into frankapy's virtual walls.")

    arm.reset_joints()
    arm.goto_joints(list(fs.HOME_Q), duration=4.0)
    if not args.no_gripper:
        arm.open_gripper()

    try:
        for i, (name, a, b, steps) in enumerate(jobs, 1):
            print(f"\n{i}. {name}: {a} -> {b}")
            for kind, *rest in steps:
                if kind == "move":
                    path, speed = rest
                    err = np.max(np.abs(np.array(arm.get_joints()) - path[0]))
                    if err > START_TOL:
                        raise RuntimeError(f"robot is {err:.3f} rad from where the next move starts; stopping")
                    for q0, q1 in zip(path[:-1], path[1:]):
                        dur = max(MIN_MOVE_TIME, 1.875 * np.max(np.abs(q1 - q0)) / (speed * args.speed))
                        arm.goto_joints([float(v) for v in q1], duration=float(dur))
                else:
                    action, label = rest
                    if args.step and not (args.dry_run or args.sim_check):
                        if input(f"   next: {label}  [Enter = go, q = quit] ").strip().lower() == "q":
                            raise KeyboardInterrupt
                    print(f"   {label}")
                    if not args.no_gripper:
                        arm.close_gripper() if action == "close" else arm.open_gripper()
        arm.goto_joints(list(fs.HOME_Q), duration=4.0)
    except KeyboardInterrupt:
        sys.exit("\nStopped by the operator. The robot is left where it is.")

    print(f"\n{arm.n_moves} goto_joints, {arm.n_grips} gripper commands, about {arm.total_time / 60:.1f} min of motion")
    if args.sim_check:
        goal = {names[t]: p for t, p in problem.goal.items()}
        ok = True
        for name in sorted(goal):
            place, dist, upright, _ = arm.sim.where(name)
            good = place == goal[name] and dist <= cfg["motion"]["place_tolerance_mm"] and upright
            ok &= good
            print(f"  {name:18s} goal {goal[name]:7s} got {place:7s} {dist:4.0f} mm {'ok' if good else 'WRONG'}")
        print("SIM CHECK " + ("PASSED" if ok else "FAILED"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
