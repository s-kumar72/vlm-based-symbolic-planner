"""Shared code for kinova_sim.py and franka_sim.py: scene, IK, pick-and-place, checks, command line."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import mink
import mujoco
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "gemini-robotics-er2"))
import json_to_pddl as jp  # noqa: E402

CTRL_DT = 0.1


class MoveFailed(RuntimeError):
    pass


# ================================================================ config + positions

def load_config(robot: str, path=HERE / "sim_config.toml") -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    with open((Path(path).parent / cfg["grid_config"]).resolve(), "rb") as f:
        cfg["grid"] = tomllib.load(f)["grid"]
    cfg["robot"] = cfg[robot]
    return cfg


def place_xy(cfg, place):
    r = cfg["robot"]
    if place.startswith("buffer"):
        x, y = r["buffers_mm"][int(place[6:])]
        return x / 1000, y / 1000
    row, col = (int(v) for v in place[1:].split("c"))
    g = cfg["grid"]
    gx, gy = (col + 0.5) * g["cell_width_mm"], (row + 0.5) * g["cell_height_mm"]
    return grid_to_world(cfg, gx, gy)


def grid_to_world(cfg, gx, gy):
    r = cfg["robot"]
    th = math.radians(r["grid_rotation_deg"])
    ox, oy = r["grid_origin_mm"]
    return ((ox + gx * math.cos(th) - gy * math.sin(th)) / 1000,
            (oy + gx * math.sin(th) + gy * math.cos(th)) / 1000)


def all_places(cfg):
    return jp.cell_ids(cfg["grid"]["rows"], cfg["grid"]["cols"]) + \
        [jp.buffer_id(i) for i in range(len(cfg["robot"]["buffers_mm"]))]


def body_name(token):
    return "cube_" + token.replace(" ", "_")


# ================================================================ scene

def _box(name, pos, half, rgba, group):
    return (f'<geom name="{name}" type="box" group="{group}" contype="0" conaffinity="0" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" size="{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}" '
            f'rgba="{" ".join(map(str, rgba))}"/>')


def build_scene(cfg, robot, placement, name="grid_scene.xml") -> Path:
    g, r, grp = cfg["grid"], cfg["robot"], robot["scene_group"]
    top, half = r["table_top_mm"] / 1000, cfg["cubes"]["size_mm"] / 2000
    w, h = g["cols"] * g["cell_width_mm"], g["rows"] * g["cell_height_mm"]
    cx, cy = grid_to_world(cfg, w / 2, h / 2)
    th = math.radians(r["grid_rotation_deg"])
    blue, t, m = (0.05, 0.2, 0.65, 1), 0.006, 0.01

    world = []
    if robot["add_floor"]:
        world.append('<light pos="0.5 0 2" dir="0 0 -1" directional="true" castshadow="false" '
                     'diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>')
        world.append(f'<geom name="floor" type="plane" group="{grp}" size="2 2 0.1" rgba="0.55 0.55 0.55 1"/>')
    if top > 0:
        world.append(f'<body name="table" pos="{cx:.3f} 0 {top / 2:.4f}"><geom name="table" type="box" '
                     f'group="{grp}" size="0.25 0.35 {top / 2:.4f}" rgba="0.6 0.4 0.3 1"/></body>')
    world.append(f'<body name="grid" pos="{cx:.4f} {cy:.4f} {top:.4f}" quat="{math.cos(th / 2):.6f} 0 0 {math.sin(th / 2):.6f}">')
    world.append(_box("paper", (0, 0, 0.0002), (w / 2000 + m, h / 2000 + m, 0.0002), (1, 1, 1, 1), grp))
    for i in range(g["cols"] + 1):
        world.append(_box(f"tape_c{i}", (i * g["cell_width_mm"] / 1000 - w / 2000, 0, 0.0005), (t, h / 2000 + m, 0.0003), blue, grp))
    for j in range(g["rows"] + 1):
        world.append(_box(f"tape_r{j}", (0, j * g["cell_height_mm"] / 1000 - h / 2000, 0.0006), (w / 2000 + m, t, 0.0003), blue, grp))
    world.append("</body>")
    for i, (bx, by) in enumerate(r["buffers_mm"]):
        world.append(_box(f"buffer{i}", (bx / 1000, by / 1000, top + 0.0003), (half + 0.01, half + 0.01, 0.0003), (0.8, 0.8, 0.8, 1), grp))
    for tok, place in placement.items():
        x, y = place_xy(cfg, place)
        rgba = " ".join(map(str, cfg["cubes"]["colors"].get(tok, [0.6, 0.6, 0.6, 1])))
        world.append(f'<body name="{body_name(tok)}" pos="{x:.4f} {y:.4f} {top + half + 0.001:.4f}"><freejoint/>'
                     f'<geom type="box" group="{grp}" size="{half} {half} {half}" rgba="{rgba}" '
                     f'mass="{cfg["cubes"]["mass_kg"]}" friction="{cfg["cubes"]["friction"]} 0.01 0.001"/></body>')
    world.append(f'<camera name="overhead" pos="{cx:.4f} {cy:.4f} 0.75" xyaxes="0 -1 0 1 0 0" fovy="30"/>')
    world.append(f'<camera name="grid_cam" mode="targetbody" target="grid" pos="{cx + 0.45:.3f} {cy - 0.45:.3f} 0.55" fovy="50"/>')

    xml = f"""<mujoco model="{robot['name']} grid scene">
  <include file="{robot['include']}"/>
  <option timestep="0.002"/>
  <visual><headlight diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.3" specular="0 0 0"/><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    {chr(10).join('    ' + s for s in world).lstrip()}
  </worldbody>
  {robot.get('actuators', '')}
</mujoco>
"""
    path = robot["dir"] / name
    path.write_text(xml)
    return path


# ================================================================ simulation

class GridSim:
    def __init__(self, cfg, robot, placement, viewer=False, gif=None):
        self.cfg, self.robot, self.mo, self.r = cfg, robot, cfg["motion"], cfg["robot"]
        self.n = robot["n_arm"]
        self.top = self.r["table_top_mm"] / 1000
        self.m = mujoco.MjModel.from_xml_path(str(build_scene(cfg, robot, placement)))
        self.d = mujoco.MjData(self.m)
        self.n_sub = round(CTRL_DT / self.m.opt.timestep)
        self.places = all_places(cfg)

        self.ik_model = mujoco.MjModel.from_xml_path(str(robot["dir"] / robot["include"]))
        self.ik = mink.Configuration(self.ik_model)
        self.task = mink.FrameTask(robot["ik_frame"], "body", position_cost=1.0, orientation_cost=0.5)
        self.posture = mink.PostureTask(self.ik_model, cost=1e-2)
        self.posture.set_target(self._full(robot["ready"]))
        self.limits = [mink.ConfigurationLimit(self.ik_model)]
        self.R = robot["rotation"](math.radians(self.r["yaw_deg"]))
        self.offset = self.R @ np.array([0, 0, robot["tool_offset"]])

        self.viewer = None
        if viewer:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.m, self.d)
        self.gif, self.frames, self.ticks = gif, [], 0
        self.renderer = mujoco.Renderer(self.m, 360, 480) if gif else None

        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:self.n] = robot["ready"]
        for j, v in robot.get("gripper_qpos", {}).items():
            self.d.joint(j).qpos = v
        self.q_des = np.array(robot["ready"], float)
        self.grip = self.r["gripper_open"]
        mujoco.mj_forward(self.m, self.d)
        self.sim(10)

    def _full(self, q):
        return np.concatenate([q, np.zeros(self.ik_model.nq - self.n)])

    def sim(self, control_steps=1):
        for _ in range(control_steps):
            t0 = time.time()
            for _ in range(self.n_sub):
                self.robot["control"](self, self.q_des, self.grip)
                mujoco.mj_step(self.m, self.d)
            if self.viewer is not None:
                if not self.viewer.is_running():
                    raise KeyboardInterrupt
                self.viewer.sync()
                time.sleep(max(0.0, CTRL_DT - (time.time() - t0)))
            self.ticks += 1
            if self.renderer is not None and self.ticks % 2 == 0:
                self.renderer.update_scene(self.d, camera="grid_cam")
                self.frames.append(self.renderer.render().copy())

    def tool(self):
        return self.d.body(self.robot["ik_frame"]).xpos + self.d.body(self.robot["ik_frame"]).xmat.reshape(3, 3) @ [0, 0, self.robot["tool_offset"]]

    def _ik(self, tool_target):
        self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(self.R), tool_target - self.offset))
        for _ in range(30):
            vel = mink.solve_ik(self.ik, [self.task, self.posture], 0.05, "daqp", 1e-3, limits=self.limits)
            self.ik.integrate_inplace(vel, 0.05)
        return self.ik.q[:self.n].copy()

    def go(self, target):
        start = self.tool()
        steps = max(1, math.ceil(np.linalg.norm(target - start) * 1000 / self.mo["step_mm"]))
        for k in range(1, steps + 1):
            self.q_des = self._ik(start + (target - start) * k / steps)
            self.sim()
        self.sim(3)

    def set_gripper(self, value):
        self.grip = value
        self.sim(self.mo["close_steps"])

    def where(self, token):
        b = self.d.body(body_name(token))
        x, y, z = b.xpos
        place = min(self.places, key=lambda p: math.dist(place_xy(self.cfg, p), (x, y)))
        dist = math.dist(place_xy(self.cfg, place), (x, y)) * 1000
        upright = np.abs(b.xmat.reshape(3, 3)[2]).max() > math.cos(math.radians(15))
        height = (z - self.cfg["cubes"]["size_mm"] / 2000 - self.top) * 1000
        return place, dist, upright, height

    def pick_and_place(self, token, src, dst):
        top, mo, r = self.top, self.mo, self.r
        transit = top + mo["transit_height_mm"] / 1000
        grasp = top + r["grasp_height_mm"] / 1000
        (sx, sy), (dx, dy) = place_xy(self.cfg, src), place_xy(self.cfg, dst)
        self.ik.update(self._full(self.d.qpos[:self.n]))
        here = self.tool()
        self.go(np.array([here[0], here[1], max(here[2], transit)]))
        self.go(np.array([sx, sy, transit]))
        self.set_gripper(r["gripper_open"])
        self.go(np.array([sx, sy, grasp]))
        self.set_gripper(r["gripper_closed"])
        self.go(np.array([sx, sy, transit]))
        if self.d.body(body_name(token)).xpos[2] - top < mo["transit_height_mm"] / 2000:
            raise MoveFailed(f"{token} wasn't lifted from {src}")
        self.go(np.array([dx, dy, transit]))
        self.go(np.array([dx, dy, grasp + mo["place_drop_mm"] / 1000]))
        self.set_gripper(r["gripper_open"])
        self.go(np.array([dx, dy, transit]))
        self.sim(5)
        place, dist, upright, height = self.where(token)
        if place != dst or dist > mo["place_tolerance_mm"] or not upright or abs(height) > 5:
            raise MoveFailed(f"{token} ended at {place} ({dist:.0f} mm off, "
                             f"{'upright' if upright else 'tipped over'}), expected {dst}")
        return dist

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.gif and self.frames:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in self.frames]
            imgs[0].save(self.gif, save_all=True, append_images=imgs[1:], duration=int(CTRL_DT * 2000), loop=0)
            print(f"saved {self.gif}")


def overhead_image(cfg, robot, placement=None, sim=None):
    """Top view framed like the phone photos, robot hidden."""
    if sim is None:
        m = mujoco.MjModel.from_xml_path(str(build_scene(cfg, robot, placement, "_snapshot.xml")))
        d = mujoco.MjData(m)
        d.qpos[:robot["n_arm"]] = robot["ready"]
        mujoco.mj_forward(m, d)
    else:
        m, d = sim.m, sim.d
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in robot["snapshot_groups"]:
        opt.geomgroup[g] = 1
    r = mujoco.Renderer(m, 480, 640)
    r.update_scene(d, camera="overhead", scene_option=opt)
    img = r.render().copy()
    r.close()
    return img


# ================================================================ command line

def main(robot, description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="?", help="JSON saved by scene_to_goal.py -o")
    ap.add_argument("--labels", nargs=2, metavar=("CURRENT", "GOAL"), help="labelled test photos, e.g. A C")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="watch it (on a Mac use mjpython)")
    ap.add_argument("--gif", help="save an animation")
    ap.add_argument("--snapshots", help="folder for overhead before / goal / after images")
    ap.add_argument("--out-dir", help="also write domain.pddl, problem.pddl, plan.txt")
    args = ap.parse_args()

    cfg = load_config(robot["name"])
    if args.labels:
        states = jp.states_from_labels(*args.labels)
    elif args.states:
        states = json.loads(Path(args.states).read_text())
    else:
        ap.error("give a states JSON or --labels CURRENT GOAL")
    try:
        problem = jp.problem_from_states(states, cfg["grid"]["rows"], cfg["grid"]["cols"], len(cfg["robot"]["buffers_mm"]))
        steps = jp.plan(problem)
    except jp.ProblemError as e:
        sys.exit(f"error: {e}")
    print(f"plan: {len(steps)} moves")
    for i, (t, a, b) in enumerate(steps, 1):
        print(f"  {i:2d}. move {t}: {a} -> {b}")
    if args.out_dir:
        jp.write_pddl(problem, steps, args.out_dir)
    if args.plan_only:
        return

    names = problem.tokens
    start = {names[t]: p for t, p in problem.init.items()}
    goal = {names[t]: p for t, p in problem.goal.items()}
    if args.snapshots:
        from PIL import Image
        out = Path(args.snapshots)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overhead_image(cfg, robot, start)).save(out / "before.png")
        Image.fromarray(overhead_image(cfg, robot, goal)).save(out / "goal.png")

    sim = GridSim(cfg, robot, start, viewer=args.viewer, gif=args.gif)
    t0, ok = time.time(), True
    try:
        for i, (t, a, b) in enumerate(steps, 1):
            try:
                dist = sim.pick_and_place(names[t], a, b)
                print(f"  {i:2d}. {names[t]:18s} {a} -> {b}  ok ({dist:.0f} mm off)")
            except MoveFailed as e:
                print(f"  {i:2d}. FAILED: {e}")
                ok = False
                break
        sim.sim(10)
        print("\nfinal check:")
        for name in sorted(goal):
            place, dist, upright, _ = sim.where(name)
            good = place == goal[name] and dist <= cfg["motion"]["place_tolerance_mm"] and upright
            ok &= good
            print(f"  {name:18s} goal {goal[name]:7s} got {place:7s} {dist:4.0f} mm {'ok' if good else 'WRONG'}")
        if args.snapshots:
            from PIL import Image
            Image.fromarray(overhead_image(cfg, robot, sim=sim)).save(Path(args.snapshots) / "after.png")
    finally:
        sim.close()
    print(f"\n{'SUCCESS' if ok else 'FAILED'}: {len(steps)} moves, {sim.d.time:.0f} s sim time ({time.time() - t0:.0f} s wall)")
    sys.exit(0 if ok else 1)
