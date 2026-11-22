"""Franka Panda moves the cubes from the current state to the goal state (Lab 3 model and controller).

    python franka_sim.py --labels A C
    python franka_sim.py states.json --viewer      # on a Mac: mjpython franka_sim.py ...
    python franka_sim.py --labels A C --gif run.gif --snapshots out/
"""
import numpy as np

import grid_sim

HOME_Q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]   # Lab 2 start pose
KP = np.array([480, 480, 400, 360, 240, 160, 120], float)
KD = np.array([16, 16, 12, 10, 8, 6, 4], float)


def rotation(yaw):
    """Hand pointing down, fingers closing along (cos yaw, sin yaw, 0). Same as Lab 3."""
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


def torque_control(sim, q_des, grip):
    """Lab 3's PD + gravity compensation (the model's arm actuators are torque motors)."""
    q, qd = sim.d.qpos[:7], sim.d.qvel[:7]
    sim.d.ctrl[:7] = KP * (q_des - q) - KD * qd + sim.d.qfrc_bias[:7]
    sim.d.ctrl[7] = grip


FRANKA = dict(
    name="franka",
    dir=grid_sim.HERE / "assets" / "franka_emika_panda",
    include="panda_torque_table.xml",
    n_arm=7,
    ready=HOME_Q,
    gripper_qpos={"finger_joint1": 0.04, "finger_joint2": 0.04},
    ik_frame="hand",
    tool_offset=0.1034,  # hand frame -> centre of the finger pads
    rotation=rotation,
    control=torque_control,
    scene_group=1,       # arm uses geom groups 2-3
    snapshot_groups=[0, 1],
    add_floor=False,     # the model has its own ground plane and light
)

if __name__ == "__main__":
    grid_sim.main(FRANKA, __doc__)
