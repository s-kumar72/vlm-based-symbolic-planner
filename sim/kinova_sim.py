"""Kinova Gen3 lite moves the cubes from the current state to the goal state.

    python kinova_sim.py --labels A C
    python kinova_sim.py states.json --viewer      # on a Mac: mjpython kinova_sim.py ...
    python kinova_sim.py --labels A C --gif run.gif --snapshots out/
"""
import math

import numpy as np

import grid_sim

DOWN = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ DOWN


def position_control(sim, q_des, grip):
    sim.d.ctrl[:6] = q_des
    sim.d.ctrl[6] = grip


KINOVA = dict(
    name="kinova",
    dir=grid_sim.HERE / "assets" / "gen3_lite",
    include="gen3_lite.xml",
    n_arm=6,
    ready=[-0.118, -0.575, 0.968, 1.571, 1.599, -1.689],
    ik_frame="tool_frame",
    tool_offset=0.0,
    rotation=rotation,
    control=position_control,
    scene_group=2,      # arm uses geom groups 0-1
    snapshot_groups=[2],
    add_floor=True,     # also adds a light
    actuators="""<actuator>
    <position name="a1" joint="joint_1" kp="100" kv="10" ctrlrange="-2.69 2.69"/>
    <position name="a2" joint="joint_2" kp="100" kv="10" ctrlrange="-2.69 2.69"/>
    <position name="a3" joint="joint_3" kp="100" kv="10" ctrlrange="-2.69 2.69"/>
    <position name="a4" joint="joint_4" kp="50" kv="5" ctrlrange="-2.59 2.59"/>
    <position name="a5" joint="joint_5" kp="50" kv="5" ctrlrange="-2.57 2.57"/>
    <position name="a6" joint="joint_6" kp="50" kv="5" ctrlrange="-2.59 2.59"/>
    <position name="gripper" joint="right_finger_bottom_joint" kp="10" kv="1" ctrlrange="0 0.85"/>
  </actuator>""",
)

if __name__ == "__main__":
    grid_sim.main(KINOVA, __doc__)
