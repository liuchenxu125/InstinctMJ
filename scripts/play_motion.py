#!/usr/bin/env python3
"""Play CASBOT_02 retargeted motion data with MuJoCo viewer (XML-based).

Usage:
  uv run python scripts/play_motion.py 直行1步态
  uv run python scripts/play_motion.py CASBOT02_ShangTaiJie_1000HZ
  uv run python scripts/play_motion.py --list   # list available motions
"""

from __future__ import annotations

import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# --- config ---
XML_PATH = os.path.expanduser(
    "~/Desktop/InstinctMJ/src/instinct_mj/assets/resources/casbot_23dof/xml/CASBOT_02_23dof.xml"
)
DATA_DIR = os.path.expanduser(
    "~/Desktop/InstinctMJ/Datasets/casbot02/parkour_motion_reference"
)

# XML actuator order (23 joints) — must match the data
XML_JOINT_ORDER = [
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
    "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
    "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "waist_yaw_joint",
    "upper_left_1_joint", "upper_left_2_joint", "upper_left_3_joint",
    "upper_left_4_joint", "upper_left_5_joint",
    "upper_right_1_joint", "upper_right_2_joint", "upper_right_3_joint",
    "upper_right_4_joint", "upper_right_5_joint",
]


def list_motions():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_retargeted.npz"))
    for f in files:
        name = f.replace("_retargeted.npz", "")
        d = np.load(os.path.join(DATA_DIR, f), allow_pickle=True)
        n_frames = d["joint_pos"].shape[0]
        fps = float(d["framerate"])
        print(f"  {name:40s}  {n_frames:5d} frames  {fps:.0f} fps  ({n_frames/fps:.1f}s)")
        d.close()


def play_motion(motion_name: str):
    npz_path = os.path.join(DATA_DIR, f"{motion_name}_retargeted.npz")
    if not os.path.exists(npz_path):
        print(f"Motion not found: {npz_path}")
        print("Available motions:")
        list_motions()
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    joint_pos_all = np.array(data["joint_pos"])  # (T, 23)
    base_pos_all = np.array(data["base_pos_w"])  # (T, 3)
    base_quat_all = np.array(data["base_quat_w"])  # (T, 4)
    data_joint_names = list(data["joint_names"])

    n_frames = joint_pos_all.shape[0]
    print(f"Loaded: {motion_name}  ({n_frames} frames)")

    # Verify joint order matches
    assert len(data_joint_names) == 23, f"Expected 23 joints, got {len(data_joint_names)}"
    for i, (dn, xn) in enumerate(zip(data_joint_names, XML_JOINT_ORDER)):
        if dn != xn:
            print(f"  WARNING: joint[{i}] data='{dn}' != xml='{xn}'")

    # Load MuJoCo model
    spec = mujoco.MjSpec.from_file(XML_PATH)
    model = spec.compile()
    mj_data = mujoco.MjData(model)

    # Get actuator (motor) ID → joint ID mapping
    joint_name_to_id = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_name_to_id[name] = i

    # Get the qpos indices for each of our 23 joints
    joint_qpos_ids = []
    for jname in XML_JOINT_ORDER:
        jid = joint_name_to_id.get(jname)
        if jid is not None:
            # Get the qpos address for this joint
            addr = model.jnt_qposadr[jid]
            joint_qpos_ids.append(addr)
        else:
            print(f"WARNING: joint '{jname}' not found in XML!")
            joint_qpos_ids.append(None)

    print(f"XML joints mapped: {sum(1 for x in joint_qpos_ids if x is not None)}/23")
    print("Press ESC to exit, SPACE to pause")

    with mujoco.viewer.launch_passive(model, mj_data) as viewer:
        viewer.cam.lookat = (0.0, 0.0, 0.9)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 180

        frame = 0
        paused = False
        sim_time = 0.0
        fps = 50.0
        dt = 1.0 / fps

        while viewer.is_running():
            if not paused:
                # Set base (freejoint) position and orientation
                mj_data.qpos[0:3] = base_pos_all[frame]
                mj_data.qpos[3:7] = base_quat_all[frame]

                # Set joint positions
                for col, addr in enumerate(joint_qpos_ids):
                    if addr is not None:
                        mj_data.qpos[addr] = joint_pos_all[frame, col]

                # Forward kinematics
                mujoco.mj_forward(model, mj_data)

                frame = (frame + 1) % n_frames
                sim_time += dt

            # Control playback with keyboard
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1

            time.sleep(dt)

            # Sync to real-time roughly
            if not paused:
                viewer.sync()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("--list", "-l"):
        print("Available motions:")
        list_motions()
        print("\nUsage: python scripts/play_motion.py <motion_name>")
        sys.exit(0)

    play_motion(sys.argv[1])
