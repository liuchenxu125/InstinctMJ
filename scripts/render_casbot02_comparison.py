"""Render paired evaluation trajectories without rerunning the policy or physics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("logs/analysis/casbot02_checkpoint_comparison"))
    args = parser.parse_args()
    root = args.directory
    rows = json.loads((root / "results.json").read_text())
    crossings = json.loads((root / "crossings.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    traces = np.load(root / "trajectories.npz")
    qpos, dt = traces["qpos"], float(traces["dt"])
    n = len(rows) // 2
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    # Choose one well moving flat/up trial, and an early down-stair fall.
    # These illustrate outcomes; aggregate statistics include every trial.
    selections = [
        ("flat_08", 0.0, "up", "max_progress"),
        ("up_10cm_08", 0.1, "up", "max_progress"),
        ("down_05cm_08", 0.05, "down", "upright_crossing_max_progress"),
        ("down_10cm_08", 0.1, "down", "first_fall"),
    ]
    selected = []
    with mujoco.Renderer(model, height=400, width=640) as renderer:
        for name, height, direction, rule in selections:
            candidates = [
                i
                for i in range(n, 2 * n)
                if rows[i]["height_m"] == height and rows[i]["direction"] == direction and rows[i]["speed"] == 0.8
            ]
            falls = [i for i in candidates if rows[i]["outcome"] == "fall"]
            if rule == "upright_crossing_max_progress":
                completed = [i for i in candidates if crossings[i]["crossed_upright_for_1s"]]
                candidates = completed or candidates
            j = (
                min(falls, key=lambda i: rows[i]["time_s"])
                if rule == "first_fall" and falls
                else max(candidates, key=lambda i: rows[i]["max_progress_m"])
            )
            i = j - n
            meta = protocol["heightfields"][rows[j]["height_index"]]
            center = meta["center_xy"]
            start = meta["first_step_x"] - 0.65 if direction == "up" else 0
            goal = meta["plateau_left_x"] + 0.65 if direction == "up" else meta["last_step_x"] + 0.75
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [center[0] + (start + goal) / 2, center[1], meta["top_m"] / 2 + 0.85]
            camera.distance = 5.3
            camera.azimuth = 110
            camera.elevation = -22
            end = min(len(qpos), int((max(rows[i]["time_s"], rows[j]["time_s"]) + 1) / dt) + 1)
            snapshot_time = crossings[j]["passed_at_s"]
            snapshot_time = snapshot_time + 0.3 if snapshot_time is not None else max(0, rows[j]["time_s"] - 0.2)
            snapshot_frame = min(end - 1, round(snapshot_time / dt))
            with imageio.get_writer(
                root / f"{name}.mp4", fps=round(1 / dt), codec="libx264", quality=7, macro_block_size=None
            ) as writer:
                for frame in range(end):
                    panels = []
                    time_s = frame * dt + protocol["step_dt_s"]
                    for k in [i, j]:
                        data.qpos[:] = qpos[frame, k]
                        mujoco.mj_kinematics(model, data)
                        mujoco.mj_comPos(model, data)
                        renderer.update_scene(data, camera)
                        rgb = renderer.render().copy()
                        cv2.rectangle(rgb, (0, 0), (640, 77), (18, 18, 18), -1)
                        text = f"model_{rows[k]['checkpoint']}  {direction} {height*100:.0f}cm  vx=0.8"
                        cv2.putText(
                            rgb, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA
                        )
                        state = f"t={time_s:.1f}s"
                        if time_s >= rows[k]["time_s"]:
                            state += "  " + rows[k]["outcome"].upper() + " (pose held)"
                        cv2.putText(
                            rgb, state, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 220, 160), 1, cv2.LINE_AA
                        )
                        if crossings[k]["passed_at_s"] is not None and time_s >= crossings[k]["passed_at_s"]:
                            cv2.putText(
                                rgb,
                                "STAIR FLIGHT CROSSED (upright 1s)",
                                (12, 390),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (80, 255, 80),
                                1,
                                cv2.LINE_AA,
                            )
                        panels.append(rgb)
                    paired = np.concatenate(panels, axis=1)
                    writer.append_data(paired)
                    if frame == snapshot_frame:
                        imageio.imwrite(root / f"{name}.png", paired)
            selected.append(
                {
                    "file": f"{name}.mp4",
                    "selection_rule": rule,
                    "case_id": rows[j]["case_id"],
                    "model_2000": rows[i],
                    "model_8000": rows[j],
                }
            )
            print("VIDEO", name, rows[i]["outcome"], rows[j]["outcome"], flush=True)
    (root / "video_selection.json").write_text(json.dumps(selected, indent=2) + "\n")


if __name__ == "__main__":
    main()
