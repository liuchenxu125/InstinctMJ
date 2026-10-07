"""GPU regression check for CASBOT02 heightfield contacts, including fallen poses.

Run with: uv run python scripts/check_casbot02_hfield_contacts.py
Each physical robot geom is isolated to identify overflowing geometry, rather
than relying on a short rollout to randomly reproduce a rare collision.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import tempfile
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from instinct_mj.assets.casbot_02 import get_casbot02_parkour_spec


def numbers(values):
    return " ".join(f"{value:.9g}" for value in np.asarray(values).ravel())


def make_model(source, resolution):
    ids = np.flatnonzero((source.geom_contype | source.geom_conaffinity) != 0)
    assets = []
    bodies = []
    shapes = []
    for gid in ids:
        kind = source.geom_type[gid]
        if kind not in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError(f"Unsupported physical geom {gid}: {kind}")
        rotation = Rotation.from_quat(source.geom_quat[gid][[1, 2, 3, 0]]).as_matrix()
        position = source.geom_pos[gid]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = source.geom_dataid[gid]
            start, count = source.mesh_vertadr[mid], source.mesh_vertnum[mid]
            vertices = source.mesh_vert[start : start + count]
            vertices = vertices[ConvexHull(vertices).vertices]
            assets.append(f'<mesh name="mesh_{gid}" vertex="{numbers(vertices)}"/>')
            geometry = f'type="mesh" mesh="mesh_{gid}"'
            shape = {"vertices": vertices @ rotation.T + position}
        else:
            radius = source.geom_size[gid, 0]
            half_length = source.geom_size[gid, 1] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            geometry = (
                f'type="capsule" size="{radius} {half_length}"' if half_length else f'type="sphere" size="{radius}"'
            )
            shape = {"position": position, "axis": rotation[:, 2], "radius": radius, "half_length": half_length}
        name = source.geom(gid).name or source.body(source.geom_bodyid[gid]).name
        shapes.append((name, shape))
        bodies.append(
            f'<body name="body_{gid}" pos="{5 * (len(bodies) + 1)} 0 5">'
            '<freejoint/><inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>'
            f'<geom name="geom_{gid}" {geometry} pos="{numbers(position)}" '
            f'quat="{numbers(source.geom_quat[gid])}" condim="3"/></body>'
        )
    cells = 60
    extent = cells * resolution / 2
    assets.append(f'<hfield name="terrain" nrow="{cells+1}" ncol="{cells+1}" size="{extent} {extent} 1 1"/>')
    xml = '<mujoco><option cone="pyramidal"/><asset>' + "".join(assets) + "</asset><worldbody>"
    xml += '<geom name="terrain" type="hfield" hfield="terrain"/>' + "".join(bodies) + "</worldbody></mujoco>"
    return mujoco.MjModel.from_xml_string(xml), shapes


def check_contacts(source, resolution, device, random_poses=0):
    model, shapes = make_model(source, resolution)
    # Standing, toe/heel contact, side contact and fully tipped geometries.
    rotations = [
        Rotation.from_euler("xyz", [roll, pitch, yaw])
        for roll, pitch in [(0, 0), (0, np.pi / 4), (0, np.pi / 2), (np.pi / 2, 0), (np.pi / 4, np.pi / 4), (0, np.pi)]
        for yaw in [0, np.pi / 8, np.pi / 4, np.pi / 2]
    ]
    scenarios = [
        (rotation, penetration, phase)
        for rotation in rotations
        for penetration in [0.001, 0.02, 0.07, 0.15]
        for phase in [0.0, 0.031]
    ]
    # Exercise orientations and offsets between the deterministic samples.
    rng = np.random.default_rng(42)
    for _ in range(random_poses):
        scenarios.append(
            (
                Rotation.random(random_state=rng),
                float(rng.choice([0.001, 0.02, 0.07, 0.15])),
                rng.uniform(0, resolution, 2),
            )
        )
    native = mujoco.MjData(model)
    qpos0 = native.qpos.copy()
    report = {"resolution_m": resolution, "scenarios_per_geom": len(scenarios), "geoms": []}
    with wp.ScopedDevice(device):
        warp_model = mjwarp.put_model(model)
        data = mjwarp.put_data(model, native, nworld=len(scenarios), nconmax=64, njmax=128, nccdmax=64)
        grid = np.linspace(-30 * resolution, 30 * resolution, 61)
        xx, yy = np.meshgrid(grid, grid)
        terrains = {
            "flat": np.zeros_like(xx),
            "rough": 0.03 * (1 + np.sin(23 * xx) * np.cos(17 * yy)),
            "stairs": np.clip(np.floor((xx + 0.35) / 0.35), 0, 3) * 0.1,
        }
        for terrain_name, heights in terrains.items():
            warp_model.hfield_data.assign(heights.astype(np.float32).ravel())
            reference_height = float(heights[30, 30])
            for index, (name, shape) in enumerate(shapes):
                qpos = np.tile(qpos0, (len(scenarios), 1))
                for world, (rotation, penetration, phase) in enumerate(scenarios):
                    phase_x, phase_y = np.broadcast_to(phase, (2,))
                    matrix = rotation.as_matrix()
                    if "vertices" in shape:
                        lowest = (shape["vertices"] @ matrix.T)[:, 2].min()
                    else:
                        lowest = (matrix @ shape["position"])[2]
                        lowest -= abs((matrix @ shape["axis"])[2]) * shape["half_length"] + shape["radius"]
                    qpos[world, index * 7 : index * 7 + 3] = [phase_x, phase_y, reference_height - lowest - penetration]
                    qpos[world, index * 7 + 3 : index * 7 + 7] = rotation.as_quat()[[3, 0, 1, 2]]
                data.qpos.assign(qpos.astype(np.float32))
                # CUDA printf goes to the process stdout FD. Capture it per
                # isolated geom and report the real warning, not hide it.
                with tempfile.TemporaryFile(mode="w+b") as output:
                    saved_stdout = os.dup(1)
                    ctypes.CDLL(None).fflush(None)
                    try:
                        os.dup2(output.fileno(), 1)
                        mjwarp.kinematics(warp_model, data)
                        mjwarp.collision(warp_model, data)
                        wp.synchronize()
                        ctypes.CDLL(None).fflush(None)
                    finally:
                        os.dup2(saved_stdout, 1)
                        os.close(saved_stdout)
                    output.seek(0)
                    warnings = output.read().decode(errors="replace")
                count = warnings.count("height field collision overflow")
                contacts = int(data.nacon.numpy()[0])
                if contacts == 0:
                    raise AssertionError(f"Stress check produced no contacts: {terrain_name}/{name}")
                result = {"terrain": terrain_name, "geom": name, "overflow_warnings": count, "contacts": contacts}
                report["geoms"].append(result)
                print(json.dumps(result), flush=True)
        report["overflow_warnings"] = sum(item["overflow_warnings"] for item in report["geoms"])
        report["checked_geom_terrain_pairs"] = len(report["geoms"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, help="Optional baseline .mjb model")
    parser.add_argument(
        "--resolution", type=float, help="Collision grid spacing in meters; defaults to the CASBOT02 task"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--random-poses", type=int, default=0, help="Additional seeded random orientations and grid offsets"
    )
    parser.add_argument("--report", type=Path, default=Path("/tmp/casbot02_hfield_contact_report.json"))
    parser.add_argument("--expect-overflow", action="store_true", help="Check that the baseline reproduces the warning")
    args = parser.parse_args()
    if args.resolution is None:
        from instinct_mj.tasks.parkour.config.casbot02.casbot02_parkour_target_amp_cfg import (
            CASBOT02_PARKOUR_HFIELD_RESOLUTION,
        )

        args.resolution = CASBOT02_PARKOUR_HFIELD_RESOLUTION
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.random_poses < 0:
        parser.error("--random-poses must be nonnegative")
    source = (
        mujoco.MjModel.from_binary_path(str(args.source_model))
        if args.source_model
        else get_casbot02_parkour_spec().compile()
    )
    report = check_contacts(source, args.resolution, args.device, args.random_poses)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[result] {report['overflow_warnings']} overflow warnings; {args.report}", flush=True)
    passed = report["overflow_warnings"] > 0 if args.expect_overflow else report["overflow_warnings"] == 0
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
