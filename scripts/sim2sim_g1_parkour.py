#!/usr/bin/env python3
"""Run InstinctMJ G1 parkour ONNX policies in standalone CPU MuJoCo.

Reference workflow: https://github.com/liuchenxu125/roboparty_train
(robolab/scripts/mujoco/sim2sim_rpo_parkour.py).
This implementation uses InstinctMJ's G1 model, observations and PD settings.
It does not import Isaac Lab, start a training environment, or load motion data.

Examples:
  uv run python scripts/sim2sim_g1_parkour.py --model-dir <run>/exported
  uv run python scripts/sim2sim_g1_parkour.py --terrain stairs --depth-preview
  uv run python scripts/sim2sim_g1_parkour.py --headless --duration 10 --command 0.5 0 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "Instinct-Parkour-Target-Amp-G1-v0"
COMPONENTS = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")


@dataclass
class Robot:
    model: mujoco.MjModel
    joint_names: tuple[str, ...]
    qpos_ids: np.ndarray
    qvel_ids: np.ndarray
    position_ctrl_ids: np.ndarray
    velocity_ctrl_ids: np.ndarray
    default_pos: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    action_clip: np.ndarray | None
    root_body_id: int
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    env_cfg: object

    @property
    def policy_dt(self) -> float:
        return self.model.opt.timestep * self.env_cfg.decimation


def resolve_values(values, names, default: float) -> np.ndarray:
    from mjlab.utils.lab_api.string import resolve_matching_names_values

    if isinstance(values, (int, float)):
        return np.full(len(names), values, dtype=np.float64)
    result = np.full(len(names), default, dtype=np.float64)
    ids, _, matched = resolve_matching_names_values(values, names)
    result[ids] = matched
    return result


def add_terrain(spec: mujoco.MjSpec, args) -> float:
    """Return spawn surface height; stair heights are measured from z=0."""
    ground = spec.worldbody.add_geom(name="sim2sim_floor", type=mujoco.mjtGeom.mjGEOM_PLANE)
    ground.size = [0, 0, 1]
    ground.rgba = [0.22, 0.25, 0.29, 1]
    ground.group = 0
    ground.friction = [0.9, 0.005, 0.0001]

    def box(name: str, x: float, length: float, height: float):
        geom = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX)
        geom.pos = [x + length / 2, 0, height / 2]
        geom.size = [length / 2, args.stairs_width / 2, height / 2]
        geom.group = 0
        geom.friction = [0.9, 0.005, 0.0001]
        geom.rgba = [0.55, 0.60, 0.66, 1]

    n, h, w, start = args.num_stairs, args.step_height, args.step_width, args.stairs_start
    if args.terrain in ("stairs", "up"):
        for i in range(n):
            box(f"stair_up_{i}", start + i * w, w, (i + 1) * h)
        top = start + n * w
        box("stair_platform", top, args.platform_length, n * h)
        if args.terrain == "stairs":
            for i in range(n - 1):
                box(f"stair_down_{i}", top + args.platform_length + i * w, w, (n - i - 1) * h)
    elif args.terrain == "down":
        box("spawn_platform", -2, start + 2, n * h)
        for i in range(n - 1):
            box(f"stair_down_{i}", start + i * w, w, (n - i - 1) * h)
        return n * h
    elif args.terrain == "obstacles":
        for i, height in enumerate((0.08, 0.12, 0.18, 0.23)):
            box(f"obstacle_{i}", start + i * 1.3, 0.45, height)
    return 0.0


def build_robot(args) -> Robot:
    from mjlab.entity import Entity
    from mjlab.utils.lab_api.string import resolve_matching_names_values

    from instinct_mj.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg

    cfg = instinct_g1_parkour_amp_final_cfg(play=False)
    entity = Entity(cfg.scene.entities["robot"])
    action_cfg = cfg.actions["joint_pos"]
    _, names = entity.find_joints_by_actuator_names(action_cfg.actuator_names)
    names = tuple(names)
    if names != tuple(entity.joint_names) or len(names) != 29:
        raise ValueError("This runner requires the 29-joint G1 parkour action/observation order.")
    spawn_height = add_terrain(entity.spec, args)
    model = entity.spec.compile()
    # Compilation defaults to Euler, whereas training uses implicitfast.
    # Apply every physics option through the same path as mjlab Simulation.
    cfg.sim.mujoco.apply(model)
    key = model.key("init_state")
    initial_qpos, initial_qvel = key.qpos.copy(), key.qvel.copy()
    initial_qpos[2] += spawn_height
    initial_qpos[3:7] = [math.cos(args.spawn_yaw / 2), 0, 0, math.sin(args.spawn_yaw / 2)]
    joints = [model.joint(name) for name in names]
    qpos_ids = np.array([j.qposadr[0] for j in joints], dtype=np.int32)
    qvel_ids = np.array([j.dofadr[0] for j in joints], dtype=np.int32)
    default_pos = key.qpos[qpos_ids].copy()
    position_ids = np.array([model.actuator(name + "_pd_pos").id for name in names])
    velocity_ids = np.array([model.actuator(name + "_pd_vel").id for name in names])
    if model.nu != 58 or len(set(position_ids) | set(velocity_ids)) != 58:
        raise ValueError("Expected 29 pairs of position/velocity PD controls.")
    clip = None
    if action_cfg.clip is not None:
        clip = np.tile([-np.inf, np.inf], (len(names), 1))
        ids, _, values = resolve_matching_names_values(action_cfg.clip, names)
        clip[ids] = values
    offset = default_pos if action_cfg.use_default_offset else resolve_values(action_cfg.offset, names, 0)
    return Robot(
        model,
        names,
        qpos_ids,
        qvel_ids,
        position_ids,
        velocity_ids,
        default_pos,
        resolve_values(action_cfg.scale, names, 1),
        offset,
        clip,
        model.body("torso_link").id,
        initial_qpos,
        initial_qvel,
        cfg,
    )


def rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    result = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=np.float64))
    return result.reshape(3, 3)


class DepthCamera:
    def __init__(self, robot: Robot):
        self.cfg = next(s for s in robot.env_cfg.scene.sensors if s.name == "camera")
        self.model = robot.model
        self.body_id = robot.model.body(self.cfg.frame.name).id
        if self.cfg.offset.convention != "world":
            raise ValueError("Expected camera +X-forward, +Z-up world convention.")
        self.offset_pos = np.array(self.cfg.offset.pos, dtype=np.float64)
        self.offset_rotation = rotation_matrix(np.array(self.cfg.offset.rot))
        h, w = self.cfg.pattern.height, self.cfg.pattern.width
        self.shape = (h, w)
        fx = w * self.cfg.focal_length / self.cfg.horizontal_aperture
        fy = h * self.cfg.focal_length / self.cfg.vertical_aperture
        cx = w / 2 + self.cfg.horizontal_aperture_offset * fx
        cy = h / 2 + self.cfg.vertical_aperture_offset * fy
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays = np.stack([np.ones((h, w)), -(u - cx) / fx, -(v - cy) / fy], axis=-1)
        self.directions = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        self.radial_distances = np.empty(h * w, dtype=np.float64)
        self.geom_ids = np.empty(h * w, dtype=np.int32)
        self.geom_groups = None
        if self.cfg.include_geom_groups is not None:
            self.geom_groups = np.zeros(6, dtype=np.uint8)
            self.geom_groups[list(self.cfg.include_geom_groups)] = 1
        self.exclude_body = self.body_id if self.cfg.exclude_parent_body else -1
        self.crop = self.cfg.noise_pipeline["crop_and_resize"].crop_region
        self.output_shape = (h - self.crop[0] - self.crop[1], w - self.crop[2] - self.crop[3])
        if self.output_shape != (18, 32):
            raise ValueError(f"Expected cropped depth 18x32, got {self.output_shape}.")

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        body_rotation = data.xmat[self.body_id].reshape(3, 3)
        return (data.xpos[self.body_id] + body_rotation @ self.offset_pos, body_rotation @ self.offset_rotation)

    def capture(self, data: mujoco.MjData) -> np.ndarray:
        origin, rotation = self.pose(data)
        directions = np.ascontiguousarray((self.directions @ rotation.T).reshape(-1, 3))
        mujoco.mj_multiRay(
            self.model,
            data,
            origin,
            directions.ravel(),
            self.geom_groups,
            True,
            self.exclude_body,
            self.geom_ids,
            self.radial_distances,
            None,
            len(directions),
            self.cfg.max_distance,
        )
        # Skip hits inside the near range, matching GroupedRayCaster's continuation.
        for i in np.flatnonzero((self.radial_distances >= 0) & (self.radial_distances < self.cfg.min_distance)):
            travel = self.radial_distances[i]
            for _ in range(self.cfg.mesh_filter_max_hops):
                travel += self.cfg.mesh_filter_epsilon
                hit = np.empty(1, dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.model,
                    data,
                    origin + travel * directions[i],
                    directions[i],
                    self.geom_groups,
                    True,
                    self.exclude_body,
                    hit,
                )
                if distance < 0:
                    travel = -1
                    break
                travel += distance
                if travel >= self.cfg.min_distance:
                    break
            self.radial_distances[i] = travel if travel >= self.cfg.min_distance else -1
        valid = (self.radial_distances >= 0) & (self.radial_distances <= self.cfg.max_distance)
        depth = np.full(len(directions), self.cfg.max_distance, dtype=np.float64)
        depth[valid] = self.radial_distances[valid] * self.directions.reshape(-1, 3)[valid, 0]
        return depth.reshape(self.shape)

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        import cv2

        h, w = raw.shape
        top, bottom, left, right = self.crop
        image = raw[top : h - bottom, left : w - right].astype(np.float32)
        image = np.nan_to_num(image, nan=self.cfg.max_distance, posinf=self.cfg.max_distance, neginf=0)
        blur = self.cfg.noise_pipeline["gaussian_blur"]
        image = cv2.GaussianBlur(
            image, (blur.kernel_size, blur.kernel_size), blur.sigma, borderType=cv2.BORDER_REFLECT_101
        )
        normalization = self.cfg.noise_pipeline["depth_normalization"]
        low, high = normalization.depth_range
        image = np.clip(image, low, high)
        if normalization.normalize:
            out_low, out_high = normalization.output_range
            image = (image - low) / (high - low) * (out_high - out_low) + out_low
        return image.astype(np.float32)


class Histories:
    """Per-component chronological history, as used by ObservationManager."""

    def __init__(self, robot: Robot, camera: DepthCamera, depth_delay: int):
        terms = robot.env_cfg.observations["policy"].terms
        if tuple(terms) != (*COMPONENTS, "depth_image"):
            raise ValueError("Unsupported policy observation order.")
        self.proprio = {name: deque(maxlen=terms[name].history_length) for name in COMPONENTS}
        self.depth = deque(maxlen=camera.cfg.data_histories["distance_to_image_plane_noised"])
        visual = terms["depth_image"].params
        self.frame_indices = (
            self.depth.maxlen
            - 1
            - depth_delay
            - np.arange(visual["num_output_frames"] - 1, -1, -1) * visual["history_skip_frames"]
        )
        if self.frame_indices.min() < 0:
            raise ValueError("Depth delay exceeds camera history capacity.")
        self.proprio_dim = sum(
            self.proprio[name].maxlen * (29 if name in ("joint_pos", "joint_vel", "actions") else 3)
            for name in COMPONENTS
        )
        self.depth_shape = (len(self.frame_indices), *camera.output_shape)

    def reset(self, obs: dict, depth: np.ndarray):
        for name, history in self.proprio.items():
            history.clear()
            history.extend(obs[name].copy() for _ in range(history.maxlen))
        self.depth.clear()
        self.depth.extend(depth.copy() for _ in range(self.depth.maxlen))

    def append(self, obs: dict, depth: np.ndarray):
        for name, history in self.proprio.items():
            history.append(obs[name].copy())
        self.depth.append(depth.copy())

    def inputs(self) -> tuple[np.ndarray, np.ndarray]:
        proprio = np.concatenate([np.concatenate(self.proprio[name]) for name in COMPONENTS]).astype(np.float32)
        frames = list(self.depth)
        return proprio, np.stack([frames[i] for i in self.frame_indices]).astype(np.float32)


def observations(robot: Robot, data: mujoco.MjData, command: np.ndarray, last_action: np.ndarray) -> dict:
    velocity = np.empty(6, dtype=np.float64)
    # The XML IMU is in the pelvis; policy angular velocity belongs to torso_link.
    mujoco.mj_objectVelocity(robot.model, data, mujoco.mjtObj.mjOBJ_BODY, robot.root_body_id, velocity, True)
    rotation = data.xmat[robot.root_body_id].reshape(3, 3)
    values = {
        "base_ang_vel": velocity[:3],
        "projected_gravity": rotation.T @ np.array([0, 0, -1.0]),
        "velocity_commands": command,
        "joint_pos": data.qpos[robot.qpos_ids] - robot.default_pos,
        "joint_vel": data.qvel[robot.qvel_ids] - robot.initial_qvel[robot.qvel_ids],
        "actions": last_action,
    }
    terms = robot.env_cfg.observations["policy"].terms
    return {
        name: (value * (1 if terms[name].scale is None else terms[name].scale)).astype(np.float32)
        for name, value in values.items()
    }


def resolve_model_dir(path: Path | None) -> Path:
    if path is not None:
        path = path.expanduser().resolve()
        if (path / "exported").is_dir():
            path = path / "exported"
        return path
    root = REPO_ROOT / "logs/instinct_rl/g1_parkour"
    candidates = sorted(root.glob("*/exported"), reverse=True)
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("actor.onnx", "0-depth_encoder.onnx")):
            return candidate
    raise FileNotFoundError(
        "No exported G1 policy found. Export a checkpoint with instinct-play --export-onnx, then pass --model-dir."
    )


class OnnxPolicy:
    def __init__(self, path: Path, histories: Histories):
        import onnxruntime as ort

        missing = [name for name in ("actor.onnx", "0-depth_encoder.onnx") if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing exported ONNX files in {path}: {', '.join(missing)}")
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("num_actions") != 29 or metadata.get("task_id", "").replace("-Play", "") != TASK_ID:
                raise ValueError("ONNX metadata is not from this InstinctMJ G1 parkour task.")
            segments = metadata.get("obs_format", {}).get("policy", {})
            expected = {
                name: [29 * 8 if name in ("joint_pos", "joint_vel", "actions") else 3 * 8] for name in COMPONENTS
            }
            expected["depth_image"] = list(histories.depth_shape)
            if segments != expected or list(segments) != list(expected):
                raise ValueError(f"Policy observation metadata mismatch: {segments} != {expected}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(path / "0-depth_encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.actor = ort.InferenceSession(
            str(path / "actor.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.validate(self.encoder, histories.depth_shape, "depth encoder input")
        encoded_shape = self.encoder.get_outputs()[0].shape
        if len(encoded_shape) != 2 or not isinstance(encoded_shape[1], int):
            raise ValueError(f"Expected encoder output [batch, latent_size], got {encoded_shape}")
        self.validate(self.actor, (histories.proprio_dim + encoded_shape[1],), "actor input")
        if self.actor.get_outputs()[0].shape[1:] != [29]:
            raise ValueError(
                "Expected actor output [batch, 29]; other robots/Isaac joint orders need separate policies."
            )
        print(f"[policy] {path}\n[policy] depth={histories.depth_shape}, proprio={histories.proprio_dim}, actions=29")

    @staticmethod
    def validate(session, shape: tuple, label: str):
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError(f"Expected a single float32 {label}.")
        actual = inputs[0].shape
        if len(actual) != len(shape) + 1 or any(isinstance(a, int) and a != e for a, e in zip(actual[1:], shape)):
            raise ValueError(f"Incorrect {label}: {actual}, expected [batch, {shape}].")

    @staticmethod
    def run_single(session, value: np.ndarray) -> np.ndarray:
        info = session.get_inputs()[0]
        batch = info.shape[0] if isinstance(info.shape[0], int) and info.shape[0] > 0 else 1
        tensor = np.repeat(value[None].astype(np.float32), batch, axis=0)
        return session.run(None, {info.name: tensor})[0][0]

    def __call__(self, proprio: np.ndarray, depth: np.ndarray) -> np.ndarray:
        latent = self.run_single(self.encoder, depth)
        action = self.run_single(self.actor, np.concatenate([proprio, latent]))
        if action.shape != (29,) or not np.isfinite(action).all():
            raise RuntimeError("ONNX produced invalid G1 actions.")
        return action.astype(np.float32)


class Controls:
    """Incremental key presses work with MuJoCo's built-in passive viewer."""

    def __init__(self, command):
        self.target = np.array(command, dtype=np.float64)
        self.command = self.target.copy()
        self.reset_requested = False
        self.follow = True
        self.chase = False

    def key(self, key: int):
        deltas = {
            ord("W"): (0, 0.1),
            ord("S"): (0, -0.1),
            ord("A"): (1, 0.1),
            ord("D"): (1, -0.1),
            ord("Q"): (2, 0.1),
            ord("E"): (2, -0.1),
        }
        if key in deltas:
            axis, delta = deltas[key]
            self.target[axis] += delta
            self.target[:] = np.clip(self.target, [-0.8, -0.6, -1], [0.8, 0.6, 1])
        elif key == ord(" "):
            self.target[:] = 0
        elif key == ord("R"):
            self.reset_requested = True
        elif key == ord("F"):
            self.follow = not self.follow
        elif key == ord("C"):
            self.chase = not self.chase
            self.follow = True

    def advance(self, dt: float):
        change = np.clip(self.target - self.command, -np.array([2, 2, 3]) * dt, np.array([2, 2, 3]) * dt)
        self.command += change


def update_viewer(viewer, data, robot: Robot, controls: Controls):
    with viewer.lock():
        if controls.follow:
            viewer.cam.lookat[:] = data.xpos[robot.root_body_id]
            if controls.chase:
                forward = data.xmat[robot.root_body_id].reshape(3, 3)[:, 0]
                viewer.cam.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
                viewer.cam.elevation = -20
                viewer.cam.distance = 3
    viewer.sync()


def preview_depth(raw: np.ndarray, processed: np.ndarray, camera: DepthCamera, controls: Controls):
    import cv2

    far = camera.cfg.max_distance
    full = cv2.applyColorMap(np.uint8(np.clip(raw / far, 0, 1) * 255), cv2.COLORMAP_TURBO)
    small = cv2.applyColorMap(np.uint8(np.clip(processed, 0, 1) * 255), cv2.COLORMAP_TURBO)
    top, bottom, left, right = camera.crop
    cv2.rectangle(full, (left, top), (raw.shape[1] - right - 1, raw.shape[0] - bottom - 1), (255, 255, 255), 1)
    full = cv2.resize(full, (384, 216), interpolation=cv2.INTER_NEAREST)
    small = cv2.resize(small, (384, 216), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("G1 depth: raw / policy crop", np.concatenate([full, small], axis=1))
    key = cv2.waitKey(1)
    if 0 <= key < 256:
        controls.key(ord(chr(key).upper()))


def run(args) -> dict:
    if args.headless and args.depth_preview:
        raise ValueError("--depth-preview requires a display; omit it for --headless.")
    robot = build_robot(args)
    camera = DepthCamera(robot)
    histories = Histories(robot, camera, args.depth_delay)
    policy = None if args.zero_policy else OnnxPolicy(resolve_model_dir(args.model_dir), histories)
    if policy is None:
        print("[policy] ZERO ACTIONS: checking simulation only, no trained policy loaded.")
    controls = Controls(args.command)
    data = mujoco.MjData(robot.model)
    action = np.zeros(29, dtype=np.float32)

    def reset():
        mujoco.mj_resetData(robot.model, data)
        data.qpos[:] = robot.initial_qpos
        data.qvel[:] = robot.initial_qvel
        action[:] = 0
        data.ctrl[robot.position_ctrl_ids] = robot.action_offset
        data.ctrl[robot.velocity_ctrl_ids] = robot.initial_qvel[robot.qvel_ids]
        mujoco.mj_forward(robot.model, data)
        raw = camera.capture(data)
        histories.reset(observations(robot, data, controls.command, action), camera.preprocess(raw))
        controls.reset_requested = False

    reset()
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(robot.model, data, key_callback=controls.key)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4, 90, -20
        print("[keys] W/S=vx, A/D=vy, Q/E=yaw (0.1 increments); Space=stop; R=reset; F=follow; C=chase")
    print(
        f"[sim] terrain={args.terrain}, physics={1/robot.model.opt.timestep:g}Hz, policy={1/robot.policy_dt:g}Hz,"
        f" depth_delay={args.depth_delay}"
    )
    records = {
        name: []
        for name in ("time", "command", "base_velocity", "base_pos", "joint_pos", "target_pos", "action", "joint_force")
    }
    step, resets, start = 0, 0, time.perf_counter()
    try:
        while args.duration == 0 or step * robot.policy_dt < args.duration - 1e-9:
            if viewer is not None and not viewer.is_running():
                break
            if controls.reset_requested:
                reset()
                resets += 1
            controls.advance(robot.policy_dt)
            proprio, depth = histories.inputs()
            # Current command replaces the newest frame, without appending a duplicate state.
            command_end = sum(
                histories.proprio[name].maxlen * len(histories.proprio[name][0]) for name in COMPONENTS[:3]
            )
            proprio[command_end - 3 : command_end] = controls.command
            action[:] = 0 if policy is None else policy(proprio, depth)
            targets = robot.action_offset + action * robot.action_scale
            if robot.action_clip is not None:
                targets = np.clip(targets, robot.action_clip[:, 0], robot.action_clip[:, 1])
            for _ in range(robot.env_cfg.decimation):
                data.ctrl[robot.position_ctrl_ids] = targets
                data.ctrl[robot.velocity_ctrl_ids] = robot.initial_qvel[robot.qvel_ids]
                mujoco.mj_step(robot.model, data)
            # mj_step leaves derived poses at the preceding substep; sensing requires fresh poses.
            mujoco.mj_forward(robot.model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Nonfinite MuJoCo state.")
            raw = camera.capture(data)
            processed = camera.preprocess(raw)
            histories.append(observations(robot, data, controls.command, action), processed)
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(robot.model, data, mujoco.mjtObj.mjOBJ_BODY, robot.root_body_id, velocity, True)
            if args.log_file is not None:
                values = (
                    (step + 1) * robot.policy_dt,
                    controls.command.copy(),
                    np.r_[velocity[3:5], velocity[2]],
                    data.xpos[robot.root_body_id].copy(),
                    data.qpos[robot.qpos_ids].copy(),
                    targets.copy(),
                    action.copy(),
                    data.qfrc_actuator[robot.qvel_ids].copy(),
                )
                for key, value in zip(records, values):
                    records[key].append(value)
            if step % max(1, round(1 / robot.policy_dt)) == 0:
                print(
                    f"t={step*robot.policy_dt:6.2f}s"
                    f" cmd={controls.command.round(2)} vel={np.r_[velocity[3:5],velocity[2]].round(2)} z={data.qpos[2]:.3f}"
                )
            if viewer is not None:
                update_viewer(viewer, data, robot, controls)
            if args.depth_preview and step % 2 == 0:
                preview_depth(raw, processed, camera, controls)
            step += 1
            if not args.no_realtime:
                remaining = start + step * robot.policy_dt - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if args.depth_preview:
            import cv2

            cv2.destroyAllWindows()
        if args.log_file is not None:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.log_file, joint_names=np.array(robot.joint_names), **records)
            print(f"[log] {args.log_file}")
    summary = {
        "policy_steps": step,
        "sim_seconds": round(step * robot.policy_dt, 3),
        "resets": resets,
        "final_base_pos": data.xpos[robot.root_body_id].round(4).tolist(),
        "wall_seconds": round(time.perf_counter() - start, 3),
    }
    print("[done] " + json.dumps(summary))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="InstinctMJ G1 parkour sim2sim (29 joints, depth encoder + actor ONNX)."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Exported ONNX directory or training run directory; defaults to latest g1_parkour export.",
    )
    parser.add_argument("--terrain", choices=("flat", "stairs", "up", "down", "obstacles"), default="stairs")
    parser.add_argument("--step-height", type=float, default=0.15)
    parser.add_argument("--step-width", type=float, default=0.35)
    parser.add_argument("--num-stairs", type=int, default=6)
    parser.add_argument("--stairs-width", type=float, default=2)
    parser.add_argument("--stairs-start", type=float, default=1.5)
    parser.add_argument("--platform-length", type=float, default=1.2)
    parser.add_argument("--spawn-yaw", type=float, default=0, help="Initial heading in radians.")
    parser.add_argument("--command", type=float, nargs=3, default=(0, 0, 0), metavar=("VX", "VY", "WZ"))
    parser.add_argument(
        "--depth-delay",
        type=int,
        choices=(0, 1),
        default=0,
        help="Fixed camera delay in policy frames, within training's randomized range.",
    )
    parser.add_argument("--duration", type=float, default=60, help="Simulation seconds; 0 runs until closed.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true", help="Do not wait for wall-clock synchronization.")
    parser.add_argument("--depth-preview", action="store_true")
    parser.add_argument("--log-file", type=Path, help="Optional .npz output with joint/velocity tracking data.")
    parser.add_argument(
        "--zero-policy",
        action="store_true",
        help="Test physics and observations using zero actions, without ONNX files.",
    )
    args = parser.parse_args(argv)
    for name in ("step_height", "step_width", "stairs_width", "stairs_start", "platform_length"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_','-')} must be finite and positive.")
    if args.num_stairs < 1 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--num-stairs must be >=1 and --duration must be finite and >=0.")
    if not np.isfinite(args.command).all() or not math.isfinite(args.spawn_yaw):
        parser.error("Command and spawn yaw must be finite.")
    if args.log_file is not None and args.log_file.suffix != ".npz":
        parser.error("--log-file must end in .npz.")
    return args


if __name__ == "__main__":
    run(parse_args())
