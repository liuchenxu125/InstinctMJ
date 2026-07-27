#!/usr/bin/env python3
"""CASBOT_02 23-DOF parkour sim2sim with ONNX inference + joystick control.

Usage:
  # Auto-find latest exported ONNX
  uv run python scripts/sim2sim_casbot02_parkour.py

  # Specify model directory
  uv run python scripts/sim2sim_casbot02_parkour.py --model-dir <exported_dir>

  # Keyboard mode (no joystick)
  uv run python scripts/sim2sim_casbot02_parkour.py --keyboard

Controls:
  Joystick: left stick = vx/vy, right stick x = yaw rate
  Keyboard: 8/2 = vx, 4/6 = vy, 7/9 = yaw, 0 = reset, Space = stop
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from instinct_mj.assets.casbot_02 import (
    casbot02_23dof_delayed_actuator_cfgs,
    casbot02_action_scale,
    CASBOT02_INIT_STATE,
    get_casbot02_spec,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1] if __file__ != "__main__" else Path.cwd()
XML_PATH = (
    REPO_ROOT
    / "src/instinct_mj/assets/resources/casbot_23dof/xml/CASBOT_02_23dof.xml"
)
DEFAULT_ONNX_DIR = (
    REPO_ROOT
    / "logs/instinct_rl/casbot02_parkour/2026-07-26_18-19-10/exported"
)

# ---------------------------------------------------------------------------
# Joint definitions (CASBOT 23-DOF, must match policy output order)
# ---------------------------------------------------------------------------
JOINT_NAMES = (
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
    "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
    "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "waist_yaw_joint",
    "upper_left_1_joint", "upper_left_2_joint", "upper_left_3_joint",
    "upper_left_4_joint", "upper_left_5_joint",
    "upper_right_1_joint", "upper_right_2_joint", "upper_right_3_joint",
    "upper_right_4_joint", "upper_right_5_joint",
)
NUM_ACTIONS = len(JOINT_NAMES)  # 23

# Default joint positions from training config
DEFAULT_JOINT_POS = np.array(
    [CASBOT02_INIT_STATE.joint_pos.get(n, 0.0) for n in JOINT_NAMES],
    dtype=np.float64,
)

def make_action_scale() -> np.ndarray:
    """action_scale per joint, matching training casbot02_action_scale."""
    return np.array([casbot02_action_scale[n] for n in JOINT_NAMES], dtype=np.float64)

# ---------------------------------------------------------------------------
# Simulation parameters (matching parkour training config)
# ---------------------------------------------------------------------------
SIM_TIMESTEP = 0.005
DECIMATION = 4
POLICY_DT = SIM_TIMESTEP * DECIMATION  # 0.02s = 50 Hz
HISTORY_LENGTH = 8  # obs history frames

# Per-frame proprioceptive dims (must match policy obs group)
# base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
# + joint_pos(23) + joint_vel(23) + actions(23) = 78
SINGLE_FRAME_PROPRIO_DIM = 3 + 3 + 3 + NUM_ACTIONS + NUM_ACTIONS + NUM_ACTIONS  # 78
FLAT_PROPRIO_DIM = HISTORY_LENGTH * SINGLE_FRAME_PROPRIO_DIM  # 624

# Depth encoder
DEPTH_ENCODER_OUTPUT = 128
ACTOR_INPUT_DIM = FLAT_PROPRIO_DIM + DEPTH_ENCODER_OUTPUT  # 752

# ---------------------------------------------------------------------------
# Depth camera (matching NoisyGroupedRayCasterCameraCfg)
# ---------------------------------------------------------------------------
CAMERA_BODY = "torso"
CAMERA_OFFSET_POS = np.array([0.05, 0.0, 0.45], dtype=np.float64)
CAMERA_OFFSET_QUAT_WXYZ = np.array(
    [0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0],
    dtype=np.float64,
)
_FOV_Y_DEG = 58.29
_FOV_X_DEG = 89.51
_RAW_DEPTH_H, _RAW_DEPTH_W = 36, 64
_ENCODER_H, _ENCODER_W = 18, 32
_CROP_UP, _CROP_DOWN, _CROP_LEFT, _CROP_RIGHT = 18, 0, 16, 16
_DEPTH_CLIP_MIN, _DEPTH_CLIP_MAX = 0.0, 2.5
_DEPTH_HISTORY_LEN = 37
# Indices into the 37-frame history buffer (oldest=0, newest=36)
_DEPTH_FRAME_INDICES = np.array([1, 6, 11, 16, 21, 26, 31, 36], dtype=np.int64)

# ---------------------------------------------------------------------------
# Joystick
# ---------------------------------------------------------------------------
JOYSTICK_DEV = "/dev/input/js0"


def build_model() -> mujoco.MjModel:
    """Build CASBOT_02 model with properly configured position actuators via mjlab Entity."""
    import copy

    # Build EntityCfg matching the parkour training setup
    entity_cfg = EntityCfg(
        init_state=copy.deepcopy(CASBOT02_INIT_STATE),
        spec_fn=get_casbot02_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=copy.deepcopy(casbot02_23dof_delayed_actuator_cfgs),
            soft_joint_pos_limit_factor=0.9,
        ),
        sort_actuators=True,
    )

    entity = Entity(entity_cfg)
    spec = entity.spec

    # Remove any existing ground, add flat ground plane
    for geom in list(spec.geoms):
        name = getattr(geom, "name", "") or ""
        if "ground" in str(name).lower():
            spec.delete(geom)

    ground = spec.worldbody.add_geom(
        name="sim2sim_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
    )
    ground.size = [0.0, 0.0, 1.0]
    ground.condim = 4
    ground.friction = [0.9, 0.2, 0.2]

    model = spec.compile()
    model.opt.timestep = SIM_TIMESTEP
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    return model


def build_joint_layout(model: mujoco.MjModel) -> dict:
    """Map policy-order joint names to MuJoCo qpos/qvel/ctrl indices."""
    qpos_ids = np.empty(NUM_ACTIONS, dtype=np.int32)
    qvel_ids = np.empty(NUM_ACTIONS, dtype=np.int32)
    ctrl_ids = np.empty(NUM_ACTIONS, dtype=np.int32)

    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint '{name}' not found in model")
        qpos_ids[i] = model.jnt_qposadr[jid]
        qvel_ids[i] = model.jnt_dofadr[jid]

        # Find actuator for this joint
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            raise RuntimeError(f"Actuator '{name}' not found in model")
        ctrl_ids[i] = act_id

    free_qpos_dim = 7  # freejoint: x,y,z,qw,qx,qy,qz
    free_qvel_dim = 6

    return {
        "qpos_ids": qpos_ids,
        "qvel_ids": qvel_ids,
        "ctrl_ids": ctrl_ids,
        "free_qpos_dim": free_qpos_dim,
        "free_qvel_dim": free_qvel_dim,
        "ctrl_lo": model.actuator_ctrlrange[ctrl_ids, 0].copy(),
        "ctrl_hi": model.actuator_ctrlrange[ctrl_ids, 1].copy(),
    }


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def quat_apply_inverse_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector v into body frame using quaternion inverse."""
    w, x, y, z = q_wxyz
    qv = np.array([-x, -y, -z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_apply_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (body-frame → world-frame)."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


# ---------------------------------------------------------------------------
# Depth camera (MuJoCo ray-cast, matching NoisyGroupedRayCasterCameraCfg)
# ---------------------------------------------------------------------------
def _build_ray_dirs(height, width, fovy_deg, fovx_deg) -> np.ndarray:
    """Unit ray directions in camera frame (+X forward, +Z up). Shape (H, W, 3)."""
    fovy, fovx = np.radians(fovy_deg), np.radians(fovx_deg)
    u = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    v = (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0
    uu, vv = np.meshgrid(u, v)
    x = np.ones_like(uu)
    y = -uu * np.tan(fovx / 2.0)
    z = -vv * np.tan(fovy / 2.0)
    d = np.stack([x, y, z], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


class DepthCamera:
    """Ray-cast depth camera mounted on torso, matching training pipeline."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.dirs_cam = _build_ray_dirs(_RAW_DEPTH_H, _RAW_DEPTH_W, _FOV_Y_DEG, _FOV_X_DEG)
        self.forward_components = self.dirs_cam[..., 0].copy()  # projection onto +X

    def capture(self, data: mujoco.MjData) -> np.ndarray:
        """Capture raw 64×36 depth (distance_to_image_plane), return (H, W) float64."""
        # Camera world pose
        base_pos = data.qpos[0:3].copy()
        base_quat = data.qpos[3:7].copy()  # wxyz

        cam_pos = base_pos + quat_apply_wxyz(base_quat, CAMERA_OFFSET_POS)
        cam_quat = quat_multiply_wxyz(base_quat, CAMERA_OFFSET_QUAT_WXYZ)

        # Rotate ray dirs from camera frame to world frame
        r_world = np.zeros((3, 3), dtype=np.float64)
        mujoco.mju_quat2Mat(r_world.reshape(-1), cam_quat)
        dirs_world = self.dirs_cam @ r_world.T  # (H, W, 3)

        depth = np.full((_RAW_DEPTH_H, _RAW_DEPTH_W), _DEPTH_CLIP_MAX, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)

        for j in range(_RAW_DEPTH_H):
            for i in range(_RAW_DEPTH_W):
                dist = mujoco.mj_ray(
                    self.model, data, cam_pos, dirs_world[j, i],
                    None, 1, -1, geomid,
                )
                if dist is not None and dist >= 0.0:
                    # Convert radial distance to distance-to-image-plane
                    depth[j, i] = min(
                        float(dist) * float(self.forward_components[j, i]),
                        _DEPTH_CLIP_MAX,
                    )
        return depth


def preprocess_depth(raw_depth: np.ndarray) -> np.ndarray:
    """64×36 raw → crop → blur → clip → normalize → (18, 32) float32 [0, 1]."""
    # Crop
    h, w = raw_depth.shape
    cropped = raw_depth[
        _CROP_UP : h - _CROP_DOWN,
        _CROP_LEFT : w - _CROP_RIGHT,
    ].copy()

    # Ensure exact size
    if cropped.shape != (_ENCODER_H, _ENCODER_W):
        import cv2
        cropped = cv2.resize(cropped, (_ENCODER_W, _ENCODER_H), interpolation=cv2.INTER_AREA)

    # Replace NaN/Inf
    cropped = np.nan_to_num(cropped, nan=_DEPTH_CLIP_MAX, posinf=_DEPTH_CLIP_MAX, neginf=0.0)

    # Gaussian blur 3×3 σ=1
    # Simple manual implementation to avoid cv2 dependency for just blur
    kernel = np.exp(-0.5 * np.square(np.arange(-1, 2, dtype=np.float64)))
    kernel /= kernel.sum()
    kernel_2d = np.outer(kernel, kernel)
    padded = np.pad(cropped, ((1, 1), (1, 1)), mode="reflect")
    result = np.zeros_like(cropped)
    for r in range(3):
        for c in range(3):
            result += kernel_2d[r, c] * padded[r:r+_ENCODER_H, c:c+_ENCODER_W]

    # Clip and normalize
    result = np.clip(result, _DEPTH_CLIP_MIN, _DEPTH_CLIP_MAX)
    result = result / (_DEPTH_CLIP_MAX - _DEPTH_CLIP_MIN)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Observation buffer
# ---------------------------------------------------------------------------
class ObsHistory:
    """Ring buffer for proprioceptive observation history."""

    COMPONENTS = (
        "base_ang_vel", "projected_gravity", "velocity_commands",
        "joint_pos", "joint_vel", "actions",
    )

    def __init__(self):
        self.buffers: dict[str, deque] = {
            name: deque(maxlen=HISTORY_LENGTH) for name in self.COMPONENTS
        }

    def reset(self):
        for d in self.buffers.values():
            d.clear()

    def fill(self, values: dict[str, np.ndarray]):
        """Fill entire history with repeated frame."""
        for name in self.COMPONENTS:
            v = values[name].astype(np.float32).copy()
            self.buffers[name].clear()
            for _ in range(HISTORY_LENGTH):
                self.buffers[name].append(v.copy())

    def append(self, values: dict[str, np.ndarray]):
        for name in self.COMPONENTS:
            self.buffers[name].append(values[name].astype(np.float32).copy())

    def flatten(self) -> np.ndarray:
        parts = []
        for name in self.COMPONENTS:
            parts.append(np.concatenate(list(self.buffers[name])))
        result = np.concatenate(parts)
        assert result.shape == (FLAT_PROPRIO_DIM,), f"{result.shape} != ({FLAT_PROPRIO_DIM},)"
        return result.astype(np.float32)


class DepthHistory:
    """Ring buffer for depth frames."""

    def __init__(self):
        self.frames: deque = deque(maxlen=_DEPTH_HISTORY_LEN)

    def reset(self):
        self.frames.clear()

    def fill(self, frame: np.ndarray):
        self.frames.clear()
        for _ in range(_DEPTH_HISTORY_LEN):
            self.frames.append(frame.astype(np.float32).copy())

    def append(self, frame: np.ndarray):
        self.frames.append(frame.astype(np.float32).copy())

    def sample(self) -> np.ndarray:
        """Return (8, 18, 32) sampled frames for encoder input."""
        buf = list(self.frames)
        sampled = np.stack([buf[i] for i in _DEPTH_FRAME_INDICES], axis=0)
        assert sampled.shape == (8, _ENCODER_H, _ENCODER_W), f"{sampled.shape}"
        return sampled.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX policy
# ---------------------------------------------------------------------------
def load_onnx_policy(model_dir: Path):
    """Load depth encoder and actor ONNX models."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime required. Run: uv pip install onnxruntime")

    encoder_path = model_dir / "0-depth_encoder.onnx"
    actor_path = model_dir / "actor.onnx"
    for p in (encoder_path, actor_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    encoder = ort.InferenceSession(str(encoder_path), providers=providers)
    actor = ort.InferenceSession(str(actor_path), providers=providers)

    enc_in = encoder.get_inputs()[0]
    act_in = actor.get_inputs()[0]
    print(f"[ONNX] encoder: {enc_in.name} {enc_in.shape} → {encoder.get_outputs()[0].shape}")
    print(f"[ONNX] actor:   {act_in.name} {act_in.shape} → {actor.get_outputs()[0].shape}")
    print(f"[ONNX] providers: {encoder.get_providers()}")

    # Model exported with a fixed batch size (usually 10 from play mode)
    _enc_batch = encoder.get_inputs()[0].shape[0]
    _act_batch = actor.get_inputs()[0].shape[0]
    if isinstance(_enc_batch, int) and _enc_batch > 0:
        _sim_batch = _enc_batch
    elif isinstance(_act_batch, int) and _act_batch > 0:
        _sim_batch = _act_batch
    else:
        _sim_batch = 1
    print(f"[ONNX] model batch size: {_sim_batch}")

    def infer(proprio_flat: np.ndarray, depth_batch: np.ndarray) -> np.ndarray:
        """Run full inference: depth → encoder → concat → actor → action."""
        # Broadcast to model batch size
        depth_in = np.tile(
            depth_batch.reshape(1, 8, _ENCODER_H, _ENCODER_W).astype(np.float32),
            (_sim_batch, 1, 1, 1),
        )
        latent_all = encoder.run(None, {enc_in.name: depth_in})[0]  # (B, 128)

        actor_in = np.concatenate(
            [
                np.tile(proprio_flat.reshape(1, -1).astype(np.float32), (_sim_batch, 1)),
                latent_all,
            ],
            axis=1,
        )
        action_all = actor.run(None, {act_in.name: actor_in})[0]  # (B, 23)
        # Take first batch element
        return action_all[0].astype(np.float32)

    return infer


# ---------------------------------------------------------------------------
# Joystick (pygame)
# ---------------------------------------------------------------------------
class JoystickController:
    def __init__(self):
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            self.js = None
            count = pygame.joystick.get_count()
            if count > 0:
                self.js = pygame.joystick.Joystick(0)
                self.js.init()
                print(f"[joystick] {self.js.get_name()} ({self.js.get_numaxes()} axes, {self.js.get_numbuttons()} buttons)")
            else:
                print("[joystick] No joystick found, using keyboard fallback")
        except Exception as e:
            print(f"[joystick] pygame init failed: {e}")
            self.js = None

    def process_events(self) -> tuple[float, float, float, bool]:
        """Return (vx, vy, dyaw, reset)."""
        try:
            import pygame
            vx = vy = dyaw = 0.0
            reset = False

            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0:  # A button → reset
                        reset = True

            if self.js is not None:
                # Left stick: vx (axis 1, up/down), vy (axis 0, left/right)
                raw_vy = -self.js.get_axis(0)  # left positive
                raw_vx = -self.js.get_axis(1)  # up positive
                # Right stick x (axis 2): yaw rate
                raw_dyaw = -self.js.get_axis(2)

                # Dead zone
                dead = 0.1
                vx = raw_vx if abs(raw_vx) > dead else 0.0
                vy = raw_vy if abs(raw_vy) > dead else 0.0
                dyaw = raw_dyaw if abs(raw_dyaw) > dead else 0.0

                # Scale to command ranges
                vx *= 1.0    # max ±1.0 m/s
                vy *= 0.5    # max ±0.5 m/s
                dyaw *= 1.0  # max ±1.0 rad/s

            # Also poll keyboard events through pygame
            keys = pygame.key.get_pressed()
            if keys[pygame.K_8] or keys[pygame.K_UP]:
                vx = 0.8
            if keys[pygame.K_2] or keys[pygame.K_DOWN]:
                vx = -0.3
            if keys[pygame.K_4] or keys[pygame.K_LEFT]:
                vy = 0.3
            if keys[pygame.K_6] or keys[pygame.K_RIGHT]:
                vy = -0.3
            if keys[pygame.K_7]:
                dyaw = 1.0
            if keys[pygame.K_9]:
                dyaw = -1.0
            if keys[pygame.K_SPACE]:
                vx = vy = dyaw = 0.0
            if keys[pygame.K_0]:
                reset = True
            if keys[pygame.K_ESCAPE]:
                raise KeyboardInterrupt

            return vx, vy, dyaw, reset
        except KeyboardInterrupt:
            raise
        except Exception:
            return 0.0, 0.0, 0.0, False


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run(model_dir: Path, headless: bool = False):
    # ---- Build model & layout ----
    print("[model] Building CASBOT_02 MuJoCo model...")
    model = build_model()
    layout = build_joint_layout(model)
    data = mujoco.MjData(model)

    if model.nu != NUM_ACTIONS:
        print(f"[WARN] Model has {model.nu} actuators, policy expects {NUM_ACTIONS}")

    print(f"[model] nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={model.opt.timestep}")
    print(f"[model] policy Hz={1.0 / POLICY_DT:.1f}")

    # ---- Load policy ----
    print("[policy] Loading ONNX models...")
    policy_fn = load_onnx_policy(model_dir)
    action_scale = make_action_scale()
    ctrl_lo, ctrl_hi = layout["ctrl_lo"], layout["ctrl_hi"]

    # ---- Depth camera ----
    depth_cam = DepthCamera(model)
    depth_hist = DepthHistory()
    obs_hist = ObsHistory()

    # ---- Joystick ----
    joystick = JoystickController()

    # ---- Initial state ----
    def reset_env():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.92]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, name in enumerate(JOINT_NAMES):
            data.qpos[layout["qpos_ids"][i]] = DEFAULT_JOINT_POS[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # Fill observation buffers
        obs_frame = _get_obs_dict(data, layout, np.zeros(3), np.zeros(NUM_ACTIONS, dtype=np.float32))
        obs_hist.fill(obs_frame)

        # Fill depth buffer
        raw = depth_cam.capture(data)
        proc = preprocess_depth(raw)
        depth_hist.fill(proc)

    reset_env()
    target_pos = DEFAULT_JOINT_POS.copy()
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    command = np.zeros(3, dtype=np.float64)
    policy_step = 0
    step = 0
    start_time = time.perf_counter()

    # ---- Viewer ----
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
        print("\n[controls] Joystick: left stick=vx/vy, right stick x=yaw, A=reset")
        print("[controls] Keyboard: 8/2=vx, 4/6=vy, 7/9=yaw, Space=stop, 0=reset, Esc=quit\n")
    else:
        viewer = None

    print("[sim2sim] Starting simulation loop...")

    try:
        while viewer is None or viewer.is_running():
            # ---- Process joystick/keyboard ----
            try:
                vx, vy, dyaw, reset_req = joystick.process_events()
                if reset_req:
                    reset_env()
                    command[:] = 0.0
                    target_pos = DEFAULT_JOINT_POS.copy()
                    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    policy_step = 0
                    print("[sim2sim] Reset")
            except KeyboardInterrupt:
                break

            command[0] = vx
            command[1] = vy
            command[2] = dyaw

            # ---- Policy inference at policy rate ----
            if step % DECIMATION == 0:
                # Update camera and body tracking
                viewer.cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])]

                # Capture and preprocess depth
                raw_depth = depth_cam.capture(data)
                proc_depth = preprocess_depth(raw_depth)
                depth_hist.append(proc_depth)

                # Gather proprioceptive observation
                obs_dict = _get_obs_dict(data, layout, command, last_action)
                obs_hist.append(obs_dict)

                # Build inputs and run inference
                proprio_flat = obs_hist.flatten()  # (624,)
                depth_batch = depth_hist.sample()   # (8, 18, 32)
                action = policy_fn(proprio_flat, depth_batch)

                last_action = action
                target_pos = DEFAULT_JOINT_POS + action.astype(np.float64) * action_scale
                target_pos = np.clip(target_pos, ctrl_lo, ctrl_hi)

                # Log
                if policy_step % 50 == 0:
                    print(
                        f"t={step * SIM_TIMESTEP:6.1f}s  "
                        f"cmd=[{vx:+.2f},{vy:+.2f},{dyaw:+.2f}]  "
                        f"base_z={data.qpos[2]:.3f}  "
                        f"action=[{action.min():+.3f},{action.max():+.3f}]"
                    )

                policy_step += 1

            # ---- Apply control ----
            data.ctrl[layout["ctrl_ids"]] = target_pos
            mujoco.mj_step(model, data)

            # ---- Viewer sync ----
            if viewer is not None and step % 4 == 0:
                viewer.sync()

            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass

    print(f"\n[sim2sim] Finished. Total steps: {step}, sim time: {step * SIM_TIMESTEP:.1f}s")


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------
def _get_obs_dict(
    data: mujoco.MjData,
    layout: dict,
    command: np.ndarray,
    last_action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a single-frame proprioceptive observation dict (matches policy obs group)."""
    # Angular velocity from gyro sensor
    base_ang_vel = data.sensor("angular-velocity").data.copy().astype(np.float64) * 0.25

    # Projected gravity
    quat_wxyz = data.qpos[3:7].copy()
    projected_gravity = quat_apply_inverse_wxyz(
        quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )

    # Joint positions (relative to default)
    joint_pos = np.empty(NUM_ACTIONS, dtype=np.float64)
    for i in range(NUM_ACTIONS):
        joint_pos[i] = data.qpos[layout["qpos_ids"][i]]
    joint_pos_rel = joint_pos - DEFAULT_JOINT_POS

    # Joint velocities (scaled)
    joint_vel = np.empty(NUM_ACTIONS, dtype=np.float64)
    for i in range(NUM_ACTIONS):
        joint_vel[i] = data.qvel[layout["qvel_ids"][i]]
    joint_vel_scaled = joint_vel * 0.05

    return {
        "base_ang_vel": base_ang_vel.astype(np.float32),
        "projected_gravity": projected_gravity.astype(np.float32),
        "velocity_commands": command.astype(np.float32),
        "joint_pos": joint_pos_rel.astype(np.float32),
        "joint_vel": joint_vel_scaled.astype(np.float32),
        "actions": last_action.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CASBOT_02 parkour sim2sim")
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_ONNX_DIR,
        help="Directory with 0-depth_encoder.onnx and actor.onnx",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without viewer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.model_dir.exists():
        # Try to auto-find
        log_root = REPO_ROOT / "logs/instinct_rl/casbot02_parkour"
        if log_root.exists():
            runs = sorted(log_root.iterdir(), reverse=True)
            for r in runs:
                exported = r / "exported"
                if exported.exists():
                    args.model_dir = exported
                    break
        if not args.model_dir.exists():
            print(f"ERROR: ONNX model dir not found. Run `uv run instinct-play --export-onnx` first.")
            print(f"Tried: {args.model_dir}")
            sys.exit(1)

    print(f"[sim2sim] ONNX dir: {args.model_dir}")
    run(args.model_dir, headless=args.headless)
