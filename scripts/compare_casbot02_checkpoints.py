"""Paired deterministic Play of CASBOT02 checkpoints on stair flights.

Uses InstinctRlEnv, its trained observations/camera and native terrain generator.
Each checkpoint receives the same cases; actor means are used without sampling.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import instinct_rl.modules as policy_modules
import mujoco
import numpy as np
import torch
from mjlab.managers import EventTermCfg
from mjlab.terrains import FlatPatchSamplingCfg

from instinct_mj.envs import InstinctRlEnv
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.scripts.instinct_rl.play import _disable_headless_debug_visualization
from instinct_mj.tasks.parkour.config.casbot02.casbot02_parkour_target_amp_cfg import (
    instinct_casbot02_parkour_amp_final_cfg,
)
from instinct_mj.tasks.parkour.mdp.commands import PoseVelocityCommandCfg
from instinct_mj.tasks.parkour.mdp.commands.pose_velocity_command import PoseVelocityCommand
from instinct_mj.tasks.registry import load_instinct_rl_cfg
from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinPyramidStairsTerrainCfg


class FlightCommand(PoseVelocityCommand):
    def _resample_command(self, env_ids):
        self.pos_command_w[env_ids] = self._env.flight_goals[env_ids]
        self.max_command_b[env_ids, 0] = self._env.flight_speeds[env_ids]
        self.max_command_b[env_ids, 1] = 0.15
        self.is_standing_env[env_ids] = False
        self.random_velocity_indices[env_ids] = False

    def _update_command(self):
        super()._update_command()
        # One second to settle and fill camera/proprioception histories.
        standing = (self._env.episode_length_buf * self._env.step_dt < 1.0) | self._env.flight_finished
        self.vel_command_b[standing] = 0


@dataclass(kw_only=True)
class FlightCommandCfg(PoseVelocityCommandCfg):
    def build(self, env):
        return FlightCommand(self, env)


def reset_flight(env, env_ids):
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] = env.flight_starts[env_ids]
    yaw = env.flight_yaws[env_ids]
    root[:, 3:7] = 0
    root[:, 3] = torch.cos(yaw / 2)
    root[:, 6] = torch.sin(yaw / 2)
    root[:, 7:] = 0
    robot.write_root_state_to_sim(root, env_ids)


class FlightEnv(InstinctRlEnv):
    def __init__(self, cfg, cases, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cases = cases
        self.flight_finished = torch.zeros(len(cases), dtype=torch.bool, device=self.device)
        self.flight_speeds = torch.tensor([c["speed"] for c in cases], device=self.device)
        self.flight_yaws = torch.tensor([c["yaw"] for c in cases], device=self.device)
        terrain = self.scene.terrain
        types = torch.tensor([c["height_index"] for c in cases], device=self.device)
        terrain.terrain_types[:] = types
        terrain.terrain_levels[:] = 0
        terrain.env_origins[:] = terrain.terrain_origins[0, types]

        model = self.sim.mj_model
        gids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)
        self.heightfields = []
        self.heightfield_metadata = []
        starts, goals, limits = [], [], []
        for index in range(5):
            origin = terrain.terrain_origins[0, index].cpu().numpy()
            gid = min(gids, key=lambda g: np.linalg.norm(self.sim.mj_data.geom_xpos[g, :2] - origin[:2]))
            hid = model.geom_dataid[gid]
            nr, nc = model.hfield_nrow[hid], model.hfield_ncol[hid]
            adr = model.hfield_adr[hid]
            size = model.hfield_size[hid]
            pos = self.sim.mj_data.geom_xpos[gid].copy()
            heights = model.hfield_data[adr : adr + nr * nc].reshape(nr, nc) * size[2] + pos[2]
            assert np.allclose(self.sim.mj_data.geom_xmat[gid].reshape(3, 3), np.eye(3))
            self.heightfields.append((torch.tensor(heights, dtype=torch.float32, device=self.device), pos, size))
            profile = heights[nr // 2]
            xx = np.linspace(-size[0], size[0], nc)
            top, bottom = float(profile.max()), float(profile.min())
            assert abs(top - index * 0.25) < 1e-5, f"Unexpected stair tile assignment: {index}, {top}"
            high = xx[profile >= top - 1e-5]
            elevated = xx[profile > bottom + 1e-5]
            # Use actual generated geometry rather than ideal stair coordinates.
            first = float(elevated.min()) if len(elevated) else -2.65
            last = float(elevated.max()) if len(elevated) else 2.65
            plateau = float(high.min()) if len(elevated) else -1.25
            self.heightfield_metadata.append(
                {
                    "height_index": index,
                    "nrow": int(nr),
                    "ncol": int(nc),
                    "spacing_m": [float(2 * size[0] / (nc - 1)), float(2 * size[1] / (nr - 1))],
                    "bottom_m": bottom,
                    "top_m": top,
                    "first_step_x": first,
                    "last_step_x": last,
                    "plateau_left_x": plateau,
                    "center_xy": pos[:2].tolist(),
                }
            )
        for c in cases:
            _, pos, _ = self.heightfields[c["height_index"]]
            m = self.heightfield_metadata[c["height_index"]]
            up = c["direction"] == "up"
            start_x = m["first_step_x"] - 0.65 if up else 0.0
            goal_x = m["plateau_left_x"] + 0.65 if up else m["last_step_x"] + 0.75
            ground = m["bottom_m"] if up else m["top_m"]
            starts.append([pos[0] + start_x + c["x_offset"], pos[1] + c["y_offset"], ground + 0.92])
            goals.append([pos[0] + goal_x, pos[1], m["top_m"] if up else m["bottom_m"]])
            limits.append(pos[0] + (m["plateau_left_x"] + 0.10 if up else m["last_step_x"] + 0.15))
        self.flight_starts = torch.tensor(starts, dtype=torch.float32, device=self.device)
        self.flight_goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
        self.flight_foot_limits = torch.tensor(limits, dtype=torch.float32, device=self.device)
        self.flight_types = types
        self.flight_center_y = torch.tensor(
            [self.heightfield_metadata[c["height_index"]]["center_xy"][1] for c in cases], device=self.device
        )

    def ground_height(self, root):
        result = torch.zeros(len(root), device=self.device)
        for index, (grid, pos, size) in enumerate(self.heightfields):
            selected = self.flight_types == index
            p = root[selected]
            x = ((p[:, 0] - pos[0] + size[0]) / (2 * size[0]) * (grid.shape[1] - 1)).clamp(0, grid.shape[1] - 1.001)
            y = ((p[:, 1] - pos[1] + size[1]) / (2 * size[1]) * (grid.shape[0] - 1)).clamp(0, grid.shape[0] - 1.001)
            col, row = x.long(), y.long()
            fx, fy = x - col, y - row
            h00, h10 = grid[row, col], grid[row, col + 1]
            h01, h11 = grid[row + 1, col], grid[row + 1, col + 1]
            result[selected] = torch.where(
                fx >= fy, h00 + fx * (h10 - h00) + fy * (h11 - h10), h00 + fx * (h11 - h01) + fy * (h01 - h00)
            )
        return result


def make_cases(variants, seed, speeds):
    rng = np.random.default_rng(seed)
    cases = []
    for hi, height in enumerate([0.0, 0.05, 0.10, 0.15, 0.20]):
        for direction in ["up", "down"]:
            for speed in speeds:
                for variant in range(variants):
                    cases.append(
                        {
                            "case_id": len(cases),
                            "height_index": hi,
                            "height_m": height,
                            "direction": direction,
                            "speed": speed,
                            "variant": variant,
                            "x_offset": float(rng.uniform(-0.08, 0.08)),
                            "y_offset": float(rng.uniform(-0.12, 0.12)),
                            "yaw": float(rng.uniform(-0.05, 0.05)),
                        }
                    )
    return cases


def make_cfg(cases, seed, duration):
    cfg = instinct_casbot02_parkour_amp_final_cfg(play=True)
    cfg.seed = seed
    cfg.scene.num_envs = len(cases)
    cfg.episode_length_s = duration + 1
    cfg.terminations = {}  # Outcomes are measured before explicit resets below.
    cfg.curriculum = {}
    cfg.monitors = {}
    cfg.auto_reset = False
    gen = cfg.scene.terrain.terrain_generator
    # Generator curriculum fixes the terrain type assigned to each column.
    # Environment curriculum stays disabled, so cases never change level.
    gen.num_rows, gen.num_cols, gen.curriculum = 1, 5, True
    cfg.scene.terrain.max_init_terrain_level = 0
    gen.sub_terrains = {
        f"stairs_{int(h*100)}cm": PerlinPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(h, h),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            perlin_cfg=None,
            wall_prob=[0.0] * 4,
            flat_patch_sampling={"target": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.1, max_height_diff=0.01)},
        )
        for h in [0.0, 0.05, 0.10, 0.15, 0.20]
    }
    cmd = copy.deepcopy(cfg.commands["base_velocity"])
    cfg.commands["base_velocity"] = FlightCommandCfg(**{k: v for k, v in vars(cmd).items() if k != "class_type"})
    cmd = cfg.commands["base_velocity"]
    cmd.velocity_ranges = None
    cmd.random_velocity_terrain = None
    cmd.ranges.ang_vel_z = (-0.5, 0.5)
    cmd.resampling_time_range = (duration + 2, duration + 2)
    cmd.rel_standing_envs = 0.0
    cfg.events["reset_base"] = EventTermCfg(func=reset_flight, mode="reset")
    for group in cfg.observations.values():
        if group is not None:
            group.enable_corruption = False
            for term in group.terms.values():
                if term is not None and "delayed_frame_ranges" in term.params:
                    term.params["delayed_frame_ranges"] = (1, 1)
                if term is not None and term.delay_max_lag > 0:
                    term.delay_min_lag = term.delay_max_lag = 1
    for act in cfg.scene.entities["robot"].articulation.actuators:
        act.delay_min_lag = act.delay_max_lag = 1
    _disable_headless_debug_visualization(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("logs/instinct_rl/casbot02_parkour/2026-10-05_20-54-33"))
    parser.add_argument("--output", type=Path, default=Path("logs/analysis/casbot02_checkpoint_comparison"))
    parser.add_argument("--variants", type=int, default=16)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_cases = make_cases(args.variants, args.seed, args.speeds)
    n = len(base_cases)
    cases = [dict(c, checkpoint=cp) for cp in [2000, 8000] for c in base_cases]
    env = FlightEnv(make_cfg(cases, args.seed, args.duration), cases, device=args.device)
    print("ACTUAL_HEIGHTFIELDS", json.dumps(env.heightfield_metadata), flush=True)
    vec = InstinctRlVecEnvWrapper(env)
    agent = load_instinct_rl_cfg("Instinct-Parkour-Target-Amp-CASBOT02-v0").to_dict()
    assert not agent["empirical_normalization"]
    models, hashes = [], {}
    for cp in [2000, 8000]:
        path = args.run_dir / f"model_{cp}.pt"
        cfg = copy.deepcopy(agent["policy"])
        cls = getattr(policy_modules, cfg.pop("class_name"))
        model = cls(obs_format=vec.get_obs_format(), num_actions=vec.num_actions, **cfg).to(args.device)
        model.load_state_dict(torch.load(path, map_location=args.device, weights_only=False)["model_state_dict"])
        model.eval()
        models.append(model)
        hashes[str(cp)] = hashlib.sha256(path.read_bytes()).hexdigest()
    obs, _ = vec.get_observations()
    difference = float((obs[:n] - obs[n:]).abs().max())
    assert difference < 1e-5, f"Unpaired initial observations: {difference}"
    robot = env.scene["robot"]
    feet_ids = [robot.body_names.index(name) for name in ("leg_l6_link", "leg_r6_link")]
    device = env.device
    outcomes = [None] * len(cases)
    finish_time = torch.full((len(cases),), args.duration, device=device)
    stable = torch.zeros(len(cases), dtype=torch.long, device=device)
    stable_steps = round(1.0 / env.step_dt)
    stuck_steps = round(5.0 / env.step_dt)
    history = []
    trace = []
    command_trace, velocity_trace, diagnostics = [], [], []
    segments = vec.get_obs_segments()
    command_offset = 0
    for name, shape in segments.items():
        if name == "velocity_commands":
            command_length = math.prod(shape)
            break
        command_offset += math.prod(shape)
    frozen = env.sim.data.qpos.clone()
    max_progress = torch.zeros(len(cases), device=device)
    start_x = env.flight_starts[:, 0].clone()
    count_steps = round(args.duration / env.step_dt)
    with torch.inference_mode():
        for step in range(count_steps):
            action = torch.cat([models[0].act_inference(obs[:n]), models[1].act_inference(obs[n:])])
            assert torch.isfinite(action).all()
            obs, _, dones, _ = vec.step(action)
            assert not dones.any()
            root = robot.data.root_link_pos_w
            assert torch.isfinite(root).all()
            active = ~env.flight_finished
            max_progress = torch.where(active, torch.maximum(max_progress, root[:, 0] - start_x), max_progress)
            fall = (robot.data.projected_gravity_b[:, 2] > -math.cos(1.0)) | (
                root[:, 2] - env.ground_height(root) < 0.45
            )
            off_course = (root[:, 1] - env.flight_center_y).abs() > 0.60
            feet_x = robot.data.body_link_pos_w[:, feet_ids, 0]
            goal = torch.linalg.vector_norm(root[:, :2] - env.flight_goals[:, :2], dim=-1) < 0.25
            goal &= torch.all(feet_x > env.flight_foot_limits[:, None], dim=-1)
            goal &= robot.data.projected_gravity_b[:, 2] < -math.cos(0.6)
            goal &= torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1) < 0.25
            stable = torch.where(goal & active, stable + 1, 0)
            success = stable >= stable_steps
            history.append(root[:, 0].clone())
            stuck = torch.zeros_like(active)
            if step >= stuck_steps + round(1.5 / env.step_dt):
                progress = root[:, 0] - history[-stuck_steps - 1]
                command = env.command_manager.get_term("base_velocity").vel_command_b
                stuck = (progress < 0.15) & (command[:, 0] > 0.30) & ~goal
            newly = torch.zeros_like(active)
            for label, mask in [("fall", fall), ("off_course", off_course), ("success", success), ("stuck", stuck)]:
                ids = (mask & active & ~newly).nonzero().flatten()
                for i in ids.cpu().tolist():
                    outcomes[i] = label
                finish_time[ids] = (step + 1) * env.step_dt
                newly[ids] = True
            frozen[newly] = env.sim.data.qpos[newly]
            env.flight_finished |= newly
            if step % 5 == 0:
                trace.append(torch.where(env.flight_finished[:, None], frozen, env.sim.data.qpos).cpu().numpy().copy())
                command_trace.append(env.command_manager.get_term("base_velocity").command.cpu().numpy().copy())
                velocity_trace.append(robot.data.root_link_lin_vel_b.cpu().numpy().copy())
            if step in [50, 100, 250]:
                command = env.command_manager.get_term("base_velocity").command
                observed = obs[:, command_offset : command_offset + command_length].reshape(len(cases), -1, 3)[:, -1]
                error = float((observed - command).abs().max())
                assert error < 1e-5, f"Policy command observation mismatch: {error}"
                diagnostics.append(
                    {
                        "time_s": (step + 1) * env.step_dt,
                        "command_observation_max_error": error,
                        "samples": [
                            dict(
                                index=i,
                                checkpoint=cases[i]["checkpoint"],
                                height_m=cases[i]["height_m"],
                                speed=cases[i]["speed"],
                                command=command[i].tolist(),
                                observed_command=observed[i].tolist(),
                                velocity_b=robot.data.root_link_lin_vel_b[i].tolist(),
                                action_mean=action[i].tolist(),
                            )
                            for i in [
                                0,
                                args.variants,
                                2 * n // 5,
                                2 * n // 5 + args.variants,
                                n,
                                n + args.variants,
                                n + 2 * n // 5,
                                n + 2 * n // 5 + args.variants,
                            ]
                        ],
                    }
                )
            if newly.any():
                env.reset(env_ids=newly.nonzero().flatten())
                obs, _ = vec.get_observations()
            if step % 100 == 0:
                print(
                    "PROGRESS",
                    step,
                    {label: outcomes.count(label) for label in ["success", "fall", "stuck", "off_course", None]},
                    flush=True,
                )
            if env.flight_finished.all():
                break
    rows = [
        dict(c, outcome=outcomes[i] or "timeout", time_s=float(finish_time[i]), max_progress_m=float(max_progress[i]))
        for i, c in enumerate(cases)
    ]
    with (args.output / "trials.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = {
        "run_dir": str(args.run_dir),
        "seed": args.seed,
        "variants_per_height_direction_speed": args.variants,
        "cases_per_checkpoint": n,
        "duration_s": args.duration,
        "speeds_m_s": args.speeds,
        "policy": "act_inference (mean, no sampling)",
        "step_dt_s": env.step_dt,
        "physics_dt_s": env.physics_dt,
        "motor_delay_physics_steps": 1,
        "depth_observation_delay_policy_steps": 1,
        "initial_obs_max_pair_difference": difference,
        "checkpoint_sha256": hashes,
        "heightfields": env.heightfield_metadata,
        "success": "reach goal within 0.25m, both feet beyond last stair edge, upright, speed<0.25m/s for 1s",
        "fall": "tilt>1 rad or torso-ground clearance<0.45m",
        "stuck": "forward progress<0.15m over 5s with vx command>0.30m/s, after 1.5s warmup",
        "off_course": "lateral corridor error>0.60m",
        "note": "Nominal deterministic Play with paired spawn variations; not a general real-world success estimate.",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    np.savez_compressed(
        args.output / "trajectories.npz",
        qpos=np.stack(trace),
        command=np.stack(command_trace),
        velocity_b=np.stack(velocity_trace),
        dt=env.step_dt * 5,
    )
    mujoco.mj_saveModel(env.sim.mj_model, str(args.output / "model.mjb"))
    for cp in [2000, 8000]:
        selected = [r for r in rows if r["checkpoint"] == cp and r["height_m"] > 0]
        print(
            "RESULT",
            cp,
            {
                label: sum(r["outcome"] == label for r in selected)
                for label in ["success", "fall", "stuck", "off_course", "timeout"]
            },
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
