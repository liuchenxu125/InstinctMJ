"""CPU integration checks: python -m unittest discover -s scripts/tests -v."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

SCRIPT = Path(__file__).resolve().parents[1] / "sim2sim_g1_parkour.py"
spec = importlib.util.spec_from_file_location("sim2sim_g1_parkour", SCRIPT)
sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)


class G1Sim2SimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = sim.build_robot(sim.parse_args(["--terrain", "flat"]))
        cls.camera = sim.DepthCamera(cls.robot)

    def test_control_channels_and_torso_observations(self):
        robot = self.robot
        # The source MJCF has no integrator override; compilation alone uses
        # Euler. Deployment must retain training's implicit PD integration.
        self.assertEqual(robot.model.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        self.assertAlmostEqual(robot.model.opt.timestep, 0.005)
        for name, position_id, velocity_id in zip(robot.joint_names, robot.position_ctrl_ids, robot.velocity_ctrl_ids):
            joint_id = robot.model.joint(name).id
            self.assertEqual(robot.model.actuator_trnid[position_id, 0], joint_id)
            self.assertEqual(robot.model.actuator_trnid[velocity_id, 0], joint_id)
        data = mujoco.MjData(robot.model)
        data.qpos[:] = robot.initial_qpos
        # Rotate torso without rotating its waist joints; torso and pelvis IMU
        # velocities differ when the waist joint has a nonzero angular speed.
        data.qpos[3:7] = [np.cos(0.2), 0, np.sin(0.2), 0]
        data.qvel[robot.model.joint("waist_pitch_joint").dofadr[0]] = 1
        mujoco.mj_forward(robot.model, data)
        obs = sim.observations(robot, data, np.array([0.5, 0, 0.2]), np.arange(29))
        np.testing.assert_allclose(obs["base_ang_vel"], 0, atol=1e-7)
        self.assertGreater(np.linalg.norm(data.sensor("imu_ang_vel").data), 0.5)
        np.testing.assert_allclose(obs["joint_pos"], 0, atol=1e-7)
        np.testing.assert_allclose(obs["projected_gravity"], [np.sin(0.4), 0, -np.cos(0.4)], atol=1e-7)
        np.testing.assert_allclose(obs["joint_vel"][0], 0.05, atol=1e-7)

    def test_history_matches_training_buffers(self):
        from mjlab.utils.buffers import CircularBuffer

        from instinct_mj.utils.buffers import AsyncCircularBuffer

        histories = sim.Histories(self.robot, self.camera, 0)
        reference = {name: CircularBuffer(8, 1, "cpu") for name in sim.COMPONENTS}
        depth_reference = AsyncCircularBuffer(37, 1, "cpu")
        for step in range(45):
            obs = {
                name: np.full(
                    29 if name in ("joint_pos", "joint_vel", "actions") else 3, step + i * 100, dtype=np.float32
                )
                for i, name in enumerate(sim.COMPONENTS)
            }
            image = np.full((18, 32), step, dtype=np.float32)
            if step == 0:
                histories.reset(obs, image)
            else:
                histories.append(obs, image)
            for name in sim.COMPONENTS:
                reference[name].append(torch.tensor(obs[name])[None])
            depth_reference.append(torch.tensor(image)[None])
        proprio, depth = histories.inputs()
        expected = np.concatenate([reference[name].buffer.numpy().ravel() for name in sim.COMPONENTS])
        np.testing.assert_array_equal(proprio, expected)
        np.testing.assert_array_equal(depth, depth_reference.buffer.numpy()[0, [1, 6, 11, 16, 21, 26, 31, 36]])
        self.assertEqual(proprio.shape, (768,))
        delayed = sim.Histories(self.robot, self.camera, 1)
        np.testing.assert_array_equal(delayed.frame_indices, [0, 5, 10, 15, 20, 25, 30, 35])
        histories.reset(obs, image)
        self.assertTrue(np.all(histories.inputs()[1] == 44))

    def test_depth_rays_and_preprocessing_match_training(self):
        from instinct_mj.sensors.grouped_ray_caster.grouped_ray_caster_camera import GroupedRayCasterCamera

        cfg = self.camera.cfg
        obj = SimpleNamespace(
            cfg=cfg, _runtime_device="cpu", _camera_data=SimpleNamespace(intrinsic_matrices=torch.eye(3)[None])
        )
        GroupedRayCasterCamera._compute_intrinsic_matrices(obj)
        _, directions = GroupedRayCasterCamera._compute_pinhole_rays_from_intrinsics(
            obj, obj._camera_data.intrinsic_matrices
        )
        np.testing.assert_allclose(self.camera.directions, directions.numpy().reshape(36, 64, 3), atol=1e-7)
        raw = torch.linspace(0.1, 2.4, 36 * 64).reshape(1, 36, 64, 1)
        expected = raw
        for noise in cfg.noise_pipeline.values():
            expected = noise.func(expected, noise, [0])
        np.testing.assert_allclose(
            self.camera.preprocess(raw.numpy()[0, :, :, 0]), expected.numpy()[0, :, :, 0], atol=1e-6
        )

    def test_plane_depth_is_image_plane_distance(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom type="plane" size="0 0 1" group="0"/>
          <body name="torso_link"><freejoint/><inertial mass="1" diaginertia="1 1 1" pos="0 0 0"/></body>
        </worldbody></mujoco>""")
        camera = sim.DepthCamera(SimpleNamespace(model=model, env_cfg=self.robot.env_cfg))
        data = mujoco.MjData(model)
        data.qpos[:7] = [0, 0, 0.9, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        origin, rotation = camera.pose(data)
        rays = camera.directions @ rotation.T
        radial = -origin[2] / rays[:, :, 2]
        expected = np.where(
            (radial > 0) & (radial <= camera.cfg.max_distance),
            radial * camera.directions[:, :, 0],
            camera.cfg.max_distance,
        )
        np.testing.assert_allclose(camera.capture(data), expected, atol=1e-6)

    def test_staircase_up_down_and_physics(self):
        for terrain in ("stairs", "up", "down", "obstacles"):
            with self.subTest(terrain=terrain):
                robot = sim.build_robot(sim.parse_args(["--terrain", terrain]))
                if terrain == "stairs":
                    last_up = robot.model.geom("stair_up_5")
                    last_down = robot.model.geom("stair_down_4")
                    self.assertAlmostEqual(last_up.pos[2] + last_up.size[2], 0.9)
                    self.assertAlmostEqual(last_down.pos[2] + last_down.size[2], 0.15)
                self.assertAlmostEqual(robot.initial_qpos[2], 1.8 if terrain == "down" else 0.9)
                data = mujoco.MjData(robot.model)
                data.qpos[:] = robot.initial_qpos
                data.ctrl[robot.position_ctrl_ids] = robot.default_pos
                for _ in range(100):
                    mujoco.mj_step(robot.model, data)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertTrue(np.isfinite(data.qvel).all())

    def test_viewer_reset_and_cleanup(self):
        args = sim.parse_args(["--zero-policy", "--terrain", "flat", "--duration", "0.04", "--no-realtime"])
        viewer = Mock()
        viewer.cam = SimpleNamespace(lookat=np.zeros(3), distance=4, azimuth=90, elevation=-20)
        viewer.lock.side_effect = nullcontext
        viewer.is_running.return_value = True

        def open_viewer(model, data, key_callback):
            key_callback(ord("R"))
            key_callback(ord("C"))
            return viewer

        with patch("mujoco.viewer.launch_passive", side_effect=open_viewer):
            result = sim.run(args)
        self.assertEqual(result["policy_steps"], 2)
        self.assertEqual(result["resets"], 1)
        viewer.close.assert_called_once()
        self.assertEqual(viewer.sync.call_count, 2)

    def test_real_onnx_fixed_batches_and_metadata_rejection(self):
        histories = sim.Histories(self.robot, self.camera, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            # Encoder has batch 10; actor has batch 1. Each session must be fed
            # independently instead of forcing a shared exported batch size.
            encoder = helper.make_graph(
                [
                    helper.make_node("ReduceMean", ["depth"], ["mean"], axes=[1, 2, 3], keepdims=1),
                    helper.make_node("Flatten", ["mean"], ["flat"], axis=1),
                    helper.make_node("MatMul", ["flat", "weight"], ["latent"]),
                ],
                "encoder",
                [helper.make_tensor_value_info("depth", TensorProto.FLOAT, [10, 8, 18, 32])],
                [helper.make_tensor_value_info("latent", TensorProto.FLOAT, [10, 128])],
                [numpy_helper.from_array(np.ones((1, 128), np.float32), "weight")],
            )
            actor = helper.make_graph(
                [helper.make_node("Slice", ["obs", "start", "end", "axis"], ["action"])],
                "actor",
                [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 896])],
                [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])],
                [
                    numpy_helper.from_array(np.array(v, dtype=np.int64), k)
                    for k, v in (("start", [768]), ("end", [797]), ("axis", [1]))
                ],
            )
            for filename, graph in (("0-depth_encoder.onnx", encoder), ("actor.onnx", actor)):
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
                model.ir_version = 9
                onnx.save(model, path / filename)
            policy = sim.OnnxPolicy(path, histories)
            action = policy(np.zeros(768, np.float32), np.full((8, 18, 32), 0.4, np.float32))
            # Float32 ReduceMean over 4608 pixels accumulates rounding error.
            np.testing.assert_allclose(action, 0.4, atol=5e-6)
            (path / "metadata.json").write_text(
                json.dumps({"num_actions": 18, "task_id": "Instinct-Parkour-Target-Amp-CASBOT02-v0"})
            )
            with self.assertRaisesRegex(ValueError, "metadata"):
                sim.OnnxPolicy(path, histories)


if __name__ == "__main__":
    unittest.main()
