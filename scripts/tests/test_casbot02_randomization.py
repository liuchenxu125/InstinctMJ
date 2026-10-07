"""CASBOT02 HANDOFF randomization contract and camera attachment checks."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api import math as math_utils

from instinct_mj.assets.casbot_02 import (
    CASBOT02_FOOT_GEOM_NAMES,
    CASBOT02_FOOT_SOLE_HEIGHT,
    CASBOT02_PARKOUR_INIT_STATE,
)
from instinct_mj.envs.mdp.events.randomization import randomize_camera_offsets
from instinct_mj.tasks.parkour.config.casbot02.casbot02_parkour_target_amp_cfg import (
    instinct_casbot02_parkour_amp_final_cfg,
)


class Casbot02RandomizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = instinct_casbot02_parkour_amp_final_cfg()
        cls.robot = Entity(cls.cfg.scene.entities["robot"])

    def test_handoff_ranges_and_body_actuator_selection(self):
        events = self.cfg.events
        self.assertEqual(events["base_mass"].params["ranges"], (-4.0, 4.0))
        self.assertEqual(events["non_base_mass"].params["ranges"], (0.8, 1.2))
        self.assertEqual(
            events["base_com"].params["ranges"],
            {
                0: (-0.025, 0.025),
                1: (-0.025, 0.025),
                2: (-0.03, 0.03),
            },
        )
        for key in ("base_mass", "non_base_mass", "base_com", "actuator_gains", "camera_installation"):
            self.assertEqual(events[key].mode, "startup")
        scene = {"robot": self.robot}
        for key in ("base_mass", "non_base_mass", "base_com", "actuator_gains"):
            asset = copy.deepcopy(events[key].params["asset_cfg"])
            asset.resolve(scene)
            if key == "non_base_mass":
                names = {self.robot.body_names[i] for i in asset.body_ids}
                self.assertEqual(names, set(self.robot.body_names) - {"torso"})
            elif key in ("base_mass", "base_com"):
                self.assertEqual([self.robot.body_names[i] for i in asset.body_ids], ["torso"])
            else:
                names = {name for i in asset.actuator_ids for name in self.robot.actuators[i].cfg.target_names_expr}
                self.assertEqual(names, set(self.robot.joint_names))
                self.assertEqual(sum(name.startswith("leg_") for name in names), 12)
                self.assertEqual(sum(name.startswith("upper_") for name in names), 6)
                self.assertEqual(len(names), 18)
        self.assertEqual(events["actuator_gains"].params["kp_range"], (0.85, 1.15))
        self.assertEqual(events["actuator_gains"].params["kd_range"], (0.85, 1.15))

    def test_fields_are_per_world_and_play_omits_new_randomization(self):
        fields = {
            field for term in self.cfg.events.values() if term for field in getattr(term.func, "model_fields", ())
        }
        self.assertTrue({"geom_friction", "body_mass", "body_ipos", "actuator_gainprm", "actuator_biasprm"} <= fields)
        self.assertNotIn("body_inertia", fields)  # Match HANDOFF's mass-only updates.
        play = instinct_casbot02_parkour_amp_final_cfg(play=True)
        for key in ("base_mass", "non_base_mass", "base_com", "actuator_gains", "camera_installation"):
            self.assertNotIn(key, play.events)
        camera = next(s for s in self.cfg.scene.sensors if s.name == "camera")
        self.assertEqual(list(camera.noise_pipeline), ["crop_and_resize", "gaussian_blur", "depth_normalization"])
        self.assertEqual(camera.noise_pipeline["crop_and_resize"].crop_region, (18, 0, 16, 16))
        self.assertEqual(self.cfg.observations["policy"].terms["joint_pos"].noise.n_max, 0.01)
        self.assertEqual(len(self.robot.joint_names), 18)

    def test_camera_offsets_remain_local_and_do_not_accumulate(self):
        torch.manual_seed(42)
        cfg = copy.deepcopy(next(s for s in self.cfg.scene.sensors if s.name == "camera"))
        n = 64
        nominal_pos = torch.tensor(cfg.offset.pos).expand(n, -1)
        nominal_quat = torch.tensor(cfg.offset.rot).expand(n, -1)
        # Test an arbitrarily rotated/moving parent, not only the identity pose.
        body_pos = torch.tensor([1.0, -2.0, 0.9]).expand(n, -1)
        body_quat = math_utils.quat_from_euler_xyz(torch.full((n,), 0.2), torch.full((n,), -0.1), torch.full((n,), 0.7))
        camera = SimpleNamespace(
            cfg=cfg,
            _device="cpu",
            data=SimpleNamespace(pos_w=torch.zeros(n, 3)),
            _offset_pos=nominal_pos.clone(),
            _offset_quat=nominal_quat.clone(),
        )
        camera._compute_view_world_poses = lambda ids: (body_pos[ids].clone(), body_quat[ids].clone())

        def set_world_poses(pos, quat, env_ids, convention):
            self.assertEqual(convention, "world")
            inverse = math_utils.quat_inv(body_quat[env_ids])
            camera._offset_pos[env_ids] = math_utils.quat_apply(inverse, pos - body_pos[env_ids])
            camera._offset_quat[env_ids] = math_utils.quat_mul(inverse, quat)

        camera.set_world_poses = set_world_poses

        class CameraScene(dict):
            num_envs = n

        env = SimpleNamespace(scene=CameraScene(camera=camera))
        params = self.cfg.events["camera_installation"].params
        selected = torch.arange(0, n, 2)
        untouched = torch.arange(1, n, 2)
        nominal_euler = torch.stack(math_utils.euler_xyz_from_quat(nominal_quat), dim=-1)
        for _ in range(12):
            randomize_camera_offsets(env, selected, **params)
            self.assertLessEqual((camera._offset_pos[selected] - nominal_pos[selected]).abs().max().item(), 0.030001)
            euler = torch.stack(math_utils.euler_xyz_from_quat(camera._offset_quat), dim=-1)
            self.assertLessEqual((euler[selected] - nominal_euler[selected]).abs().max().item(), 0.052361)
            torch.testing.assert_close(camera._offset_pos[untouched], nominal_pos[untouched])
            torch.testing.assert_close(camera._offset_quat[untouched], nominal_quat[untouched])
        self.assertGreater((camera._offset_pos[selected] - nominal_pos[selected]).std().item(), 0.005)
        torch.testing.assert_close(torch.linalg.vector_norm(camera._offset_quat, dim=-1), torch.ones(n))
        randomize_camera_offsets(env, None, **params)
        self.assertGreater((camera._offset_pos[untouched] - nominal_pos[untouched]).abs().max().item(), 0.001)

    def test_isaac_shell_capsule_contacts_and_sole_geometry(self):
        model = self.robot.spec.compile()
        self.assertFalse(np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
        for name in CASBOT02_FOOT_GEOM_NAMES:
            body_name = "leg_l6_link" if name.startswith("left_") else "leg_r6_link"
            geom = model.geom(name)
            gid = geom.id
            self.assertEqual(model.geom_type[gid], mujoco.mjtGeom.mjGEOM_CAPSULE)
            self.assertEqual(model.body(model.geom_bodyid[gid]).name, body_name)
            self.assertEqual(model.geom_condim[gid], 3)
            self.assertEqual(model.geom_priority[gid], 1)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
            axis = rotation.reshape(3, 3)[:, 2]
            radius, half_length = model.geom_size[gid, :2]
            index = int(name.rsplit("_", 1)[1])
            self.assertAlmostEqual(radius, 0.01)
            self.assertAlmostEqual(half_length, (0.1, 0.12, 0.13, 0.13, 0.13, 0.12, 0.1)[index])
            np.testing.assert_allclose(
                model.geom_pos[gid], (0.035, (-0.04, -0.026, -0.013, 0, 0.013, 0.026, 0.04)[index], -0.054)
            )
            self.assertGreater(abs(axis[0]), 0.99999)
            lowest = model.geom_pos[gid, 2] - abs(axis[2]) * half_length - radius
            self.assertAlmostEqual(lowest, -CASBOT02_FOOT_SOLE_HEIGHT, places=6)
            body_geoms = np.where(model.geom_bodyid == model.geom_bodyid[gid])[0]
            active = body_geoms[(model.geom_contype[body_geoms] | model.geom_conaffinity[body_geoms]) != 0]
            self.assertEqual(len(active), 7)
            self.assertTrue(np.all(model.geom_type[active] == mujoco.mjtGeom.mjGEOM_CAPSULE))
        # IsaacLab imports the shell URDF cylinders as capsules, including knees.
        for body_name in ("leg_l4_link", "leg_r4_link"):
            body_geoms = np.where(model.geom_bodyid == model.body(body_name).id)[0]
            active = body_geoms[(model.geom_contype[body_geoms] | model.geom_conaffinity[body_geoms]) != 0]
            self.assertEqual(len(active), 1)
            self.assertEqual(model.geom_type[active[0]], mujoco.mjtGeom.mjGEOM_CAPSULE)
        physical = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
        self.assertEqual(len(physical), 27)
        self.assertEqual(np.count_nonzero(model.geom_type[physical] == mujoco.mjtGeom.mjGEOM_CAPSULE), 26)
        self.assertEqual(np.count_nonzero(model.geom_type[physical] == mujoco.mjtGeom.mjGEOM_SPHERE), 1)
        nonfeet = [gid for gid in physical if model.geom(gid).name not in CASBOT02_FOOT_GEOM_NAMES]
        self.assertTrue(np.all(model.geom_condim[nonfeet] == 3))
        self.assertEqual(
            self.cfg.rewards["rewards"]["feet_at_plane"].params["height_offset"], CASBOT02_FOOT_SOLE_HEIGHT
        )
        grid = next(s for s in self.cfg.scene.sensors if s.name == "leg_volume_points").points_generator
        self.assertAlmostEqual(grid.z_min + CASBOT02_FOOT_SOLE_HEIGHT, -0.005)
        self.assertAlmostEqual(grid.z_max + CASBOT02_FOOT_SOLE_HEIGHT, 0.035)
        self.assertEqual((grid.x_min, grid.x_max), (-0.025, 0.12))
        self.assertEqual((grid.y_min, grid.y_max), (-0.03, 0.03))
        self.assertEqual((grid.x_num, grid.y_num, grid.z_num), (10, 5, 2))
        self.assertEqual(self.cfg.scene.terrain.terrain_generator.hfield_resolution, 0.10)
        self.assertEqual(self.cfg.scene.terrain.terrain_generator.horizontal_scale, 0.07)
        play = instinct_casbot02_parkour_amp_final_cfg(play=True)
        self.assertEqual(play.scene.terrain.terrain_generator.hfield_resolution, 0.10)

    def test_contact_material_is_not_clamped_and_nominal_self_contact_is_absent(self):
        spec = self.robot.spec.copy()
        spec.worldbody.add_geom(
            name="test_ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0.0, 0.0, 1.0),
            contype=1,
            conaffinity=15,
            condim=4,
            friction=(0.9, 0.2, 0.2),
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        data.qpos[:3] = CASBOT02_PARKOUR_INIT_STATE.pos
        data.qpos[3:7] = (1, 0, 0, 0)
        for name, position in CASBOT02_PARKOUR_INIT_STATE.joint_pos.items():
            data.qpos[model.joint(name).qposadr[0]] = position
        mujoco.mj_forward(model, data)
        feet = [model.geom(name).id for name in CASBOT02_FOOT_GEOM_NAMES]
        # Lower the capsule sole 0.5 mm into the plane to produce contacts.
        lowest_sole = min(data.xpos[model.geom_bodyid[gid], 2] - CASBOT02_FOOT_SOLE_HEIGHT for gid in feet)
        data.qpos[2] -= lowest_sole + 0.0005
        ground_id = model.geom("test_ground").id
        for coefficient in (0.3, 1.6):
            model.geom_friction[feet, 0] = coefficient
            mujoco.mj_forward(model, data)
            contacted_feet = set()
            for contact in data.contact:
                # No artificial shank-foot force in the nominal standing pose.
                self.assertIn(ground_id, (contact.geom1, contact.geom2))
                foot_id = contact.geom2 if contact.geom1 == ground_id else contact.geom1
                self.assertIn(foot_id, feet)
                contacted_feet.add(foot_id)
                self.assertEqual(contact.dim, 3)
                np.testing.assert_allclose(contact.friction[:2], coefficient)
            self.assertEqual(contacted_feet, set(feet))


if __name__ == "__main__":
    unittest.main()
