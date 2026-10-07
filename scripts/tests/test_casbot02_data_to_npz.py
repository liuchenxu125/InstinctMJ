"""Check CASBOT02 source conventions and the actual InstinctMJ loader schema."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from instinct_mj.motion_reference.motion_files.amass_motion import AmassMotion

SCRIPT = Path(__file__).resolve().parents[1] / "casbot02_data_to_npz.py"
spec = importlib.util.spec_from_file_location("casbot02_converter", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class Casbot02ConversionTests(unittest.TestCase):
    def test_header_detection_keeps_numeric_first_frame_and_rejects_damage(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "motion.data"
            first = ",".join(str(i) for i in range(61)) + ",\n"
            source.write_text(first * 2)
            data, skipped = converter.load_raw_data(source)
            self.assertFalse(skipped)
            np.testing.assert_array_equal(data, np.tile(np.arange(61), (2, 1)))
            source.write_text(",".join(f"column{i}" for i in range(61)) + "\n" + first * 2)
            data, skipped = converter.load_raw_data(source)
            self.assertTrue(skipped)
            self.assertEqual(len(data), 2)
            source.write_text(first.replace("0,", "bad,", 1) + first)
            with self.assertRaisesRegex(ValueError, "invalid numeric frame"):
                converter.load_raw_data(source)
            source.write_text(first + first.replace("0,", "nan,", 1))
            with self.assertRaisesRegex(ValueError, "non-finite frame"):
                converter.load_raw_data(source)

    def test_joint_mapping_coordinate_signs_and_both_pitch_conventions(self):
        raw = np.tile(np.arange(61), (201, 1)).astype(float)
        raw[:, :6] = [2, 3, 0.87, 0, 0, np.pi / 2]
        raw[:, 0] += np.arange(201) / 1000
        motion, report = converter.build_motion(raw)
        self.assertEqual(motion["joint_pos"].shape, (11, 18))
        np.testing.assert_allclose(motion["base_pos_w"][:, 0], 3)
        np.testing.assert_allclose(motion["base_pos_w"][:, 1], raw[::20, 0])
        np.testing.assert_allclose(motion["base_quat_w"][0], [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)], atol=1e-6)
        self.assertEqual(report["pitch_offset_rad"], 0)
        expected = list(range(12, 24)) + [24, 25, 27, 31, 32, 34]
        np.testing.assert_array_equal(motion["joint_pos"][0], expected)
        raw[:, 3] += np.pi / 2
        upright, report = converter.build_motion(raw)
        self.assertAlmostEqual(report["pitch_offset_rad"], np.pi / 2)
        np.testing.assert_allclose(upright["base_quat_w"], motion["base_quat_w"], atol=1e-6)

    def test_quaternion_continuity_across_yaw_wrap(self):
        raw = np.zeros((401, 61))
        yaw = np.linspace(3, 3.5, len(raw))
        raw[:, 5] = (yaw + np.pi) % (2 * np.pi) - np.pi
        motion, _ = converter.build_motion(raw)
        quat = motion["base_quat_w"]
        np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-6)
        self.assertTrue(np.all(np.sum(quat[:-1] * quat[1:], axis=1) > 0.99))

    def test_invalid_rates_and_short_clips_fail(self):
        raw = np.zeros((201, 61))
        for kwargs in ({"output_fps": 60}, {"input_fps": 0}, {"trim_start_frames": 201}, {"keep_seconds": -1}):
            with self.assertRaises(ValueError):
                converter.build_motion(raw, **kwargs)
        with self.assertRaisesRegex(ValueError, "at least 10"):
            converter.build_motion(raw[:100])

    def test_npz_is_read_by_actual_retargeted_loader_with_joint_names(self):
        raw = np.zeros((201, 61))
        raw[:, 2] = 0.87
        motion, _ = converter.build_motion(raw)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "motion_retargeted.npz"
            np.savez_compressed(target, **motion)
            # Let the real loader perform IO and name mapping. The GPU smoke
            # check separately exercises FK and velocity computation.
            loader = AmassMotion.__new__(AmassMotion)
            loader.buffer_device = "cpu"
            loader.sim_joint_names = list(reversed(motion["joint_names"].tolist()))
            loader._all_motion_files = [str(target)]
            loader._pack_retargetted_motion_sequence = Mock(return_value="packed")
            self.assertEqual(loader._read_motion_file(0), "packed")
            root, quat, joints, fps = loader._pack_retargetted_motion_sequence.call_args.args
            np.testing.assert_array_equal(root.numpy(), motion["base_pos_w"])
            np.testing.assert_array_equal(quat.numpy(), motion["base_quat_w"])
            np.testing.assert_array_equal(joints.numpy(), motion["joint_pos"][:, ::-1])
            self.assertEqual(fps, 50)


if __name__ == "__main__":
    unittest.main()
