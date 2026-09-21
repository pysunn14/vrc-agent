from pathlib import Path
import tempfile
import unittest

import numpy as np

from vrc_ardy_agent.six_point_bridge import iter_six_point_frames


class SixPointBridgeTests(unittest.TestCase):
    def _write_motion(self, root: Path, positions: np.ndarray, rotations: np.ndarray) -> Path:
        path = root / "motion.npz"
        np.savez(path, posed_joints=positions, global_rot_mats=rotations, fps=np.array(20))
        return path

    def test_auto_scale_places_toes_on_floor_and_head_at_hmd_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            positions = np.zeros((1, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (1, 27, 1, 1))
            positions[0, 6] = [0.0, 2.0, 0.0]   # head
            positions[0, 22] = [-0.1, 0.0, 0.0] # right toe floor reference
            positions[0, 26] = [0.1, 0.0, 0.0]  # left toe floor reference
            positions[0, 21] = [-0.1, 0.1, 0.0] # right ankle/foot
            positions[0, 25] = [0.1, 0.1, 0.0]  # left ankle/foot
            positions[0, 0] = [0.0, 1.1, 0.0]   # hips
            positions[0, 10] = [-0.5, 1.4, 0.0]
            positions[0, 16] = [0.5, 1.4, 0.0]
            path = self._write_motion(Path(tmp), positions, rotations)

            frame = next(iter_six_point_frames(path, hmd_base=(0.0, 1.0, 0.0)))

            self.assertAlmostEqual(frame.scale, 0.5, places=6)
            self.assertEqual(frame.head.xyz_cm, (0.0, 0.0, 0.0))
            np.testing.assert_allclose(frame.left_foot.position, (-0.05, 0.05, 0.0), atol=1e-6)
            np.testing.assert_allclose(frame.right_foot.position, (0.05, 0.05, 0.0), atol=1e-6)
            np.testing.assert_allclose(frame.hips.position, (0.0, 0.55, 0.0), atol=1e-6)

    def test_body_translation_moves_all_six_points_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            positions = np.zeros((2, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 27, 1, 1))
            base = {
                6: [0.0, 2.0, 0.0],
                22: [-0.1, 0.0, 0.0],
                26: [0.1, 0.0, 0.0],
                21: [-0.1, 0.1, 0.0],
                25: [0.1, 0.1, 0.0],
                0: [0.0, 1.1, 0.0],
                10: [-0.5, 1.4, 0.0],
                16: [0.5, 1.4, 0.0],
            }
            delta = np.array([0.2, 0.1, 0.3], dtype=np.float32)
            for idx, value in base.items():
                positions[0, idx] = value
                positions[1, idx] = np.asarray(value, dtype=np.float32) + delta
            path = self._write_motion(Path(tmp), positions, rotations)

            frames = list(iter_six_point_frames(path, hmd_base=(0.0, 1.0, 0.0)))
            expected = delta * np.array([-0.5, 0.5, 0.5], dtype=np.float32)
            for attr in ("left", "right", "hips", "left_foot", "right_foot"):
                p0 = np.asarray(getattr(frames[0], attr).position)
                p1 = np.asarray(getattr(frames[1], attr).position)
                np.testing.assert_allclose(p1 - p0, expected, atol=1e-6)
            # ARDY X points left. After the Unity-space mirror and VRto3D's
            # packet inversion, this becomes a positive packet X value.
            np.testing.assert_allclose(frames[1].head.xyz_cm, (10.0, -5.0, 15.0), atol=1e-5)

    def test_horizontal_root_motion_can_be_removed_from_tracking_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            positions = np.zeros((2, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 27, 1, 1))
            base = {
                6: [0.0, 2.0, 0.0],
                22: [-0.1, 0.0, 0.0],
                26: [0.1, 0.0, 0.0],
                21: [-0.1, 0.1, 0.0],
                25: [0.1, 0.1, 0.0],
                0: [0.0, 1.1, 0.0],
                10: [-0.5, 1.4, 0.0],
                16: [0.5, 1.4, 0.0],
            }
            root_delta = np.array([0.2, 0.0, 0.3], dtype=np.float32)
            for idx, value in base.items():
                positions[0, idx] = value
                positions[1, idx] = np.asarray(value, dtype=np.float32) + root_delta

            root_positions = np.array([[0.0, 1.1, 0.0], [0.2, 1.1, 0.3]], dtype=np.float32)
            path = Path(tmp) / "motion.npz"
            np.savez(
                path,
                posed_joints=positions,
                global_rot_mats=rotations,
                root_positions=root_positions,
                smooth_root_pos=root_positions,
                fps=np.array(20),
            )

            frames = list(
                iter_six_point_frames(
                    path,
                    hmd_base=(0.0, 1.0, 0.0),
                    remove_horizontal_root_motion=True,
                )
            )

            for attr in ("left", "right", "hips", "left_foot", "right_foot"):
                p0 = np.asarray(getattr(frames[0], attr).position)
                p1 = np.asarray(getattr(frames[1], attr).position)
                np.testing.assert_allclose(p1 - p0, np.zeros(3), atol=1e-6)
            np.testing.assert_allclose(frames[1].head.xyz_cm, (0.0, 0.0, 0.0), atol=1e-6)

    def test_locomotion_is_attached_to_six_point_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            positions = np.zeros((2, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 27, 1, 1))
            base = {
                6: [0.0, 2.0, 0.0],
                22: [-0.1, 0.0, 0.0],
                26: [0.1, 0.0, 0.0],
                21: [-0.1, 0.1, 0.0],
                25: [0.1, 0.1, 0.0],
                0: [0.0, 1.1, 0.0],
                10: [-0.5, 1.4, 0.0],
                16: [0.5, 1.4, 0.0],
            }
            root_delta = np.array([0.0, 0.0, 0.05], dtype=np.float32)
            for idx, value in base.items():
                positions[0, idx] = value
                positions[1, idx] = np.asarray(value, dtype=np.float32) + root_delta

            root_positions = np.array([[0.0, 1.1, 0.0], [0.0, 1.1, 0.05]], dtype=np.float32)
            headings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            path = Path(tmp) / "motion.npz"
            np.savez(
                path,
                posed_joints=positions,
                global_rot_mats=rotations,
                root_positions=root_positions,
                smooth_root_pos=root_positions,
                global_root_heading=headings,
                fps=np.array(20),
            )

            frames = list(
                iter_six_point_frames(
                    path,
                    hmd_base=(0.0, 1.0, 0.0),
                    remove_horizontal_root_motion=True,
                    locomotion=True,
                    full_stick_speed_mps=1.0,
                    locomotion_deadzone_mps=0.0,
                )
            )

            self.assertAlmostEqual(frames[0].locomotion_x, 0.0, places=6)
            self.assertAlmostEqual(frames[0].locomotion_y, 1.0, places=6)
            self.assertAlmostEqual(frames[1].locomotion_y, 1.0, places=6)


    def test_root_yaw_is_removed_from_tracking_pose_when_locomotion_is_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            positions = np.zeros((2, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 27, 1, 1))
            base = {
                6: [0.0, 2.0, 0.0],
                22: [-0.1, 0.0, 0.0],
                26: [0.1, 0.0, 0.0],
                21: [-0.1, 0.1, 0.0],
                25: [0.1, 0.1, 0.0],
                0: [0.0, 1.1, 0.0],
                10: [-0.5, 1.4, 0.0],
                16: [0.5, 1.4, 0.0],
            }
            for idx, value in base.items():
                positions[0, idx] = value

            theta = np.deg2rad(90.0)
            yaw = np.array(
                [
                    [np.cos(theta), 0.0, np.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)],
                ],
                dtype=np.float32,
            )
            root = np.asarray(base[0], dtype=np.float32)
            for idx, value in base.items():
                rel = np.asarray(value, dtype=np.float32) - root
                positions[1, idx] = root + yaw @ rel
                rotations[1, idx] = yaw

            root_positions = np.array([root, root], dtype=np.float32)
            headings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            path = Path(tmp) / "motion.npz"
            np.savez(
                path,
                posed_joints=positions,
                global_rot_mats=rotations,
                root_positions=root_positions,
                smooth_root_pos=root_positions,
                global_root_heading=headings,
                fps=np.array(1),
            )

            frames = list(
                iter_six_point_frames(
                    path,
                    hmd_base=(0.0, 1.0, 0.0),
                    locomotion=True,
                    full_stick_speed_mps=1.0,
                    locomotion_deadzone_mps=0.0,
                    full_turn_speed_dps=180.0,
                    turn_deadzone_dps=0.0,
                )
            )

            for attr in ("left", "right", "hips", "left_foot", "right_foot"):
                np.testing.assert_allclose(
                    getattr(frames[1], attr).position,
                    getattr(frames[0], attr).position,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    getattr(frames[1], attr).quaternion_xyzw,
                    getattr(frames[0], attr).quaternion_xyzw,
                    atol=1e-6,
                )
            np.testing.assert_allclose(frames[1].head.xyz_cm, (0.0, 0.0, 0.0), atol=1e-6)
            np.testing.assert_allclose(frames[1].head.ypr_deg, (0.0, 0.0, 0.0), atol=1e-6)
            self.assertAlmostEqual(frames[0].locomotion_turn, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
