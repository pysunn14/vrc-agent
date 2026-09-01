from pathlib import Path
import tempfile
import unittest

import numpy as np

from vrc_ardy_agent.locomotion import iter_locomotion_frames


class LocomotionTests(unittest.TestCase):
    def _write_motion(self, root: Path, root_positions: np.ndarray, headings: np.ndarray, fps: int = 20) -> Path:
        path = root / "motion.npz"
        np.savez(
            path,
            root_positions=root_positions.astype(np.float32),
            smooth_root_pos=root_positions.astype(np.float32),
            global_root_heading=headings.astype(np.float32),
            fps=np.array(fps),
        )
        return path

    def test_forward_world_z_maps_to_forward_stick(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.05]], dtype=np.float32)
            heading = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            path = self._write_motion(Path(tmp), root, heading, fps=20)

            frames = list(iter_locomotion_frames(path, full_stick_speed_mps=1.0, deadzone_mps=0.0))

            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[0].x, 0.0, places=6)
            self.assertAlmostEqual(frames[0].y, 1.0, places=6)
            self.assertAlmostEqual(frames[1].y, 1.0, places=6)

    def test_world_x_is_forward_when_heading_is_positive_90_degrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = np.array([[0.0, 1.0, 0.0], [0.05, 1.0, 0.0]], dtype=np.float32)
            heading = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
            path = self._write_motion(Path(tmp), root, heading, fps=20)

            frames = list(iter_locomotion_frames(path, full_stick_speed_mps=1.0, deadzone_mps=0.0))

            self.assertAlmostEqual(frames[0].x, 0.0, places=6)
            self.assertAlmostEqual(frames[0].y, 1.0, places=6)

    def test_deadzone_suppresses_small_root_jitter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.001]], dtype=np.float32)
            heading = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            path = self._write_motion(Path(tmp), root, heading, fps=20)

            frames = list(iter_locomotion_frames(path, full_stick_speed_mps=1.0, deadzone_mps=0.05))

            self.assertEqual(frames[0].x, 0.0)
            self.assertEqual(frames[0].y, 0.0)
            self.assertEqual(frames[1].x, 0.0)
            self.assertEqual(frames[1].y, 0.0)

    def test_diagonal_motion_is_normalized_to_unit_circle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = np.array([[0.0, 1.0, 0.0], [0.1, 1.0, 0.1]], dtype=np.float32)
            heading = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            path = self._write_motion(Path(tmp), root, heading, fps=20)

            frame = next(iter_locomotion_frames(path, full_stick_speed_mps=1.0, deadzone_mps=0.0))

            self.assertLessEqual(float(np.hypot(frame.x, frame.y)), 1.0 + 1e-6)


    def test_positive_heading_velocity_maps_to_positive_turn_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            heading = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            path = self._write_motion(Path(tmp), root, heading, fps=1)

            frames = list(
                iter_locomotion_frames(
                    path,
                    full_stick_speed_mps=1.0,
                    deadzone_mps=0.0,
                    full_turn_speed_dps=180.0,
                    turn_deadzone_dps=0.0,
                )
            )

            self.assertAlmostEqual(frames[0].turn, 0.5, places=6)
            self.assertAlmostEqual(frames[0].turn_speed_dps, 90.0, places=6)
            self.assertAlmostEqual(frames[1].turn, 0.5, places=6)

    def test_heading_wraparound_uses_shortest_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            a0 = np.deg2rad(179.0)
            a1 = np.deg2rad(-179.0)
            heading = np.array(
                [[np.cos(a0), np.sin(a0)], [np.cos(a1), np.sin(a1)]],
                dtype=np.float32,
            )
            path = self._write_motion(Path(tmp), root, heading, fps=1)

            frame = next(
                iter_locomotion_frames(
                    path,
                    full_stick_speed_mps=1.0,
                    deadzone_mps=0.0,
                    full_turn_speed_dps=180.0,
                    turn_deadzone_dps=0.0,
                )
            )

            self.assertAlmostEqual(frame.turn_speed_dps, 2.0, places=4)
            self.assertGreater(frame.turn, 0.0)


if __name__ == "__main__":
    unittest.main()