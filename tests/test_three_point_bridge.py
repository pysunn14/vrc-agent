from pathlib import Path
import tempfile
import unittest

import numpy as np

from vrc_ardy_agent.three_point_bridge import iter_three_point_frames


class ThreePointBridgeTests(unittest.TestCase):
    def test_first_frame_uses_shared_head_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            positions = np.zeros((2, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 27, 1, 1))

            positions[0, 6] = [1.0, 2.0, 3.0]
            positions[0, 16] = [1.4, 1.5, 3.2]
            positions[0, 10] = [0.6, 1.6, 3.1]

            positions[1] = positions[0]
            source = Path(tmp) / "motion.npz"
            np.savez(source, posed_joints=positions, global_rot_mats=rotations, fps=np.array(20))

            frame = next(iter_three_point_frames(source, hmd_base=(0.0, 1.0, 0.0)))

            self.assertEqual(frame.head.xyz_cm, (0.0, 0.0, 0.0))
            np.testing.assert_allclose(frame.left.position, (-0.4, 0.5, 0.2), atol=1e-6)
            np.testing.assert_allclose(frame.right.position, (0.4, 0.6, 0.1), atol=1e-6)
            self.assertEqual(frame.fps, 20.0)

    def test_body_translation_moves_head_and_both_hands_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            positions = np.zeros((2, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 27, 1, 1))

            positions[0, 6] = [0.0, 1.7, 0.0]
            positions[0, 16] = [0.3, 1.0, 0.2]
            positions[0, 10] = [-0.3, 1.0, 0.2]

            translation = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            positions[1] = positions[0] + translation

            source = Path(tmp) / "motion.npz"
            np.savez(source, posed_joints=positions, global_rot_mats=rotations, fps=np.array(20))

            frames = list(iter_three_point_frames(source, hmd_base=(0.0, 1.0, 0.0)))

            # ARDY X points left. It is mirrored before the packet is
            # pre-inverted for VRto3D.
            np.testing.assert_allclose(frames[1].head.xyz_cm, (10.0, -20.0, 30.0), atol=1e-4)
            np.testing.assert_allclose(
                np.asarray(frames[1].left.position) - np.asarray(frames[0].left.position),
                translation * np.array([-1.0, 1.0, 1.0], dtype=np.float32),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(frames[1].right.position) - np.asarray(frames[0].right.position),
                translation * np.array([-1.0, 1.0, 1.0], dtype=np.float32),
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
