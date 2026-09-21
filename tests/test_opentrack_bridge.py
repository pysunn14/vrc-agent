from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from vrc_ardy_agent.opentrack_bridge import (
    encode_opentrack_packet,
    iter_opentrack_frames,
    rotation_matrix_to_opentrack_ypr,
)


class OpenTrackBridgeTests(unittest.TestCase):
    def test_packet_is_six_native_little_endian_doubles(self):
        packet = encode_opentrack_packet(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        self.assertEqual(len(packet), 48)
        self.assertEqual(struct.unpack("<6d", packet), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def test_identity_rotation_is_zero_ypr(self):
        ypr = rotation_matrix_to_opentrack_ypr(np.eye(3, dtype=np.float64))
        np.testing.assert_allclose(ypr, (0.0, 0.0, 0.0), atol=1e-6)

    def test_positive_steamvr_yaw_requires_negative_opentrack_yaw(self):
        theta = np.deg2rad(30.0)
        rot_y = np.array(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ],
            dtype=np.float64,
        )
        yaw, pitch, roll = rotation_matrix_to_opentrack_ypr(rot_y)
        self.assertAlmostEqual(yaw, -30.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(roll, 0.0, places=5)

    def test_npz_frames_are_relative_to_first_head_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            positions = np.zeros((2, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 27, 1, 1))
            positions[0, 6] = [1.0, 2.0, 3.0]
            positions[1, 6] = [1.1, 2.2, 3.3]
            path = Path(tmp) / "motion.npz"
            np.savez(path, posed_joints=positions, global_rot_mats=rotations, fps=np.array(20))

            frames = list(iter_opentrack_frames(path, joint_index=6))

            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].xyz_cm, (0.0, 0.0, 0.0))
            # ARDY X points left, while Unity X points right. The bridge
            # mirrors it before inverting the VRto3D packet mapping.
            np.testing.assert_allclose(frames[1].xyz_cm, (10.0, -20.0, 30.0), atol=1e-4)
            np.testing.assert_allclose(frames[1].ypr_deg, (0.0, 0.0, 0.0), atol=1e-6)
            self.assertEqual(frames[1].fps, 20.0)


if __name__ == "__main__":
    unittest.main()
