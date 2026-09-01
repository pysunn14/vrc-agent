from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from vrc_ardy_agent.vmt_bridge import (
    encode_vmt_room_unity,
    iter_vmt_frames,
    matrix_to_quaternion_xyzw,
)


class VmtBridgeTests(unittest.TestCase):
    def test_matrix_to_quaternion_identity(self):
        quat = matrix_to_quaternion_xyzw(np.eye(3, dtype=np.float32))
        np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_iter_vmt_frames_reads_right_hand(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            positions = np.zeros((2, 27, 3), dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 27, 1, 1))
            positions[0, 10] = [1.0, 2.0, 3.0]
            positions[1, 10] = [4.0, 5.0, 6.0]
            source = tmp_path / "motion.npz"
            np.savez(source, posed_joints=positions, global_rot_mats=rotations, fps=np.array(20))

            frames = list(iter_vmt_frames(source, joint_index=10))

            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].position, (1.0, 2.0, 3.0))
            self.assertEqual(frames[0].quaternion_xyzw, (0.0, 0.0, 0.0, 1.0))
            self.assertEqual(frames[0].fps, 20.0)

    def test_encode_vmt_room_unity_packet_layout(self):
        packet = encode_vmt_room_unity(
            index=0,
            enable=1,
            timeoffset=0.0,
            position=(1.0, 2.0, 3.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        )

        self.assertTrue(packet.startswith(b"/VMT/Room/Unity\x00"))
        self.assertIn(b",iiffffffff\x00", packet)
        self.assertTrue(packet.endswith(struct.pack(">f", 1.0)))


if __name__ == "__main__":
    unittest.main()