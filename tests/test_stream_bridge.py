from __future__ import annotations

import math
import unittest

import numpy as np

from vrc_ardy_agent.ardy_runtime import ArdyMotionChunk
from vrc_ardy_agent.stream_bridge import SixPointStreamMapper


def _yaw_matrix(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _make_chunk(root_z: list[float], heading_deg: list[float], *, fps: float = 20.0) -> ArdyMotionChunk:
    frames = len(root_z)
    positions = np.zeros((frames, 27, 3), dtype=np.float64)
    rotations = np.zeros((frames, 27, 3, 3), dtype=np.float64)
    root_positions = np.zeros((frames, 3), dtype=np.float64)
    headings = np.zeros((frames, 2), dtype=np.float64)

    offsets = {
        0: np.array([0.0, 1.1, 0.0]),
        6: np.array([0.0, 2.0, 0.0]),
        10: np.array([-0.5, 1.4, 0.0]),
        16: np.array([0.5, 1.4, 0.0]),
        21: np.array([-0.1, 0.1, 0.0]),
        25: np.array([0.1, 0.1, 0.0]),
        22: np.array([-0.1, 0.0, 0.0]),
        26: np.array([0.1, 0.0, 0.0]),
    }

    for frame_idx, (z, deg) in enumerate(zip(root_z, heading_deg, strict=True)):
        angle = math.radians(deg)
        yaw = _yaw_matrix(angle)
        root = np.array([0.0, 0.0, z], dtype=np.float64)
        root_positions[frame_idx] = [0.0, 1.1, z]
        headings[frame_idx] = [math.cos(angle), math.sin(angle)]
        for joint_idx in range(27):
            rotations[frame_idx, joint_idx] = yaw
        for joint_idx, offset in offsets.items():
            horizontal = np.array([offset[0], 0.0, offset[2]])
            rotated = yaw @ horizontal
            positions[frame_idx, joint_idx] = root + rotated + np.array([0.0, offset[1], 0.0])

    return ArdyMotionChunk(
        posed_joints=positions,
        global_rot_mats=rotations,
        root_positions=root_positions,
        smooth_root_pos=root_positions.copy(),
        global_root_heading=headings,
        foot_contacts=np.zeros((frames, 4), dtype=np.float64),
        fps=fps,
        prompt="test",
        generation_seconds=0.0,
    )


class SixPointStreamMapperTests(unittest.TestCase):
    def test_root_velocity_continues_across_chunk_boundary(self):
        mapper = SixPointStreamMapper(
            hmd_base=(0.0, 1.0, 0.0),
            full_stick_speed_mps=1.0,
            locomotion_deadzone_mps=0.0,
            heading_smoothing_seconds=0.0,
        )

        first = mapper.map_chunk(_make_chunk([0.00, 0.05], [0.0, 0.0]))
        second = mapper.map_chunk(_make_chunk([0.10, 0.15], [0.0, 0.0]))

        self.assertEqual(first[0].locomotion_y, 0.0)
        self.assertAlmostEqual(first[1].locomotion_y, 1.0, places=6)
        self.assertAlmostEqual(second[0].locomotion_y, 1.0, places=6)
        self.assertAlmostEqual(second[1].locomotion_y, 1.0, places=6)
        for frame in [*first, *second]:
            np.testing.assert_allclose(frame.head.xyz_cm, (0.0, 0.0, 0.0), atol=1e-6)

    def test_root_yaw_is_removed_and_turn_velocity_continues_across_chunks(self):
        mapper = SixPointStreamMapper(
            hmd_base=(0.0, 1.0, 0.0),
            full_stick_speed_mps=10.0,
            locomotion_deadzone_mps=0.0,
            full_turn_speed_dps=100.0,
            turn_deadzone_dps=0.0,
            heading_smoothing_seconds=0.0,
        )

        first = mapper.map_chunk(_make_chunk([0.0, 0.0], [0.0, 5.0]))
        second = mapper.map_chunk(_make_chunk([0.0, 0.0], [10.0, 15.0]))

        self.assertEqual(first[0].locomotion_turn, 0.0)
        self.assertAlmostEqual(first[1].locomotion_turn, 1.0, places=6)
        self.assertAlmostEqual(second[0].locomotion_turn, 1.0, places=6)
        self.assertAlmostEqual(second[1].locomotion_turn, 1.0, places=6)

        reference_left = np.asarray(first[0].left.position)
        for frame in [*first, *second]:
            np.testing.assert_allclose(frame.left.position, reference_left, atol=1e-6)
            np.testing.assert_allclose(frame.head.ypr_deg, (0.0, 0.0, 0.0), atol=1e-5)


if __name__ == "__main__":
    unittest.main()