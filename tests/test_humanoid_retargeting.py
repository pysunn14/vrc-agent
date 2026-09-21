from __future__ import annotations

import math
import unittest

import numpy as np

from vrc_ardy_agent.humanoid_retargeting import (
    ARDY_ARM_JOINTS,
    HumanoidRetargeter,
    RetargetingMonitor,
)
from vrc_ardy_agent.tracking_rig import RigCalibration
from vrc_ardy_agent.vmt_bridge import matrix_to_quaternion_xyzw


def _source_pose(*, wrist_distance: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((27, 3), dtype=np.float64)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 27, axis=0)

    positions[14] = (0.2, 1.5, 0.0)
    positions[15] = (0.5, 1.5, 0.0)
    positions[16] = (0.2 + wrist_distance, 1.5, 0.0)
    positions[8] = (-0.2, 1.5, 0.0)
    positions[9] = (-0.5, 1.5, 0.0)
    positions[10] = (-0.2 - wrist_distance, 1.5, 0.0)
    return positions, rotations


def _calibration() -> RigCalibration:
    return RigCalibration(
        name="test-rig",
        reference_head_height_m=1.0,
        shoulder_half_width_m=0.1,
        head_to_shoulder_drop_m=0.2,
        arm_reach_m=0.4,
        max_arm_extension_fraction=0.9,
        left_hand_rest_euler_deg=(90.0, 0.0, 0.0),
        right_hand_rest_euler_deg=(90.0, 0.0, 0.0),
    )


class HumanoidRetargeterTests(unittest.TestCase):
    def test_extended_arms_are_rebuilt_from_target_shoulders_and_clamped(self) -> None:
        monitor = RetargetingMonitor()
        retargeter = HumanoidRetargeter(
            calibration=_calibration(),
            source_joints=ARDY_ARM_JOINTS,
            hmd_base=(0.0, 1.0, 0.0),
            body_scale=1.0,
            monitor=monitor,
        )

        positions, rotations = _source_pose()
        hands = retargeter.retarget(positions, rotations)

        np.testing.assert_allclose(hands.left.position, (-0.46, 0.8, 0.0), atol=1e-9)
        np.testing.assert_allclose(hands.right.position, (0.46, 0.8, 0.0), atol=1e-9)
        expected_rest = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        np.testing.assert_allclose(
            matrix_to_quaternion_xyzw(hands.left.rotation),
            expected_rest,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            matrix_to_quaternion_xyzw(hands.right.rotation),
            expected_rest,
            atol=1e-9,
        )

        snapshot = monitor.snapshot()
        self.assertTrue(snapshot.initialized)
        self.assertEqual(snapshot.frames, 1)
        self.assertEqual(snapshot.left_clamp_count, 1)
        self.assertEqual(snapshot.right_clamp_count, 1)
        self.assertAlmostEqual(snapshot.left_output_reach_fraction, 0.9)
        self.assertAlmostEqual(snapshot.right_output_reach_fraction, 0.9)

    def test_bent_arm_preserves_source_reach_fraction(self) -> None:
        retargeter = HumanoidRetargeter(
            calibration=_calibration(),
            source_joints=ARDY_ARM_JOINTS,
            hmd_base=(0.0, 1.0, 0.0),
            body_scale=1.0,
        )
        positions, rotations = _source_pose()
        retargeter.retarget(positions, rotations)

        bent_positions, _ = _source_pose(wrist_distance=0.3)
        hands = retargeter.retarget(bent_positions, rotations)

        left_shoulder = np.asarray((-0.1, 0.8, 0.0))
        right_shoulder = np.asarray((0.1, 0.8, 0.0))
        self.assertAlmostEqual(
            float(np.linalg.norm(hands.left.position - left_shoulder)),
            0.2,
            places=9,
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(hands.right.position - right_shoulder)),
            0.2,
            places=9,
        )

    def test_shoulder_motion_uses_body_scale_instead_of_arm_reach_scale(self) -> None:
        retargeter = HumanoidRetargeter(
            calibration=_calibration(),
            source_joints=ARDY_ARM_JOINTS,
            hmd_base=(0.0, 1.0, 0.0),
            body_scale=0.5,
        )
        positions, rotations = _source_pose(wrist_distance=0.3)
        initial = retargeter.retarget(positions, rotations)

        moved = positions.copy()
        for index in (14, 15, 16, 8, 9, 10):
            moved[index, 1] += 0.2
        shifted = retargeter.retarget(moved, rotations)

        self.assertAlmostEqual(shifted.left.position[1] - initial.left.position[1], 0.1)
        self.assertAlmostEqual(shifted.right.position[1] - initial.right.position[1], 0.1)

    def test_wrist_rotation_is_a_delta_from_the_target_rest_rotation(self) -> None:
        retargeter = HumanoidRetargeter(
            calibration=_calibration(),
            source_joints=ARDY_ARM_JOINTS,
            hmd_base=(0.0, 1.0, 0.0),
            body_scale=1.0,
        )
        positions, rotations = _source_pose(wrist_distance=0.3)
        initial = retargeter.retarget(positions, rotations)

        angle = math.pi / 2.0
        source_delta = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotated = rotations.copy()
        rotated[16] = source_delta
        changed = retargeter.retarget(positions, rotated)

        np.testing.assert_allclose(
            matrix_to_quaternion_xyzw(initial.left.rotation),
            (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            atol=1e-9,
        )
        self.assertFalse(np.allclose(changed.left.rotation, source_delta))
        self.assertFalse(np.allclose(changed.left.rotation, initial.left.rotation))

    def test_degenerate_source_arm_fails_instead_of_using_a_fallback(self) -> None:
        retargeter = HumanoidRetargeter(
            calibration=_calibration(),
            source_joints=ARDY_ARM_JOINTS,
            hmd_base=(0.0, 1.0, 0.0),
            body_scale=1.0,
        )
        positions, rotations = _source_pose()
        positions[15] = positions[14]
        positions[16] = positions[14]

        with self.assertRaisesRegex(ValueError, "left source arm length"):
            retargeter.retarget(positions, rotations)


if __name__ == "__main__":
    unittest.main()
