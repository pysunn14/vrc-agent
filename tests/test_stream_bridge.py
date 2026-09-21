from __future__ import annotations

import math
import unittest

import numpy as np

from vrc_ardy_agent.ardy_runtime import ArdyMotionChunk
from vrc_ardy_agent.humanoid_retargeting import RetargetingMonitor
from vrc_ardy_agent.lower_body_retargeting import (
    ARDY_LOWER_BODY_JOINTS,
    LowerBodyRetargetingMonitor,
    LowerBodySafetyLimits,
)
from vrc_ardy_agent.stream_bridge import RetargetingMode, SixPointStreamMapper
from vrc_ardy_agent.tracking_rig import RigCalibration, get_avatar_profile

from tests.test_avatar_rig_profile import _profile_document
from vrc_ardy_agent.avatar_rig_profile import AvatarRigProfile

TEST_PROFILE = AvatarRigProfile.from_mapping(_profile_document())
TEST_CALIBRATION = RigCalibration.from_avatar_profile(TEST_PROFILE)


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
        8: np.array([-0.2, 1.5, 0.0]),
        9: np.array([-0.35, 1.4, 0.0]),
        10: np.array([-0.5, 1.4, 0.0]),
        14: np.array([0.2, 1.5, 0.0]),
        15: np.array([0.35, 1.4, 0.0]),
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
    def test_lower_body_position_only_mode_isolates_dynamic_body_positions(self):
        mapper = SixPointStreamMapper(
            avatar_profile=TEST_PROFILE,
            retargeting_mode=RetargetingMode.LOWER_BODY_POSITION_ONLY,
            hmd_base=(0.0, 1.0, 0.0),
            scale=0.5,
            heading_smoothing_seconds=0.0,
        )
        chunk = _make_chunk([0.0, 0.1], [0.0, 15.0])
        chunk.posed_joints[1, ARDY_LOWER_BODY_JOINTS.hips, 1] -= 0.1
        chunk.posed_joints[1, ARDY_LOWER_BODY_JOINTS.left_foot, 2] += 0.1
        chunk.global_rot_mats[1] = np.einsum(
            "ij,kjl->kil",
            _yaw_matrix(0.7),
            chunk.global_rot_mats[1],
        )

        first, second = mapper.map_chunk(chunk)

        self.assertFalse(second.tracker_activation.left)
        self.assertFalse(second.tracker_activation.right)
        self.assertTrue(second.tracker_activation.hips)
        self.assertTrue(second.tracker_activation.left_foot)
        self.assertTrue(second.tracker_activation.right_foot)
        self.assertEqual(second.head.xyz_cm, (0.0, 0.0, 0.0))
        self.assertEqual(second.head.ypr_deg, (0.0, 0.0, 0.0))
        self.assertEqual(
            (second.locomotion_x, second.locomotion_y, second.locomotion_turn),
            (0.0, 0.0, 0.0),
        )
        self.assertNotEqual(second.hips.position, first.hips.position)
        self.assertNotEqual(second.left_foot.position, first.left_foot.position)
        for tracker in (second.hips, second.left_foot, second.right_foot):
            np.testing.assert_allclose(
                tracker.quaternion_xyzw,
                (0.0, 0.0, 0.0, 1.0),
                atol=1e-9,
            )

    def test_ardy_left_axis_is_mirrored_into_unity_tracker_space(self):
        mapper = SixPointStreamMapper(
            avatar_profile=TEST_PROFILE,
            hmd_base=(0.0, 1.0, 0.0),
            heading_smoothing_seconds=0.0,
        )

        frame = mapper.map_chunk(_make_chunk([0.0], [0.0]))[0]

        self.assertLess(frame.left.position[0], 0.0)
        self.assertGreater(frame.right.position[0], 0.0)
        self.assertLess(frame.left_foot.position[0], 0.0)
        self.assertGreater(frame.right_foot.position[0], 0.0)

    def test_first_body_frame_uses_avatar_profile_instead_of_source_proportions(self):
        mapper = SixPointStreamMapper(
            avatar_profile=TEST_PROFILE,
            hmd_base=(0.0, 1.0, 0.0),
            heading_smoothing_seconds=0.0,
        )

        frame = mapper.map_chunk(_make_chunk([0.0], [0.0]))[0]
        reference = TEST_PROFILE.body_tracker_poses(hmd_base=(0.0, 1.0, 0.0))

        for bone_name, tracker in (
            ("hips", frame.hips),
            ("leftFoot", frame.left_foot),
            ("rightFoot", frame.right_foot),
        ):
            np.testing.assert_allclose(
                tracker.position,
                reference[bone_name].position_m,
                atol=1e-9,
            )

    def test_body_motion_is_retargeted_as_delta_from_the_profile_pose(self):
        monitor = LowerBodyRetargetingMonitor()
        limits = LowerBodySafetyLimits(
            max_hips_offset_hmd_fraction=100.0,
            max_foot_offset_hmd_fraction=100.0,
            max_leg_extension_fraction=100.0,
            max_hips_speed_hmd_fraction_per_second=1000.0,
            max_foot_speed_hmd_fraction_per_second=1000.0,
        )
        mapper = SixPointStreamMapper(
            avatar_profile=TEST_PROFILE,
            lower_body_monitor=monitor,
            lower_body_limits=limits,
            hmd_base=(0.0, 1.0, 0.0),
            scale=0.5,
            heading_smoothing_seconds=0.0,
        )
        chunk = _make_chunk([0.0, 0.0], [0.0, 0.0])
        chunk.posed_joints[1, ARDY_LOWER_BODY_JOINTS.hips] += (0.2, -0.1, 0.1)
        chunk.posed_joints[1, ARDY_LOWER_BODY_JOINTS.left_foot] += (0.1, 0.2, -0.1)

        frames = mapper.map_chunk(chunk)
        reference = TEST_PROFILE.body_tracker_poses(hmd_base=(0.0, 1.0, 0.0))

        expected_hips = np.asarray(reference["hips"].position_m) + (-0.1, -0.05, 0.05)
        expected_left_foot = np.asarray(reference["leftFoot"].position_m) + (
            -0.05,
            0.1,
            -0.05,
        )
        np.testing.assert_allclose(frames[1].hips.position, expected_hips, atol=1e-9)
        np.testing.assert_allclose(
            frames[1].left_foot.position,
            expected_left_foot,
            atol=1e-9,
        )
        self.assertTrue(monitor.snapshot().initialized)
        self.assertEqual(monitor.snapshot().frames, 2)

    def test_root_velocity_continues_across_chunk_boundary(self):
        mapper = SixPointStreamMapper(
            avatar_profile=TEST_PROFILE,
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
            avatar_profile=TEST_PROFILE,
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

    def test_live_hands_are_retargeted_with_observable_reach_limits(self):
        monitor = RetargetingMonitor()
        mapper = SixPointStreamMapper(
            avatar_profile=TEST_PROFILE,
            retargeting_monitor=monitor,
            hmd_base=(0.0, 1.0, 0.0),
            heading_smoothing_seconds=0.0,
        )

        frame = mapper.map_chunk(_make_chunk([0.0], [0.0]))[0]
        snapshot = monitor.snapshot()

        for side, hand in (("left", frame.left), ("right", frame.right)):
            shoulder = np.asarray(
                TEST_CALIBRATION.shoulder_position((0.0, 1.0, 0.0), side=side)
            )
            distance = float(np.linalg.norm(np.asarray(hand.position) - shoulder))
            self.assertLessEqual(
                distance,
                TEST_CALIBRATION.scaled_arm_reach(1.0)
                * TEST_CALIBRATION.max_arm_extension_fraction
                + 1e-9,
            )
        self.assertTrue(snapshot.initialized)
        self.assertEqual(snapshot.frames, 1)


if __name__ == "__main__":
    unittest.main()
