from __future__ import annotations

import math
import unittest

import numpy as np

from vrc_ardy_agent.tracking_rig import (
    AVATAR_PROFILES,
    BodyTrackingMode,
    HandSelection,
    NeutralPoseConfig,
    RigCalibration,
    build_neutral_tracking_frame,
    get_avatar_profile,
)

from tests.test_avatar_rig_profile import _profile_document
from vrc_ardy_agent.avatar_rig_profile import AvatarRigProfile

TEST_PROFILE = AvatarRigProfile.from_mapping(_profile_document())
TEST_CALIBRATION = RigCalibration.from_avatar_profile(TEST_PROFILE)


class TrackingRigTests(unittest.TestCase):
    def test_empty_local_catalog_is_valid(self):
        import tempfile
        from pathlib import Path
        from vrc_ardy_agent.tracking_rig import _load_bundled_avatar_profiles
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(dict(_load_bundled_avatar_profiles(Path(directory))), {})

    def test_synthetic_arm_calibration_is_derived_from_the_exported_profile(self):
        self.assertAlmostEqual(
            TEST_CALIBRATION.reference_head_height_m,
            TEST_PROFILE.reference_height_m,
        )
        self.assertAlmostEqual(
            TEST_CALIBRATION.shoulder_half_width_m,
            TEST_PROFILE.shoulder_half_width_m,
        )
        self.assertAlmostEqual(
            TEST_CALIBRATION.arm_reach_m,
            TEST_PROFILE.arm_reach_m,
        )

    def test_neutral_pose_defaults_to_head_only_tracking(self):
        frame = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(hmd_base=(0.0, 1.0, 0.0)),
        )

        self.assertEqual(frame.left_enable, 0)
        self.assertEqual(frame.right_enable, 0)
        self.assertEqual(frame.body_enable, 0)
        self.assertFalse(frame.frame.tracker_activation.left)
        self.assertFalse(frame.frame.tracker_activation.right)
        self.assertFalse(frame.frame.tracker_activation.hips)
        self.assertFalse(frame.frame.tracker_activation.left_foot)
        self.assertFalse(frame.frame.tracker_activation.right_foot)

    def test_synthetic_neutral_hands_are_mirrored_and_inside_arm_reach(self):
        config = NeutralPoseConfig(
            hmd_base=(0.0, 1.0, 0.0),
            fps=20.0,
            reach_fraction=0.85,
        )

        frame = build_neutral_tracking_frame(TEST_PROFILE, config)

        self.assertAlmostEqual(
            frame.left.position[0], -frame.right.position[0], places=6
        )
        self.assertAlmostEqual(
            frame.left.position[1], frame.right.position[1], places=6
        )
        self.assertAlmostEqual(
            frame.left.position[2], frame.right.position[2], places=6
        )

        scaled_reach = TEST_CALIBRATION.scaled_arm_reach(config.hmd_base[1])
        for side, hand in (("left", frame.left), ("right", frame.right)):
            shoulder = TEST_CALIBRATION.shoulder_position(config.hmd_base, side=side)
            distance = float(
                np.linalg.norm(np.asarray(hand.position) - np.asarray(shoulder))
            )
            self.assertAlmostEqual(
                distance, scaled_reach * config.reach_fraction, places=6
            )
            self.assertLess(distance, scaled_reach)

    def test_neutral_pose_is_anchored_to_hmd_room_position(self):
        origin = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(hmd_base=(0.0, 1.0, 0.0)),
        )
        translated = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(hmd_base=(1.5, 1.0, -2.0)),
        )

        expected_delta = np.asarray((1.5, 0.0, -2.0))
        np.testing.assert_allclose(
            np.asarray(translated.left.position) - np.asarray(origin.left.position),
            expected_delta,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(translated.right.position) - np.asarray(origin.right.position),
            expected_delta,
            atol=1e-6,
        )

    def test_controller_orientation_points_forward_axis_down(self):
        frame = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(hmd_base=(0.0, 1.0, 0.0)),
        )

        for hand in (frame.left, frame.right):
            qx, qy, qz, qw = hand.quaternion_xyzw
            self.assertAlmostEqual(qx, math.sqrt(0.5), places=6)
            self.assertAlmostEqual(qy, 0.0, places=6)
            self.assertAlmostEqual(qz, 0.0, places=6)
            self.assertAlmostEqual(qw, math.sqrt(0.5), places=6)

    def test_hand_selection_parks_unselected_controller(self):
        frame = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(
                hmd_base=(0.0, 1.0, 0.0),
                hands=HandSelection.LEFT,
            ),
        )

        self.assertEqual(frame.left_enable, 5)
        self.assertEqual(frame.right_enable, 0)
        self.assertEqual(frame.body_enable, 0)

    def test_profile_body_pose_anchors_hips_and_feet_with_neutral_rotations(self):
        hmd_base = (1.5, 1.0, -2.0)
        frame = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(
                hmd_base=hmd_base,
                body_tracking=BodyTrackingMode.PROFILE,
                body_enable=7,
            ),
        )
        scale = TEST_PROFILE.scale_for_hmd_height(hmd_base[1])

        self.assertEqual(frame.body_enable, 7)
        self.assertFalse(frame.frame.tracker_activation.left)
        self.assertFalse(frame.frame.tracker_activation.right)
        self.assertTrue(frame.frame.tracker_activation.hips)
        self.assertTrue(frame.frame.tracker_activation.left_foot)
        self.assertTrue(frame.frame.tracker_activation.right_foot)
        for bone_name, tracker in (
            ("hips", frame.frame.hips),
            ("leftFoot", frame.frame.left_foot),
            ("rightFoot", frame.frame.right_foot),
        ):
            bone = TEST_PROFILE.bone(bone_name)
            expected_position = tuple(
                hmd_base[axis] + bone.position_m[axis] * scale for axis in range(3)
            )
            np.testing.assert_allclose(tracker.position, expected_position, atol=1e-6)
            self.assertEqual(tracker.quaternion_xyzw, (0.0, 0.0, 0.0, 1.0))

    def test_reach_fraction_cannot_request_an_unreachable_target(self):
        with self.assertRaises(ValueError):
            NeutralPoseConfig(hmd_base=(0.0, 1.0, 0.0), reach_fraction=1.01)


if __name__ == "__main__":
    unittest.main()
