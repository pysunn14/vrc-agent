from __future__ import annotations

import math
import unittest

import numpy as np

from vrc_ardy_agent.coordinate_space import (
    ardy_to_unity_position,
    ardy_to_unity_rotation,
)
from vrc_ardy_agent.lower_body_retargeting import (
    ARDY_LOWER_BODY_JOINTS,
    LowerBodyRotationMode,
    LowerBodyRetargeter,
    LowerBodyRetargetingMonitor,
    LowerBodySafetyLimits,
)
from vrc_ardy_agent.tracking_rig import get_avatar_profile

from tests.test_avatar_rig_profile import _profile_document
from vrc_ardy_agent.avatar_rig_profile import AvatarRigProfile

TEST_PROFILE = AvatarRigProfile.from_mapping(_profile_document())


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)),
        dtype=np.float64,
    )


def _source_pose() -> tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((27, 3), dtype=np.float64)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 27, axis=0)
    positions[ARDY_LOWER_BODY_JOINTS.hips] = (0.0, 1.1, 0.0)
    positions[ARDY_LOWER_BODY_JOINTS.left_foot] = (0.15, 0.1, 0.05)
    positions[ARDY_LOWER_BODY_JOINTS.right_foot] = (-0.15, 0.1, 0.05)
    return positions, rotations


def _permissive_limits() -> LowerBodySafetyLimits:
    return LowerBodySafetyLimits(
        max_hips_offset_hmd_fraction=100.0,
        max_foot_offset_hmd_fraction=100.0,
        max_leg_extension_fraction=100.0,
        max_hips_speed_hmd_fraction_per_second=1000.0,
        max_foot_speed_hmd_fraction_per_second=1000.0,
    )


class LowerBodyRetargeterTests(unittest.TestCase):
    def _retargeter(
        self,
        *,
        limits: LowerBodySafetyLimits | None = None,
        monitor: LowerBodyRetargetingMonitor | None = None,
        rotation_mode: LowerBodyRotationMode = LowerBodyRotationMode.SOURCE_DELTA,
    ) -> LowerBodyRetargeter:
        return LowerBodyRetargeter(
            avatar_profile=TEST_PROFILE,
            source_joints=ARDY_LOWER_BODY_JOINTS,
            hmd_base=(0.0, 1.0, 0.0),
            body_scale=0.5,
            fps=20.0,
            limits=limits or _permissive_limits(),
            monitor=monitor,
            rotation_mode=rotation_mode,
        )

    def test_first_source_pose_maps_to_synthetic_reference_pose(self) -> None:
        positions, rotations = _source_pose()
        positions[ARDY_LOWER_BODY_JOINTS.hips] = (4.0, 3.0, -2.0)
        positions[ARDY_LOWER_BODY_JOINTS.left_foot] = (3.4, -1.0, 8.0)
        positions[ARDY_LOWER_BODY_JOINTS.right_foot] = (-2.1, 0.4, -6.0)
        retargeter = self._retargeter()

        body = retargeter.retarget(positions, rotations)
        reference = TEST_PROFILE.body_tracker_poses(hmd_base=(0.0, 1.0, 0.0))

        for bone_name, tracker in (
            ("hips", body.hips),
            ("leftFoot", body.left_foot),
            ("rightFoot", body.right_foot),
        ):
            np.testing.assert_allclose(
                tracker.position,
                reference[bone_name].position_m,
                atol=1e-9,
            )
            np.testing.assert_allclose(tracker.rotation, np.eye(3), atol=1e-9)

    def test_source_deltas_are_composed_onto_synthetic_reference_pose(self) -> None:
        positions, rotations = _source_pose()
        rotations[ARDY_LOWER_BODY_JOINTS.hips] = _rotation_y(0.2)
        retargeter = self._retargeter()
        retargeter.retarget(positions, rotations)

        moved = positions.copy()
        moved_rotations = rotations.copy()
        deltas = {
            "hips": np.array((0.2, -0.1, 0.1)),
            "leftFoot": np.array((0.1, 0.2, -0.1)),
            "rightFoot": np.array((-0.1, 0.05, 0.2)),
        }
        for name, index in (
            ("hips", ARDY_LOWER_BODY_JOINTS.hips),
            ("leftFoot", ARDY_LOWER_BODY_JOINTS.left_foot),
            ("rightFoot", ARDY_LOWER_BODY_JOINTS.right_foot),
        ):
            moved[index] += deltas[name]
        source_rotation_delta = _rotation_y(-0.35)
        moved_rotations[ARDY_LOWER_BODY_JOINTS.hips] = (
            source_rotation_delta @ rotations[ARDY_LOWER_BODY_JOINTS.hips]
        )

        body = retargeter.retarget(moved, moved_rotations)
        reference = TEST_PROFILE.body_tracker_poses(hmd_base=(0.0, 1.0, 0.0))

        for bone_name, tracker in (
            ("hips", body.hips),
            ("leftFoot", body.left_foot),
            ("rightFoot", body.right_foot),
        ):
            expected = np.asarray(
                reference[bone_name].position_m
            ) + ardy_to_unity_position(deltas[bone_name] * 0.5)
            np.testing.assert_allclose(tracker.position, expected, atol=1e-9)
        np.testing.assert_allclose(
            body.hips.rotation,
            ardy_to_unity_rotation(source_rotation_delta),
            atol=1e-9,
        )

    def test_profile_neutral_rotation_mode_ignores_source_rotation_deltas(self) -> None:
        positions, rotations = _source_pose()
        retargeter = self._retargeter(
            rotation_mode=LowerBodyRotationMode.PROFILE_NEUTRAL,
        )
        retargeter.retarget(positions, rotations)
        moved_rotations = rotations.copy()
        moved_rotations[ARDY_LOWER_BODY_JOINTS.hips] = _rotation_y(0.8)
        moved_rotations[ARDY_LOWER_BODY_JOINTS.left_foot] = _rotation_y(-0.6)
        moved_rotations[ARDY_LOWER_BODY_JOINTS.right_foot] = _rotation_y(0.4)

        body = retargeter.retarget(positions, moved_rotations)

        for tracker in (body.hips, body.left_foot, body.right_foot):
            np.testing.assert_allclose(tracker.rotation, np.eye(3), atol=1e-9)

    def test_foot_cannot_move_below_its_profile_floor_clearance(self) -> None:
        monitor = LowerBodyRetargetingMonitor()
        positions, rotations = _source_pose()
        retargeter = self._retargeter(monitor=monitor)
        retargeter.retarget(positions, rotations)
        moved = positions.copy()
        moved[ARDY_LOWER_BODY_JOINTS.left_foot, 1] -= 2.0

        body = retargeter.retarget(moved, rotations)
        reference = TEST_PROFILE.body_tracker_poses(hmd_base=(0.0, 1.0, 0.0))

        self.assertGreaterEqual(
            body.left_foot.position[1],
            reference["leftFoot"].position_m[1],
        )
        self.assertEqual(monitor.snapshot().floor_clamp_count, 1)

    def test_leg_reach_is_limited_by_the_target_avatar(self) -> None:
        limits = _permissive_limits()
        limits = LowerBodySafetyLimits(
            max_hips_offset_hmd_fraction=limits.max_hips_offset_hmd_fraction,
            max_foot_offset_hmd_fraction=limits.max_foot_offset_hmd_fraction,
            max_leg_extension_fraction=1.0,
            max_hips_speed_hmd_fraction_per_second=(
                limits.max_hips_speed_hmd_fraction_per_second
            ),
            max_foot_speed_hmd_fraction_per_second=(
                limits.max_foot_speed_hmd_fraction_per_second
            ),
        )
        positions, rotations = _source_pose()
        retargeter = self._retargeter(limits=limits)
        retargeter.retarget(positions, rotations)
        moved = positions.copy()
        moved[ARDY_LOWER_BODY_JOINTS.left_foot, 2] += 10.0

        body = retargeter.retarget(moved, rotations)

        self.assertLessEqual(
            np.linalg.norm(body.left_foot.position - body.hips.position),
            retargeter.maximum_leg_reach_m("left") + 1e-9,
        )
        self.assertEqual(retargeter.monitor.snapshot().leg_clamp_count, 1)

    def test_output_speed_limit_contains_chunk_discontinuities(self) -> None:
        limits = LowerBodySafetyLimits(
            max_hips_offset_hmd_fraction=100.0,
            max_foot_offset_hmd_fraction=100.0,
            max_leg_extension_fraction=100.0,
            max_hips_speed_hmd_fraction_per_second=1000.0,
            max_foot_speed_hmd_fraction_per_second=1.0,
        )
        positions, rotations = _source_pose()
        retargeter = self._retargeter(limits=limits)
        first = retargeter.retarget(positions, rotations)
        moved = positions.copy()
        moved[ARDY_LOWER_BODY_JOINTS.left_foot, 0] += 10.0

        second = retargeter.retarget(moved, rotations)

        displacement = np.linalg.norm(
            second.left_foot.position - first.left_foot.position
        )
        self.assertLessEqual(displacement, 1.0 / 20.0 + 1e-9)
        self.assertEqual(retargeter.monitor.snapshot().speed_clamp_count, 1)

    def test_non_finite_source_pose_is_rejected(self) -> None:
        positions, rotations = _source_pose()
        positions[ARDY_LOWER_BODY_JOINTS.hips, 0] = math.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            self._retargeter().retarget(positions, rotations)


if __name__ == "__main__":
    unittest.main()
