from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vrc_ardy_agent.avatar_rig_profile import AvatarRigProfile


def _bone(
    name: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict[str, object]:
    return {
        "name": name,
        "positionMeters": list(position),
        "rotationXyzw": list(rotation),
    }


def _profile_document() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "profileName": "test-avatar",
        "units": "meters",
        "coordinateSpace": "avatar_root_axes_head_relative",
        "referencePose": "current_editor_pose",
        "floorReference": "toes",
        "referenceHeightMeters": 1.0,
        "bones": [
            _bone("head", (0.0, 0.0, 0.0)),
            _bone("hips", (0.0, -0.50, 0.01)),
            _bone("leftUpperArm", (-0.10, -0.10, 0.0)),
            _bone("leftLowerArm", (-0.25, -0.10, 0.0)),
            _bone("leftHand", (-0.40, -0.10, 0.0)),
            _bone("rightUpperArm", (0.10, -0.10, 0.0)),
            _bone("rightLowerArm", (0.25, -0.10, 0.0)),
            _bone("rightHand", (0.40, -0.10, 0.0)),
            _bone("leftUpperLeg", (-0.05, -0.52, 0.0)),
            _bone("leftLowerLeg", (-0.05, -0.75, 0.0)),
            _bone("leftFoot", (-0.05, -0.95, 0.05)),
            _bone("rightUpperLeg", (0.05, -0.52, 0.0)),
            _bone("rightLowerLeg", (0.05, -0.75, 0.0)),
            _bone("rightFoot", (0.05, -0.95, 0.05)),
            _bone("leftToes", (-0.05, -1.0, 0.15)),
            _bone("rightToes", (0.05, -1.0, 0.15)),
        ],
    }


class AvatarRigProfileTests(unittest.TestCase):
    def test_loads_exported_profile_and_derives_body_measurements(self) -> None:
        profile = AvatarRigProfile.from_mapping(_profile_document())

        self.assertEqual(profile.name, "test-avatar")
        self.assertAlmostEqual(profile.shoulder_half_width_m, 0.10)
        self.assertAlmostEqual(profile.head_to_shoulder_drop_m, 0.10)
        self.assertAlmostEqual(profile.arm_reach_m, 0.30)
        self.assertAlmostEqual(profile.leg_length_m, 0.436155281280883)

    def test_maps_reference_tracker_positions_to_the_hmd_anchor(self) -> None:
        profile = AvatarRigProfile.from_mapping(_profile_document())

        poses = profile.body_tracker_poses(hmd_base=(1.0, 1.5, -2.0))

        self.assertEqual(set(poses), {"hips", "leftFoot", "rightFoot"})
        for actual, expected in (
            (poses["hips"].position_m, (1.0, 0.75, -1.985)),
            (poses["leftFoot"].position_m, (0.925, 0.075, -1.925)),
            (poses["rightFoot"].position_m, (1.075, 0.075, -1.925)),
        ):
            for actual_component, expected_component in zip(actual, expected, strict=True):
                self.assertAlmostEqual(actual_component, expected_component)
        self.assertEqual(poses["hips"].rotation_xyzw, (0.0, 0.0, 0.0, 1.0))

    def test_reference_bone_rotation_is_not_used_as_tracker_mount_rotation(self) -> None:
        document = _profile_document()
        hips = next(bone for bone in document["bones"] if bone["name"] == "hips")
        hips["rotationXyzw"] = [0.0, 2**-0.5, 0.0, 2**-0.5]
        profile = AvatarRigProfile.from_mapping(document)

        poses = profile.body_tracker_poses(hmd_base=(0.0, 1.0, 0.0))

        self.assertEqual(poses["hips"].rotation_xyzw, (0.0, 0.0, 0.0, 1.0))

    def test_loads_profile_from_a_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "avatar.avatar-rig.json"
            path.write_text(json.dumps(_profile_document()), encoding="utf-8")

            profile = AvatarRigProfile.load(path)

        self.assertEqual(profile.source_path, path)
        self.assertEqual(profile.bone("leftFoot").position_m, (-0.05, -0.95, 0.05))

    def test_rejects_a_missing_required_humanoid_bone(self) -> None:
        document = _profile_document()
        document["bones"] = [
            bone for bone in document["bones"] if bone["name"] != "rightFoot"
        ]

        with self.assertRaisesRegex(ValueError, "rightFoot"):
            AvatarRigProfile.from_mapping(document)

    def test_accepts_explicit_foot_floor_reference_when_toes_are_unavailable(self) -> None:
        document = _profile_document()
        document["floorReference"] = "feet"
        document["referenceHeightMeters"] = 0.95
        document["bones"] = [
            bone for bone in document["bones"] if bone["name"] not in {"leftToes", "rightToes"}
        ]

        profile = AvatarRigProfile.from_mapping(document)

        self.assertEqual(profile.floor_reference, "feet")
        self.assertAlmostEqual(profile.reference_height_m, 0.95)

    def test_rejects_height_that_disagrees_with_the_floor_reference(self) -> None:
        document = _profile_document()
        document["referenceHeightMeters"] = 1.2

        with self.assertRaisesRegex(ValueError, "referenceHeightMeters"):
            AvatarRigProfile.from_mapping(document)

    def test_rejects_an_unexpected_coordinate_space(self) -> None:
        document = _profile_document()
        document["coordinateSpace"] = "world"

        with self.assertRaisesRegex(ValueError, "coordinateSpace"):
            AvatarRigProfile.from_mapping(document)

    def test_rejects_a_non_normalized_rotation(self) -> None:
        document = _profile_document()
        document["bones"][0]["rotationXyzw"] = [0.0, 0.0, 0.0, 2.0]

        with self.assertRaisesRegex(ValueError, "normalized"):
            AvatarRigProfile.from_mapping(document)

    def test_rejects_a_head_position_that_is_not_the_profile_origin(self) -> None:
        document = _profile_document()
        document["bones"][0]["positionMeters"] = [0.0, 0.01, 0.0]

        with self.assertRaisesRegex(ValueError, "head position"):
            AvatarRigProfile.from_mapping(document)

    def test_rejects_duplicate_bone_names(self) -> None:
        document = _profile_document()
        document["bones"].append(_bone("hips", (0.0, -0.5, 0.0)))

        with self.assertRaisesRegex(ValueError, "duplicate bone"):
            AvatarRigProfile.from_mapping(document)


if __name__ == "__main__":
    unittest.main()
