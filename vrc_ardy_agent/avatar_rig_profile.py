from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = 1
UNITS = "meters"
COORDINATE_SPACE = "avatar_root_axes_head_relative"
REFERENCE_POSE = "current_editor_pose"
FLOOR_REFERENCES = frozenset(("feet", "toes"))

# Shoulder bones and toe bones are optional in Unity Humanoid avatars. The
# segment roots below are the smallest set required for arm and leg scaling.
REQUIRED_BONES = frozenset(
    (
        "head",
        "hips",
        "leftUpperArm",
        "leftLowerArm",
        "leftHand",
        "rightUpperArm",
        "rightLowerArm",
        "rightHand",
        "leftUpperLeg",
        "leftLowerLeg",
        "leftFoot",
        "rightUpperLeg",
        "rightLowerLeg",
        "rightFoot",
    )
)


def _finite_vector(value: object, *, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array of {length} numbers")
    if len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    converted: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain only finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers")
        converted.append(number)
    return tuple(converted)


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


@dataclass(frozen=True)
class BonePose:
    position_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> tuple[str, BonePose]:
        if not isinstance(value, Mapping):
            raise ValueError(f"bones[{index}] must be an object")
        expected = {"name", "positionMeters", "rotationXyzw"}
        if set(value) != expected:
            raise ValueError(
                f"bones[{index}] must contain exactly: {', '.join(sorted(expected))}"
            )
        name = value["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"bones[{index}].name must be a non-empty string")
        position = _finite_vector(
            value["positionMeters"],
            name=f"bones[{index}].positionMeters",
            length=3,
        )
        rotation = _finite_vector(
            value["rotationXyzw"],
            name=f"bones[{index}].rotationXyzw",
            length=4,
        )
        norm = math.sqrt(sum(component * component for component in rotation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f"bones[{index}].rotationXyzw must be normalized")
        return name, cls(
            position_m=(position[0], position[1], position[2]),
            rotation_xyzw=(rotation[0], rotation[1], rotation[2], rotation[3]),
        )


@dataclass(frozen=True)
class BodyTrackerPose:
    position_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class AvatarRigProfile:
    name: str
    reference_height_m: float
    floor_reference: str
    bones: Mapping[str, BonePose]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bones", MappingProxyType(dict(self.bones)))

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        source_path: Path | None = None,
    ) -> AvatarRigProfile:
        if not isinstance(value, Mapping):
            raise ValueError("avatar rig profile must be a JSON object")

        expected = {
            "schemaVersion",
            "profileName",
            "units",
            "coordinateSpace",
            "referencePose",
            "floorReference",
            "referenceHeightMeters",
            "bones",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            unexpected = sorted(set(value) - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ValueError("invalid avatar rig profile fields: " + " ".join(details))

        if type(value["schemaVersion"]) is not int or value["schemaVersion"] != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
        if value["units"] != UNITS:
            raise ValueError(f"units must be {UNITS!r}")
        if value["coordinateSpace"] != COORDINATE_SPACE:
            raise ValueError(f"coordinateSpace must be {COORDINATE_SPACE!r}")
        if value["referencePose"] != REFERENCE_POSE:
            raise ValueError(f"referencePose must be {REFERENCE_POSE!r}")

        name = value["profileName"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("profileName must be a non-empty string")
        floor_reference = value["floorReference"]
        if floor_reference not in FLOOR_REFERENCES:
            raise ValueError("floorReference must be 'toes' or 'feet'")
        reference_height = _positive_number(
            value["referenceHeightMeters"],
            name="referenceHeightMeters",
        )

        raw_bones = value["bones"]
        if not isinstance(raw_bones, Sequence) or isinstance(raw_bones, (str, bytes)):
            raise ValueError("bones must be an array")
        bones: dict[str, BonePose] = {}
        for index, raw_bone in enumerate(raw_bones):
            bone_name, bone_pose = BonePose.from_mapping(raw_bone, index=index)
            if bone_name in bones:
                raise ValueError(f"duplicate bone name: {bone_name}")
            bones[bone_name] = bone_pose

        missing_bones = sorted(REQUIRED_BONES - set(bones))
        if missing_bones:
            raise ValueError(f"missing required humanoid bones: {', '.join(missing_bones)}")
        if any(abs(component) > 1e-5 for component in bones["head"].position_m):
            raise ValueError("head position must be the profile origin")

        floor_bones = (
            ("leftToes", "rightToes")
            if floor_reference == "toes"
            else ("leftFoot", "rightFoot")
        )
        missing_floor_bones = [bone_name for bone_name in floor_bones if bone_name not in bones]
        if missing_floor_bones:
            raise ValueError(
                f"floorReference {floor_reference!r} requires: "
                + ", ".join(missing_floor_bones)
            )
        derived_height = -sum(bones[bone].position_m[1] for bone in floor_bones) / 2.0
        if not math.isclose(
            reference_height,
            derived_height,
            rel_tol=0.0,
            abs_tol=1e-4,
        ):
            raise ValueError(
                "referenceHeightMeters does not match the declared floorReference"
            )

        profile = cls(
            name=name.strip(),
            reference_height_m=reference_height,
            floor_reference=floor_reference,
            bones=bones,
            source_path=source_path,
        )
        profile._validate_derived_measurements()
        return profile

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        source_path: Path | None = None,
    ) -> AvatarRigProfile:
        try:
            document: Any = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid avatar rig profile JSON: {exc.msg}") from exc
        return cls.from_mapping(document, source_path=source_path)

    @classmethod
    def load(cls, path: str | Path) -> AvatarRigProfile:
        source_path = Path(path)
        try:
            document = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read avatar rig profile {source_path}: {exc}") from exc
        return cls.from_json(document, source_path=source_path)

    def bone(self, name: str) -> BonePose:
        try:
            return self.bones[name]
        except KeyError as exc:
            raise ValueError(f"avatar rig profile has no bone {name!r}") from exc

    @property
    def shoulder_half_width_m(self) -> float:
        left = self.bone("leftUpperArm").position_m
        right = self.bone("rightUpperArm").position_m
        return (right[0] - left[0]) * 0.5

    @property
    def head_to_shoulder_drop_m(self) -> float:
        left = self.bone("leftUpperArm").position_m
        right = self.bone("rightUpperArm").position_m
        return -(left[1] + right[1]) * 0.5

    @property
    def arm_reach_m(self) -> float:
        reaches = []
        for side in ("left", "right"):
            upper = self.bone(f"{side}UpperArm").position_m
            lower = self.bone(f"{side}LowerArm").position_m
            hand = self.bone(f"{side}Hand").position_m
            reaches.append(_distance(upper, lower) + _distance(lower, hand))
        return sum(reaches) * 0.5

    @property
    def leg_length_m(self) -> float:
        lengths = []
        for side in ("left", "right"):
            upper = self.bone(f"{side}UpperLeg").position_m
            lower = self.bone(f"{side}LowerLeg").position_m
            foot = self.bone(f"{side}Foot").position_m
            lengths.append(_distance(upper, lower) + _distance(lower, foot))
        return sum(lengths) * 0.5

    def scale_for_hmd_height(self, hmd_height_m: float) -> float:
        height = _positive_number(hmd_height_m, name="HMD height")
        return height / self.reference_height_m

    def body_tracker_poses(
        self,
        *,
        hmd_base: tuple[float, float, float],
    ) -> Mapping[str, BodyTrackerPose]:
        base = _finite_vector(hmd_base, name="hmd_base", length=3)
        if base[1] <= 0:
            raise ValueError("hmd_base Y must be positive")
        scale = self.scale_for_hmd_height(base[1])
        result: dict[str, BodyTrackerPose] = {}
        for bone_name in ("hips", "leftFoot", "rightFoot"):
            bone = self.bone(bone_name)
            result[bone_name] = BodyTrackerPose(
                position_m=(
                    base[0] + bone.position_m[0] * scale,
                    base[1] + bone.position_m[1] * scale,
                    base[2] + bone.position_m[2] * scale,
                ),
                # A Unity humanoid bone rotation describes the skeleton bind
                # pose, not the orientation of a physical SteamVR tracker.
                # Reusing it as a tracker mount rotation twists the solved
                # avatar. Static synthetic trackers therefore start aligned
                # with room space; dynamic retargeting composes its own deltas.
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        return MappingProxyType(result)

    def _validate_derived_measurements(self) -> None:
        for name, value in (
            ("shoulder width", self.shoulder_half_width_m),
            ("head-to-shoulder drop", self.head_to_shoulder_drop_m),
            ("arm reach", self.arm_reach_m),
            ("leg length", self.leg_length_m),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"derived {name} must be finite and positive")
