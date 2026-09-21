from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from .avatar_rig_profile import AvatarRigProfile
from .opentrack_bridge import OpenTrackFrame
from .six_point_bridge import SixPointFrame, TrackerActivation
from .vmt_bridge import VmtFrame


def _validate_vector(values: tuple[float, ...], name: str) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain only finite values")


class HandSelection(str, Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"

    @property
    def includes_left(self) -> bool:
        return self in (HandSelection.LEFT, HandSelection.BOTH)

    @property
    def includes_right(self) -> bool:
        return self in (HandSelection.RIGHT, HandSelection.BOTH)


class BodyTrackingMode(str, Enum):
    OFF = "off"
    PROFILE = "profile"


@dataclass(frozen=True)
class RigCalibration:
    name: str
    reference_head_height_m: float
    shoulder_half_width_m: float
    head_to_shoulder_drop_m: float
    arm_reach_m: float
    max_arm_extension_fraction: float = 0.92
    left_hand_rest_euler_deg: tuple[float, float, float] = (90.0, 0.0, 0.0)
    right_hand_rest_euler_deg: tuple[float, float, float] = (90.0, 0.0, 0.0)

    @classmethod
    def from_avatar_profile(
        cls,
        profile: AvatarRigProfile,
    ) -> RigCalibration:
        return cls(
            name=profile.name.casefold(),
            reference_head_height_m=profile.reference_height_m,
            shoulder_half_width_m=profile.shoulder_half_width_m,
            head_to_shoulder_drop_m=profile.head_to_shoulder_drop_m,
            arm_reach_m=profile.arm_reach_m,
        )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("avatar profile name must not be empty")
        for field_name in (
            "reference_head_height_m",
            "shoulder_half_width_m",
            "head_to_shoulder_drop_m",
            "arm_reach_m",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if (
            not math.isfinite(self.max_arm_extension_fraction)
            or not 0 < self.max_arm_extension_fraction <= 1
        ):
            raise ValueError("max_arm_extension_fraction must be in (0, 1]")
        _validate_vector(self.left_hand_rest_euler_deg, "left_hand_rest_euler_deg")
        _validate_vector(self.right_hand_rest_euler_deg, "right_hand_rest_euler_deg")

    def scale_for_hmd_height(self, hmd_height_m: float) -> float:
        height = float(hmd_height_m)
        if not math.isfinite(height) or height <= 0:
            raise ValueError("HMD height must be finite and positive")
        return height / self.reference_head_height_m

    def scaled_arm_reach(self, hmd_height_m: float) -> float:
        return self.arm_reach_m * self.scale_for_hmd_height(hmd_height_m)

    def shoulder_position(
        self,
        hmd_base: tuple[float, float, float],
        *,
        side: str,
    ) -> tuple[float, float, float]:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        _validate_vector(hmd_base, "hmd_base")
        scale = self.scale_for_hmd_height(hmd_base[1])
        sign = -1.0 if side == "left" else 1.0
        return (
            float(hmd_base[0]) + sign * self.shoulder_half_width_m * scale,
            float(hmd_base[1]) - self.head_to_shoulder_drop_m * scale,
            float(hmd_base[2]),
        )


def _load_bundled_avatar_profiles(directory: Path) -> Mapping[str, AvatarRigProfile]:
    paths = sorted(directory.glob("*.avatar-rig.json"))
    # Avatar data is user-owned and is not shipped with the repository.
    # An empty local catalog is valid; selection still validates an explicit file.

    profiles: dict[str, AvatarRigProfile] = {}
    for path in paths:
        profile = AvatarRigProfile.load(path)
        key = profile.name.casefold()
        if key in profiles:
            raise RuntimeError(f"duplicate bundled avatar profile name: {profile.name}")
        profiles[key] = profile
    return MappingProxyType(profiles)


_PROFILE_DIRECTORY = Path(__file__).resolve().with_name("rig_profiles")
AVATAR_PROFILES = _load_bundled_avatar_profiles(_PROFILE_DIRECTORY)


@dataclass(frozen=True)
class NeutralPoseConfig:
    hmd_base: tuple[float, float, float]
    fps: float = 20.0
    hands: HandSelection = HandSelection.NONE
    reach_fraction: float = 0.85
    # Outward, downward, and forward components of the shoulder-to-hand ray.
    hand_direction: tuple[float, float, float] = (0.25, 0.94, 0.20)
    # Unity-style Euler order: Y * X * Z. A positive X quarter-turn points the
    # controller's Unity-forward axis down while the arm rests beside the body.
    left_hand_euler_deg: tuple[float, float, float] = (90.0, 0.0, 0.0)
    right_hand_euler_deg: tuple[float, float, float] = (90.0, 0.0, 0.0)
    body_tracking: BodyTrackingMode = BodyTrackingMode.OFF
    left_enable: int = 5
    right_enable: int = 6
    body_enable: int = 7

    def __post_init__(self) -> None:
        _validate_vector(self.hmd_base, "hmd_base")
        if self.hmd_base[1] <= 0:
            raise ValueError("hmd_base Y must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not math.isfinite(self.reach_fraction) or not 0 < self.reach_fraction <= 1:
            raise ValueError("reach_fraction must be in (0, 1]")
        _validate_vector(self.hand_direction, "hand_direction")
        if self.hand_direction[1] <= 0:
            raise ValueError("hand_direction DOWN component must be positive")
        _validate_vector(self.left_hand_euler_deg, "left_hand_euler_deg")
        _validate_vector(self.right_hand_euler_deg, "right_hand_euler_deg")
        if not isinstance(self.hands, HandSelection):
            raise TypeError("hands must be a HandSelection")
        if not isinstance(self.body_tracking, BodyTrackingMode):
            raise TypeError("body_tracking must be a BodyTrackingMode")
        if self.left_enable <= 0 or self.right_enable <= 0:
            raise ValueError("controller enable modes must be positive")
        if self.body_enable <= 0:
            raise ValueError("body tracker enable mode must be positive")


@dataclass(frozen=True)
class NeutralTrackingFrame:
    frame: SixPointFrame
    left_enable: int
    right_enable: int
    body_enable: int = 0

    @property
    def head(self) -> OpenTrackFrame:
        return self.frame.head

    @property
    def left(self) -> VmtFrame:
        return self.frame.left

    @property
    def right(self) -> VmtFrame:
        return self.frame.right


def get_avatar_profile(name: str) -> AvatarRigProfile:
    if Path(name).expanduser().is_file():
        return AvatarRigProfile.load(Path(name).expanduser())
    try:
        return AVATAR_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(AVATAR_PROFILES))
        raise ValueError(
            f"unknown avatar profile {name!r}; expected one of: {choices}"
        ) from exc


def build_neutral_tracking_frame(
    avatar_profile: AvatarRigProfile,
    config: NeutralPoseConfig,
) -> NeutralTrackingFrame:
    calibration = RigCalibration.from_avatar_profile(avatar_profile)
    direction_out, direction_down, direction_forward = config.hand_direction
    direction_norm = math.sqrt(
        direction_out * direction_out
        + direction_down * direction_down
        + direction_forward * direction_forward
    )
    if direction_norm <= 1e-9:
        raise ValueError("hand_direction must have non-zero length")
    direction_out /= direction_norm
    direction_down /= direction_norm
    direction_forward /= direction_norm

    reach = calibration.scaled_arm_reach(config.hmd_base[1]) * config.reach_fraction

    def hand_pose(side: str, euler_deg: tuple[float, float, float]) -> VmtFrame:
        shoulder = calibration.shoulder_position(config.hmd_base, side=side)
        sign = -1.0 if side == "left" else 1.0
        return VmtFrame(
            position=(
                shoulder[0] + sign * direction_out * reach,
                shoulder[1] - direction_down * reach,
                shoulder[2] + direction_forward * reach,
            ),
            quaternion_xyzw=unity_euler_to_quaternion_xyzw(euler_deg),
            fps=config.fps,
        )

    left = hand_pose("left", config.left_hand_euler_deg)
    right = hand_pose("right", config.right_hand_euler_deg)
    inactive = VmtFrame(
        position=(0.0, 0.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        fps=config.fps,
    )
    hips = inactive
    left_foot = inactive
    right_foot = inactive
    body_enable = 0
    if config.body_tracking is BodyTrackingMode.PROFILE:
        body_poses = avatar_profile.body_tracker_poses(hmd_base=config.hmd_base)

        def body_frame(bone_name: str) -> VmtFrame:
            pose = body_poses[bone_name]
            return VmtFrame(
                position=pose.position_m,
                quaternion_xyzw=pose.rotation_xyzw,
                fps=config.fps,
            )

        hips = body_frame("hips")
        left_foot = body_frame("leftFoot")
        right_foot = body_frame("rightFoot")
        body_enable = config.body_enable

    frame = SixPointFrame(
        head=OpenTrackFrame(
            xyz_cm=(0.0, 0.0, 0.0),
            ypr_deg=(0.0, 0.0, 0.0),
            fps=config.fps,
        ),
        left=left,
        right=right,
        hips=hips,
        left_foot=left_foot,
        right_foot=right_foot,
        fps=config.fps,
        scale=calibration.scale_for_hmd_height(config.hmd_base[1]),
        locomotion_x=0.0,
        locomotion_y=0.0,
        locomotion_turn=0.0,
        tracker_activation=TrackerActivation(
            left=config.hands.includes_left,
            right=config.hands.includes_right,
            hips=config.body_tracking is BodyTrackingMode.PROFILE,
            left_foot=config.body_tracking is BodyTrackingMode.PROFILE,
            right_foot=config.body_tracking is BodyTrackingMode.PROFILE,
        ),
    )
    return NeutralTrackingFrame(
        frame=frame,
        left_enable=config.left_enable if config.hands.includes_left else 0,
        right_enable=config.right_enable if config.hands.includes_right else 0,
        body_enable=body_enable,
    )


def unity_euler_to_quaternion_xyzw(
    euler_deg: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    _validate_vector(euler_deg, "euler_deg")
    x_deg, y_deg, z_deg = euler_deg
    qx = _axis_angle_quaternion((1.0, 0.0, 0.0), math.radians(x_deg))
    qy = _axis_angle_quaternion((0.0, 1.0, 0.0), math.radians(y_deg))
    qz = _axis_angle_quaternion((0.0, 0.0, 1.0), math.radians(z_deg))
    return _normalize_quaternion(_multiply_quaternion(_multiply_quaternion(qy, qx), qz))


def _axis_angle_quaternion(
    axis: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float, float]:
    half = angle_rad * 0.5
    scale = math.sin(half)
    return (axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half))


def _multiply_quaternion(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _normalize_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-9:
        raise ValueError("quaternion must have non-zero length")
    return tuple(value / norm for value in quaternion)
