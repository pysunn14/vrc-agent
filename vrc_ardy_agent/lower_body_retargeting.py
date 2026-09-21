from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .avatar_rig_profile import AvatarRigProfile
from .coordinate_space import ardy_to_unity_position, ardy_to_unity_rotation


@dataclass(frozen=True, slots=True)
class LowerBodyJoints:
    hips: int
    left_foot: int
    right_foot: int
    left_toe: int
    right_toe: int

    def __post_init__(self) -> None:
        indices = (
            self.hips,
            self.left_foot,
            self.right_foot,
            self.left_toe,
            self.right_toe,
        )
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise TypeError("lower-body joint indices must be integers")
        if any(index < 0 for index in indices):
            raise ValueError("lower-body joint indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("lower-body joint indices must be distinct")


# ARDY CoreSkeleton27 mapping. Source-model metadata belongs here instead of in
# the Felis profile so another avatar can reuse the same ARDY interpretation.
ARDY_LOWER_BODY_JOINTS = LowerBodyJoints(
    hips=0,
    left_foot=25,
    right_foot=21,
    left_toe=26,
    right_toe=22,
)


class LowerBodyRotationMode(str, Enum):
    SOURCE_DELTA = "source-delta"
    PROFILE_NEUTRAL = "profile-neutral"


@dataclass(frozen=True, slots=True)
class LowerBodySafetyLimits:
    max_hips_offset_hmd_fraction: float = 0.35
    max_foot_offset_hmd_fraction: float = 0.75
    max_leg_extension_fraction: float = 1.10
    max_hips_speed_hmd_fraction_per_second: float = 2.0
    max_foot_speed_hmd_fraction_per_second: float = 4.0

    def __post_init__(self) -> None:
        for field_name in (
            "max_hips_offset_hmd_fraction",
            "max_foot_offset_hmd_fraction",
            "max_leg_extension_fraction",
            "max_hips_speed_hmd_fraction_per_second",
            "max_foot_speed_hmd_fraction_per_second",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.max_leg_extension_fraction < 1.0:
            raise ValueError("max_leg_extension_fraction must be at least 1")


@dataclass(frozen=True, slots=True)
class RetargetedBodyTracker:
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True, slots=True)
class RetargetedLowerBody:
    hips: RetargetedBodyTracker
    left_foot: RetargetedBodyTracker
    right_foot: RetargetedBodyTracker


@dataclass(frozen=True, slots=True)
class LowerBodyRetargetingSnapshot:
    avatar_profile: str | None = None
    rotation_mode: str | None = None
    initialized: bool = False
    frames: int = 0
    workspace_clamp_count: int = 0
    floor_clamp_count: int = 0
    leg_clamp_count: int = 0
    speed_clamp_count: int = 0
    last_clamped_trackers: tuple[str, ...] = ()
    hips_position: tuple[float, float, float] | None = None
    left_foot_position: tuple[float, float, float] | None = None
    right_foot_position: tuple[float, float, float] | None = None
    heartbeat_monotonic: float = 0.0


class LowerBodyRetargetingMonitor:
    """Thread-safe lower-body safety and heartbeat state for agentctl."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = LowerBodyRetargetingSnapshot(
            heartbeat_monotonic=time.monotonic()
        )

    def reset(
        self,
        avatar_profile: str,
        rotation_mode: LowerBodyRotationMode,
    ) -> None:
        with self._lock:
            self._snapshot = LowerBodyRetargetingSnapshot(
                avatar_profile=avatar_profile,
                rotation_mode=rotation_mode.value,
                heartbeat_monotonic=time.monotonic(),
            )

    def initialized(self) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                initialized=True,
                heartbeat_monotonic=time.monotonic(),
            )

    def record(
        self,
        body: RetargetedLowerBody,
        *,
        workspace_clamps: tuple[str, ...],
        floor_clamps: tuple[str, ...],
        leg_clamps: tuple[str, ...],
        speed_clamps: tuple[str, ...],
    ) -> None:
        clamped = tuple(
            sorted(set(workspace_clamps + floor_clamps + leg_clamps + speed_clamps))
        )
        with self._lock:
            current = self._snapshot
            self._snapshot = replace(
                current,
                frames=current.frames + 1,
                workspace_clamp_count=(
                    current.workspace_clamp_count + len(workspace_clamps)
                ),
                floor_clamp_count=current.floor_clamp_count + len(floor_clamps),
                leg_clamp_count=current.leg_clamp_count + len(leg_clamps),
                speed_clamp_count=current.speed_clamp_count + len(speed_clamps),
                last_clamped_trackers=clamped,
                hips_position=_position_tuple(body.hips.position),
                left_foot_position=_position_tuple(body.left_foot.position),
                right_foot_position=_position_tuple(body.right_foot.position),
                heartbeat_monotonic=time.monotonic(),
            )

    def snapshot(self) -> LowerBodyRetargetingSnapshot:
        with self._lock:
            return self._snapshot


class LowerBodyRetargeter:
    """Compose ARDY lower-body deltas onto an avatar's calibrated rest pose.

    ARDY joint positions are not valid tracker coordinates for an arbitrary
    avatar. The first stable source frame is therefore used only as a delta
    origin. Absolute tracker placement always comes from the avatar profile.
    """

    def __init__(
        self,
        *,
        avatar_profile: AvatarRigProfile,
        source_joints: LowerBodyJoints,
        hmd_base: tuple[float, float, float],
        body_scale: float,
        fps: float,
        limits: LowerBodySafetyLimits | None = None,
        monitor: LowerBodyRetargetingMonitor | None = None,
        rotation_mode: LowerBodyRotationMode = LowerBodyRotationMode.SOURCE_DELTA,
    ) -> None:
        if len(hmd_base) != 3 or any(not math.isfinite(float(v)) for v in hmd_base):
            raise ValueError("hmd_base must contain three finite values")
        if hmd_base[1] <= 0:
            raise ValueError("hmd_base Y must be positive")
        if not math.isfinite(body_scale) or body_scale <= 0:
            raise ValueError("body_scale must be finite and positive")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not isinstance(rotation_mode, LowerBodyRotationMode):
            raise TypeError("rotation_mode must be a LowerBodyRotationMode")

        self.avatar_profile = avatar_profile
        self.source_joints = source_joints
        self.hmd_base = tuple(float(value) for value in hmd_base)
        self.body_scale = float(body_scale)
        self.fps = float(fps)
        self.limits = limits or LowerBodySafetyLimits()
        self.monitor = monitor or LowerBodyRetargetingMonitor()
        self.rotation_mode = rotation_mode

        poses = avatar_profile.body_tracker_poses(hmd_base=self.hmd_base)
        self._target_reference_positions = {
            name: np.asarray(poses[name].position_m, dtype=np.float64)
            for name in ("hips", "leftFoot", "rightFoot")
        }
        self._target_reference_rotations = {
            name: _quaternion_xyzw_to_matrix(poses[name].rotation_xyzw)
            for name in ("hips", "leftFoot", "rightFoot")
        }
        self._source_reference_positions: dict[str, np.ndarray] = {}
        self._source_reference_rotations: dict[str, np.ndarray] = {}
        self._previous_positions: dict[str, np.ndarray] = {}
        self._maximum_leg_reach = self._derive_maximum_leg_reach()
        self.monitor.reset(avatar_profile.name, self.rotation_mode)

    def reset(self) -> None:
        self._source_reference_positions.clear()
        self._source_reference_rotations.clear()
        self._previous_positions.clear()
        self.monitor.reset(self.avatar_profile.name, self.rotation_mode)

    def maximum_leg_reach_m(self, side: str) -> float:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return self._maximum_leg_reach[side]

    def retarget(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
    ) -> RetargetedLowerBody:
        source_positions = np.asarray(positions, dtype=np.float64)
        source_rotations = np.asarray(rotations, dtype=np.float64)
        self._validate_pose(source_positions, source_rotations)
        if not self._source_reference_positions:
            self._initialize(source_positions, source_rotations)

        output_positions: dict[str, np.ndarray] = {}
        output_rotations: dict[str, np.ndarray] = {}
        workspace_clamps: list[str] = []
        speed_clamps: list[str] = []
        for name, index in self._tracked_joints():
            source_delta = (
                source_positions[index] - self._source_reference_positions[name]
            )
            target = self._target_reference_positions[name] + ardy_to_unity_position(
                source_delta * self.body_scale
            )
            max_offset = (
                self.limits.max_hips_offset_hmd_fraction
                if name == "hips"
                else self.limits.max_foot_offset_hmd_fraction
            ) * self.hmd_base[1]
            target, workspace_clamped = _limit_distance(
                target,
                self._target_reference_positions[name],
                max_offset,
            )
            if workspace_clamped:
                workspace_clamps.append(name)

            previous = self._previous_positions.get(name)
            if previous is not None:
                max_speed = (
                    self.limits.max_hips_speed_hmd_fraction_per_second
                    if name == "hips"
                    else self.limits.max_foot_speed_hmd_fraction_per_second
                ) * self.hmd_base[1]
                target, speed_clamped = _limit_distance(
                    target,
                    previous,
                    max_speed / self.fps,
                )
                if speed_clamped:
                    speed_clamps.append(name)
            output_positions[name] = target

            if self.rotation_mode is LowerBodyRotationMode.PROFILE_NEUTRAL:
                output_rotations[name] = self._target_reference_rotations[name].copy()
            else:
                source_rotation_delta = (
                    source_rotations[index] @ self._source_reference_rotations[name].T
                )
                output_rotations[name] = ardy_to_unity_rotation(source_rotation_delta)

        floor_clamps: list[str] = []
        for name in ("leftFoot", "rightFoot"):
            minimum_y = self._target_reference_positions[name][1]
            if output_positions[name][1] < minimum_y:
                output_positions[name] = output_positions[name].copy()
                output_positions[name][1] = minimum_y
                floor_clamps.append(name)

        leg_clamps: list[str] = []
        hips = output_positions["hips"]
        for side, name in (("left", "leftFoot"), ("right", "rightFoot")):
            limited, clamped = _limit_distance(
                output_positions[name],
                hips,
                self._maximum_leg_reach[side],
            )
            output_positions[name] = limited
            if clamped:
                leg_clamps.append(name)

        body = RetargetedLowerBody(
            hips=RetargetedBodyTracker(
                position=output_positions["hips"],
                rotation=output_rotations["hips"],
            ),
            left_foot=RetargetedBodyTracker(
                position=output_positions["leftFoot"],
                rotation=output_rotations["leftFoot"],
            ),
            right_foot=RetargetedBodyTracker(
                position=output_positions["rightFoot"],
                rotation=output_rotations["rightFoot"],
            ),
        )
        self._previous_positions = {
            "hips": body.hips.position.copy(),
            "leftFoot": body.left_foot.position.copy(),
            "rightFoot": body.right_foot.position.copy(),
        }
        self.monitor.record(
            body,
            workspace_clamps=tuple(workspace_clamps),
            floor_clamps=tuple(floor_clamps),
            leg_clamps=tuple(leg_clamps),
            speed_clamps=tuple(speed_clamps),
        )
        return body

    def _tracked_joints(self) -> tuple[tuple[str, int], ...]:
        return (
            ("hips", self.source_joints.hips),
            ("leftFoot", self.source_joints.left_foot),
            ("rightFoot", self.source_joints.right_foot),
        )

    def _validate_pose(self, positions: np.ndarray, rotations: np.ndarray) -> None:
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError(
                f"positions must have shape (joints, 3), got {positions.shape}"
            )
        if rotations.shape != (positions.shape[0], 3, 3):
            raise ValueError(
                f"rotations must have shape ({positions.shape[0]}, 3, 3), got {rotations.shape}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
            raise ValueError("source pose must contain only finite values")
        max_index = max(
            self.source_joints.hips,
            self.source_joints.left_foot,
            self.source_joints.right_foot,
        )
        if max_index >= positions.shape[0]:
            raise IndexError(
                f"source lower-body joint index {max_index} outside "
                f"0..{positions.shape[0] - 1}"
            )

    def _initialize(self, positions: np.ndarray, rotations: np.ndarray) -> None:
        for name, index in self._tracked_joints():
            self._source_reference_positions[name] = positions[index].copy()
            self._source_reference_rotations[name] = rotations[index].copy()
        self.monitor.initialized()

    def _derive_maximum_leg_reach(self) -> dict[str, float]:
        avatar_scale = self.avatar_profile.scale_for_hmd_height(self.hmd_base[1])
        bone_chain_length = self.avatar_profile.leg_length_m * avatar_scale
        hips = self._target_reference_positions["hips"]
        result: dict[str, float] = {}
        for side, foot_name in (("left", "leftFoot"), ("right", "rightFoot")):
            reference_reach = float(
                np.linalg.norm(self._target_reference_positions[foot_name] - hips)
            )
            result[side] = max(reference_reach, bone_chain_length) * (
                self.limits.max_leg_extension_fraction
            )
        return result


def _limit_distance(
    value: np.ndarray,
    origin: np.ndarray,
    maximum_distance: float,
) -> tuple[np.ndarray, bool]:
    delta = value - origin
    distance = float(np.linalg.norm(delta))
    if distance <= maximum_distance or distance <= 1e-12:
        return value, False
    return origin + delta * (maximum_distance / distance), True


def _position_tuple(position: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in position)


def _quaternion_xyzw_to_matrix(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("tracker reference quaternion must have non-zero length")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
