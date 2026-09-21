from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time

import numpy as np

from .coordinate_space import ardy_to_unity_position, ardy_to_unity_rotation
from .tracking_rig import RigCalibration


@dataclass(frozen=True, slots=True)
class ArmJointChain:
    shoulder: int
    elbow: int
    wrist: int

    def __post_init__(self) -> None:
        indices = (self.shoulder, self.elbow, self.wrist)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("arm joint indices must be integers")
        if any(index < 0 for index in indices):
            raise ValueError("arm joint indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("arm joint indices must be distinct")


@dataclass(frozen=True, slots=True)
class HumanoidArmJoints:
    left: ArmJointChain
    right: ArmJointChain


# ARDY CoreSkeleton27 mapping. This is source-model metadata, not target-avatar data.
ARDY_ARM_JOINTS = HumanoidArmJoints(
    left=ArmJointChain(shoulder=14, elbow=15, wrist=16),
    right=ArmJointChain(shoulder=8, elbow=9, wrist=10),
)


@dataclass(frozen=True, slots=True)
class RetargetedHand:
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True, slots=True)
class RetargetedHands:
    left: RetargetedHand
    right: RetargetedHand


@dataclass(frozen=True, slots=True)
class RetargetingSnapshot:
    calibration_name: str | None = None
    initialized: bool = False
    frames: int = 0
    left_source_reach_fraction: float | None = None
    right_source_reach_fraction: float | None = None
    left_output_reach_fraction: float | None = None
    right_output_reach_fraction: float | None = None
    left_clamp_count: int = 0
    right_clamp_count: int = 0
    left_hand_position: tuple[float, float, float] | None = None
    right_hand_position: tuple[float, float, float] | None = None
    heartbeat_monotonic: float = 0.0


class RetargetingMonitor:
    """Thread-safe status shared between a transient mapper and agentctl."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = RetargetingSnapshot(heartbeat_monotonic=time.monotonic())

    def reset(self, calibration_name: str) -> None:
        with self._lock:
            self._snapshot = RetargetingSnapshot(
                calibration_name=calibration_name,
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
        *,
        left_source_reach_fraction: float,
        right_source_reach_fraction: float,
        left_output_reach_fraction: float,
        right_output_reach_fraction: float,
        left_clamped: bool,
        right_clamped: bool,
        left_hand_position: np.ndarray,
        right_hand_position: np.ndarray,
    ) -> None:
        with self._lock:
            current = self._snapshot
            self._snapshot = replace(
                current,
                frames=current.frames + 1,
                left_source_reach_fraction=float(left_source_reach_fraction),
                right_source_reach_fraction=float(right_source_reach_fraction),
                left_output_reach_fraction=float(left_output_reach_fraction),
                right_output_reach_fraction=float(right_output_reach_fraction),
                left_clamp_count=current.left_clamp_count + int(left_clamped),
                right_clamp_count=current.right_clamp_count + int(right_clamped),
                left_hand_position=tuple(float(value) for value in left_hand_position),
                right_hand_position=tuple(float(value) for value in right_hand_position),
                heartbeat_monotonic=time.monotonic(),
            )

    def snapshot(self) -> RetargetingSnapshot:
        with self._lock:
            return self._snapshot


class HumanoidRetargeter:
    """Map source arm motion onto an avatar-independent calibrated target rig."""

    def __init__(
        self,
        *,
        calibration: RigCalibration,
        source_joints: HumanoidArmJoints,
        hmd_base: tuple[float, float, float],
        body_scale: float,
        monitor: RetargetingMonitor | None = None,
    ) -> None:
        if len(hmd_base) != 3 or any(not math.isfinite(float(v)) for v in hmd_base):
            raise ValueError("hmd_base must contain three finite values")
        if hmd_base[1] <= 0:
            raise ValueError("hmd_base Y must be positive")
        if not math.isfinite(body_scale) or body_scale <= 0:
            raise ValueError("body_scale must be finite and positive")

        self.calibration = calibration
        self.source_joints = source_joints
        self.hmd_base = tuple(float(value) for value in hmd_base)
        self.body_scale = float(body_scale)
        self.monitor = monitor or RetargetingMonitor()
        self._base_shoulders: dict[str, np.ndarray] = {}
        self._base_wrist_rotations: dict[str, np.ndarray] = {}
        self._source_arm_lengths: dict[str, float] = {}
        self._target_rest_rotations = {
            "left": _unity_euler_rotation(calibration.left_hand_rest_euler_deg),
            "right": _unity_euler_rotation(calibration.right_hand_rest_euler_deg),
        }
        self.monitor.reset(calibration.name)

    def reset(self) -> None:
        self._base_shoulders.clear()
        self._base_wrist_rotations.clear()
        self._source_arm_lengths.clear()
        self.monitor.reset(self.calibration.name)

    def retarget(self, positions: np.ndarray, rotations: np.ndarray) -> RetargetedHands:
        source_positions = np.asarray(positions, dtype=np.float64)
        source_rotations = np.asarray(rotations, dtype=np.float64)
        self._validate_pose(source_positions, source_rotations)
        if not self._source_arm_lengths:
            self._initialize(source_positions, source_rotations)

        left, left_source, left_output, left_clamped = self._retarget_side(
            "left", self.source_joints.left, source_positions, source_rotations
        )
        right, right_source, right_output, right_clamped = self._retarget_side(
            "right", self.source_joints.right, source_positions, source_rotations
        )
        self.monitor.record(
            left_source_reach_fraction=left_source,
            right_source_reach_fraction=right_source,
            left_output_reach_fraction=left_output,
            right_output_reach_fraction=right_output,
            left_clamped=left_clamped,
            right_clamped=right_clamped,
            left_hand_position=left.position,
            right_hand_position=right.position,
        )
        return RetargetedHands(left=left, right=right)

    def _validate_pose(self, positions: np.ndarray, rotations: np.ndarray) -> None:
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError(f"positions must have shape (joints, 3), got {positions.shape}")
        if rotations.shape != (positions.shape[0], 3, 3):
            raise ValueError(
                f"rotations must have shape ({positions.shape[0]}, 3, 3), got {rotations.shape}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
            raise ValueError("source pose must contain only finite values")
        max_index = max(
            self.source_joints.left.shoulder,
            self.source_joints.left.elbow,
            self.source_joints.left.wrist,
            self.source_joints.right.shoulder,
            self.source_joints.right.elbow,
            self.source_joints.right.wrist,
        )
        if max_index >= positions.shape[0]:
            raise IndexError(
                f"source arm joint index {max_index} outside 0..{positions.shape[0] - 1}"
            )

    def _initialize(self, positions: np.ndarray, rotations: np.ndarray) -> None:
        for side, chain in (
            ("left", self.source_joints.left),
            ("right", self.source_joints.right),
        ):
            upper = float(np.linalg.norm(positions[chain.elbow] - positions[chain.shoulder]))
            lower = float(np.linalg.norm(positions[chain.wrist] - positions[chain.elbow]))
            arm_length = upper + lower
            if arm_length <= 1e-8:
                raise ValueError(f"{side} source arm length must be positive")
            self._base_shoulders[side] = positions[chain.shoulder].copy()
            self._base_wrist_rotations[side] = rotations[chain.wrist].copy()
            self._source_arm_lengths[side] = arm_length
        self.monitor.initialized()

    def _retarget_side(
        self,
        side: str,
        chain: ArmJointChain,
        positions: np.ndarray,
        rotations: np.ndarray,
    ) -> tuple[RetargetedHand, float, float, bool]:
        source_ray = positions[chain.wrist] - positions[chain.shoulder]
        source_distance = float(np.linalg.norm(source_ray))
        source_fraction = source_distance / self._source_arm_lengths[side]
        output_fraction = min(source_fraction, self.calibration.max_arm_extension_fraction)
        clamped = source_fraction > self.calibration.max_arm_extension_fraction

        target_shoulder = np.asarray(
            self.calibration.shoulder_position(self.hmd_base, side=side),
            dtype=np.float64,
        )
        shoulder_delta = (positions[chain.shoulder] - self._base_shoulders[side]) * self.body_scale
        target_shoulder += ardy_to_unity_position(shoulder_delta)

        if source_distance <= 1e-8:
            target_position = target_shoulder
        else:
            target_direction = ardy_to_unity_position(source_ray / source_distance)
            target_distance = (
                self.calibration.scaled_arm_reach(self.hmd_base[1]) * output_fraction
            )
            target_position = target_shoulder + target_direction * target_distance

        source_rotation_delta = (
            rotations[chain.wrist] @ self._base_wrist_rotations[side].T
        )
        target_rotation = (
            ardy_to_unity_rotation(source_rotation_delta)
            @ self._target_rest_rotations[side]
        )
        return (
            RetargetedHand(position=target_position, rotation=target_rotation),
            source_fraction,
            output_fraction,
            clamped,
        )


def _unity_euler_rotation(euler_deg: tuple[float, float, float]) -> np.ndarray:
    x, y, z = (math.radians(float(value)) for value in euler_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rotate_x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    rotate_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rotate_z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return rotate_y @ rotate_x @ rotate_z
