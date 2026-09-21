from __future__ import annotations

from collections import deque
from enum import Enum
import math

import numpy as np

from .ardy_runtime import ArdyMotionChunk
from .avatar_rig_profile import AvatarRigProfile
from .coordinate_space import ardy_to_unity_position, ardy_to_unity_rotation
from .humanoid_retargeting import (
    ARDY_ARM_JOINTS,
    HumanoidArmJoints,
    HumanoidRetargeter,
    RetargetingMonitor,
)
from .lower_body_retargeting import (
    ARDY_LOWER_BODY_JOINTS,
    LowerBodyJoints,
    LowerBodyRotationMode,
    LowerBodyRetargeter,
    LowerBodyRetargetingMonitor,
    LowerBodySafetyLimits,
    RetargetedBodyTracker,
)
from .opentrack_bridge import OpenTrackFrame, rotation_matrix_to_opentrack_ypr
from .six_point_bridge import (
    SixPointFrame,
    TrackerActivation,
    _resolve_scale,
    _validate_motion_arrays,
    _yaw_matrix,
)
from .tracking_rig import RigCalibration
from .vmt_bridge import VmtFrame, matrix_to_quaternion_xyzw


class RetargetingMode(str, Enum):
    FULL = "full"
    LOWER_BODY_POSITION_ONLY = "lower-body-position-only"


class SixPointStreamMapper:
    """Stateful ARDY chunk -> VRChat six-point mapper.

    Offline NPZ replay can derive velocity and heading smoothing from the whole
    clip. A live stream cannot reset those calculations every 40 frames, so
    this mapper carries root/heading state across chunk boundaries and keeps a
    single body calibration anchor for the entire session.
    """

    def __init__(
        self,
        *,
        avatar_profile: AvatarRigProfile,
        head_index: int = 6,
        arm_joints: HumanoidArmJoints = ARDY_ARM_JOINTS,
        lower_body_joints: LowerBodyJoints = ARDY_LOWER_BODY_JOINTS,
        hmd_base: tuple[float, float, float] = (0.0, 1.0, 0.0),
        scale: float | None = None,
        full_stick_speed_mps: float = 2.5,
        locomotion_deadzone_mps: float = 0.08,
        full_turn_speed_dps: float = 180.0,
        turn_deadzone_dps: float = 12.0,
        heading_smoothing_seconds: float = 0.5,
        retargeting_monitor: RetargetingMonitor | None = None,
        lower_body_monitor: LowerBodyRetargetingMonitor | None = None,
        lower_body_limits: LowerBodySafetyLimits | None = None,
        retargeting_mode: RetargetingMode = RetargetingMode.FULL,
    ) -> None:
        if len(hmd_base) != 3:
            raise ValueError("hmd_base must contain exactly three values")
        if full_stick_speed_mps <= 0:
            raise ValueError("full_stick_speed_mps must be positive")
        if locomotion_deadzone_mps < 0:
            raise ValueError("locomotion_deadzone_mps must be non-negative")
        if full_turn_speed_dps <= 0:
            raise ValueError("full_turn_speed_dps must be positive")
        if turn_deadzone_dps < 0:
            raise ValueError("turn_deadzone_dps must be non-negative")
        if heading_smoothing_seconds < 0:
            raise ValueError("heading_smoothing_seconds must be non-negative")
        if not isinstance(retargeting_mode, RetargetingMode):
            raise TypeError("retargeting_mode must be a RetargetingMode")

        self.head_index = head_index
        self.avatar_profile = avatar_profile
        self.rig_calibration = RigCalibration.from_avatar_profile(avatar_profile)
        self.arm_joints = arm_joints
        self.lower_body_joints = lower_body_joints
        self.hmd_base = tuple(float(v) for v in hmd_base)
        self.requested_scale = scale
        self.full_stick_speed_mps = float(full_stick_speed_mps)
        self.locomotion_deadzone_mps = float(locomotion_deadzone_mps)
        self.full_turn_speed_dps = float(full_turn_speed_dps)
        self.turn_deadzone_dps = float(turn_deadzone_dps)
        self.heading_smoothing_seconds = float(heading_smoothing_seconds)
        self.retargeting_monitor = retargeting_monitor or RetargetingMonitor()
        self.lower_body_monitor = lower_body_monitor or LowerBodyRetargetingMonitor()
        self.lower_body_limits = lower_body_limits or LowerBodySafetyLimits()
        self.retargeting_mode = retargeting_mode

        self._fps: float | None = None
        self._scale: float | None = None
        self._base_head_position: np.ndarray | None = None
        self._base_head_rotation: np.ndarray | None = None
        self._initial_root: np.ndarray | None = None
        self._initial_heading: float | None = None
        self._last_root: np.ndarray | None = None
        self._last_unwrapped_heading: float | None = None
        self._last_smoothed_heading: float | None = None
        self._heading_window: deque[float] | None = None
        self._retargeter: HumanoidRetargeter | None = None
        self._lower_body_retargeter: LowerBodyRetargeter | None = None

    @property
    def scale(self) -> float | None:
        return self._scale

    def reset(self) -> None:
        self._fps = None
        self._scale = None
        self._base_head_position = None
        self._base_head_rotation = None
        self._initial_root = None
        self._initial_heading = None
        self._last_root = None
        self._last_unwrapped_heading = None
        self._last_smoothed_heading = None
        self._heading_window = None
        self._retargeter = None
        self._lower_body_retargeter = None
        self.retargeting_monitor.reset(self.rig_calibration.name)
        self.lower_body_monitor.reset(
            self.avatar_profile.name,
            self._lower_body_rotation_mode(),
        )

    def map_chunk(self, chunk: ArdyMotionChunk) -> list[SixPointFrame]:
        positions = np.asarray(chunk.posed_joints, dtype=np.float64)
        rotations = np.asarray(chunk.global_rot_mats, dtype=np.float64)
        roots = np.asarray(chunk.smooth_root_pos, dtype=np.float64)
        headings = np.asarray(chunk.global_root_heading, dtype=np.float64)
        fps = float(chunk.fps)

        indices = (
            self.head_index,
            self.arm_joints.left.shoulder,
            self.arm_joints.left.elbow,
            self.arm_joints.left.wrist,
            self.arm_joints.right.shoulder,
            self.arm_joints.right.elbow,
            self.arm_joints.right.wrist,
            self.lower_body_joints.hips,
            self.lower_body_joints.left_foot,
            self.lower_body_joints.right_foot,
            self.lower_body_joints.left_toe,
            self.lower_body_joints.right_toe,
        )
        _validate_motion_arrays(positions, rotations, indices, fps)
        if roots.shape != (positions.shape[0], 3):
            raise ValueError(f"unexpected smooth_root_pos shape: {roots.shape}")
        if headings.shape != (positions.shape[0], 2):
            raise ValueError(f"unexpected global_root_heading shape: {headings.shape}")
        if positions.shape[0] == 0:
            return []

        norms = np.linalg.norm(headings, axis=1)
        if np.any(norms <= 1e-8):
            raise ValueError("global_root_heading contains a zero-length heading")
        headings = headings / norms[:, None]

        self._ensure_initialized(positions, rotations, roots, headings, fps)
        if self._fps is None or self._scale is None:
            raise RuntimeError("stream mapper failed to initialize")
        if not math.isclose(fps, self._fps, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"stream FPS changed from {self._fps} to {fps}")

        frames: list[SixPointFrame] = []
        for frame_idx in range(positions.shape[0]):
            unwrapped_heading = self._unwrap_heading(headings[frame_idx])
            smoothed_heading = self._smooth_heading(unwrapped_heading)
            frame = self._map_frame(
                positions=positions,
                rotations=rotations,
                roots=roots,
                frame_idx=frame_idx,
                heading=smoothed_heading,
                fps=fps,
            )
            frames.append(frame)
            self._last_root = roots[frame_idx].copy()
            self._last_smoothed_heading = smoothed_heading

        return frames

    def _ensure_initialized(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        roots: np.ndarray,
        headings: np.ndarray,
        fps: float,
    ) -> None:
        if self._fps is not None:
            return

        self._fps = fps
        self._scale = _resolve_scale(
            positions,
            head_index=self.head_index,
            left_toe_index=self.lower_body_joints.left_toe,
            right_toe_index=self.lower_body_joints.right_toe,
            hmd_base=self.hmd_base,
            scale=self.requested_scale,
        )
        self._base_head_position = positions[0, self.head_index].copy()
        self._base_head_rotation = rotations[0, self.head_index].copy()
        self._initial_root = roots[0].copy()

        initial_heading = math.atan2(float(headings[0, 1]), float(headings[0, 0]))
        self._last_unwrapped_heading = initial_heading
        self._initial_heading = initial_heading
        window_frames = max(1, int(round(self.heading_smoothing_seconds * fps)))
        self._heading_window = deque(maxlen=window_frames)
        self._retargeter = HumanoidRetargeter(
            calibration=self.rig_calibration,
            source_joints=self.arm_joints,
            hmd_base=self.hmd_base,
            body_scale=self._scale,
            monitor=self.retargeting_monitor,
        )
        self._lower_body_retargeter = LowerBodyRetargeter(
            avatar_profile=self.avatar_profile,
            source_joints=self.lower_body_joints,
            hmd_base=self.hmd_base,
            body_scale=self._scale,
            fps=fps,
            limits=self.lower_body_limits,
            monitor=self.lower_body_monitor,
            rotation_mode=self._lower_body_rotation_mode(),
        )

    def _lower_body_rotation_mode(self) -> LowerBodyRotationMode:
        if self.retargeting_mode is RetargetingMode.LOWER_BODY_POSITION_ONLY:
            return LowerBodyRotationMode.PROFILE_NEUTRAL
        return LowerBodyRotationMode.SOURCE_DELTA

    def _unwrap_heading(self, heading: np.ndarray) -> float:
        raw = math.atan2(float(heading[1]), float(heading[0]))
        if self._last_unwrapped_heading is None:
            unwrapped = raw
        else:
            delta = math.atan2(
                math.sin(raw - self._last_unwrapped_heading),
                math.cos(raw - self._last_unwrapped_heading),
            )
            unwrapped = self._last_unwrapped_heading + delta
        self._last_unwrapped_heading = unwrapped
        return unwrapped

    def _smooth_heading(self, heading: float) -> float:
        if self._heading_window is None:
            raise RuntimeError("heading smoothing is not initialized")
        self._heading_window.append(heading)
        return float(sum(self._heading_window) / len(self._heading_window))

    def _map_frame(
        self,
        *,
        positions: np.ndarray,
        rotations: np.ndarray,
        roots: np.ndarray,
        frame_idx: int,
        heading: float,
        fps: float,
    ) -> SixPointFrame:
        if (
            self._scale is None
            or self._base_head_position is None
            or self._base_head_rotation is None
            or self._initial_root is None
            or self._initial_heading is None
            or self._retargeter is None
            or self._lower_body_retargeter is None
        ):
            raise RuntimeError("stream mapper is not initialized")

        root = roots[frame_idx]
        heading_delta = heading - self._initial_heading
        remove_yaw = _yaw_matrix(-heading_delta)
        horizontal_root_delta = root - self._initial_root
        horizontal_root_delta = horizontal_root_delta.copy()
        horizontal_root_delta[1] = 0.0

        if self._last_root is None:
            velocity_xz = np.zeros(2, dtype=np.float64)
        else:
            velocity_xz = (root[[0, 2]] - self._last_root[[0, 2]]) * fps

        cos_theta = math.cos(heading)
        sin_theta = math.sin(heading)
        vx, vz = float(velocity_xz[0]), float(velocity_xz[1])
        strafe_mps = vx * cos_theta - vz * sin_theta
        forward_mps = vx * sin_theta + vz * cos_theta
        speed_mps = math.hypot(strafe_mps, forward_mps)
        if speed_mps < self.locomotion_deadzone_mps:
            locomotion_x = 0.0
            locomotion_y = 0.0
        else:
            locomotion_x = strafe_mps / self.full_stick_speed_mps
            locomotion_y = forward_mps / self.full_stick_speed_mps
            magnitude = math.hypot(locomotion_x, locomotion_y)
            if magnitude > 1.0:
                locomotion_x /= magnitude
                locomotion_y /= magnitude

        if self._last_smoothed_heading is None:
            turn_dps = 0.0
        else:
            turn_dps = math.degrees(heading - self._last_smoothed_heading) * fps
        if abs(turn_dps) < self.turn_deadzone_dps:
            locomotion_turn = 0.0
        else:
            locomotion_turn = max(-1.0, min(1.0, turn_dps / self.full_turn_speed_dps))

        stabilized_positions = (
            root
            + np.einsum("ij,kj->ki", remove_yaw, positions[frame_idx] - root)
            - horizontal_root_delta
        )
        stabilized_rotations = np.einsum(
            "ij,kjl->kil", remove_yaw, rotations[frame_idx]
        )

        def stabilized_position(joint_index: int) -> np.ndarray:
            return stabilized_positions[joint_index]

        def stabilized_rotation(joint_index: int) -> np.ndarray:
            return stabilized_rotations[joint_index]

        retargeted_hands = None
        if self.retargeting_mode is RetargetingMode.FULL:
            retargeted_hands = self._retargeter.retarget(
                stabilized_positions,
                stabilized_rotations,
            )
        retargeted_lower_body = self._lower_body_retargeter.retarget(
            stabilized_positions,
            stabilized_rotations,
        )

        def retargeted_hand_frame(side: str) -> VmtFrame:
            if retargeted_hands is None:
                return VmtFrame(
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    fps=fps,
                )
            hand = (
                retargeted_hands.left
                if side == "left"
                else retargeted_hands.right
            )
            return VmtFrame(
                position=tuple(float(value) for value in hand.position),
                quaternion_xyzw=matrix_to_quaternion_xyzw(hand.rotation),
                fps=fps,
            )

        def retargeted_body_frame(tracker: RetargetedBodyTracker) -> VmtFrame:
            return VmtFrame(
                position=tuple(float(value) for value in tracker.position),
                quaternion_xyzw=matrix_to_quaternion_xyzw(tracker.rotation),
                fps=fps,
            )

        if self.retargeting_mode is RetargetingMode.LOWER_BODY_POSITION_ONLY:
            head = OpenTrackFrame(
                xyz_cm=(0.0, 0.0, 0.0),
                ypr_deg=(0.0, 0.0, 0.0),
                fps=fps,
            )
            locomotion_x = 0.0
            locomotion_y = 0.0
            locomotion_turn = 0.0
            tracker_activation = TrackerActivation(left=False, right=False)
        else:
            head_position = stabilized_position(self.head_index)
            head_delta_position = ardy_to_unity_position(
                (head_position - self._base_head_position) * self._scale
            )
            head_rotation = ardy_to_unity_rotation(
                stabilized_rotation(self.head_index) @ self._base_head_rotation.T
            )
            head = OpenTrackFrame(
                xyz_cm=(
                    -100.0 * float(head_delta_position[0]),
                    -100.0 * float(head_delta_position[1]),
                    100.0 * float(head_delta_position[2]),
                ),
                ypr_deg=rotation_matrix_to_opentrack_ypr(head_rotation),
                fps=fps,
            )
            tracker_activation = TrackerActivation()

        return SixPointFrame(
            head=head,
            left=retargeted_hand_frame("left"),
            right=retargeted_hand_frame("right"),
            hips=retargeted_body_frame(retargeted_lower_body.hips),
            left_foot=retargeted_body_frame(retargeted_lower_body.left_foot),
            right_foot=retargeted_body_frame(retargeted_lower_body.right_foot),
            fps=fps,
            scale=self._scale,
            locomotion_x=float(locomotion_x),
            locomotion_y=float(locomotion_y),
            locomotion_turn=float(locomotion_turn),
            tracker_activation=tracker_activation,
        )
