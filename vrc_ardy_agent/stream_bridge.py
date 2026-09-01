from __future__ import annotations

from collections import deque
import math

import numpy as np

from .ardy_runtime import ArdyMotionChunk
from .opentrack_bridge import OpenTrackFrame, rotation_matrix_to_opentrack_ypr
from .six_point_bridge import SixPointFrame, _resolve_scale, _validate_motion_arrays, _yaw_matrix
from .vmt_bridge import VmtFrame, matrix_to_quaternion_xyzw


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
        head_index: int = 6,
        left_index: int = 16,
        right_index: int = 10,
        hips_index: int = 0,
        left_foot_index: int = 25,
        right_foot_index: int = 21,
        left_toe_index: int = 26,
        right_toe_index: int = 22,
        hmd_base: tuple[float, float, float] = (0.0, 1.0, 0.0),
        scale: float | None = None,
        full_stick_speed_mps: float = 2.5,
        locomotion_deadzone_mps: float = 0.08,
        full_turn_speed_dps: float = 180.0,
        turn_deadzone_dps: float = 12.0,
        heading_smoothing_seconds: float = 0.5,
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

        self.head_index = head_index
        self.left_index = left_index
        self.right_index = right_index
        self.hips_index = hips_index
        self.left_foot_index = left_foot_index
        self.right_foot_index = right_foot_index
        self.left_toe_index = left_toe_index
        self.right_toe_index = right_toe_index
        self.hmd_base = tuple(float(v) for v in hmd_base)
        self.requested_scale = scale
        self.full_stick_speed_mps = float(full_stick_speed_mps)
        self.locomotion_deadzone_mps = float(locomotion_deadzone_mps)
        self.full_turn_speed_dps = float(full_turn_speed_dps)
        self.turn_deadzone_dps = float(turn_deadzone_dps)
        self.heading_smoothing_seconds = float(heading_smoothing_seconds)

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

    def map_chunk(self, chunk: ArdyMotionChunk) -> list[SixPointFrame]:
        positions = np.asarray(chunk.posed_joints, dtype=np.float64)
        rotations = np.asarray(chunk.global_rot_mats, dtype=np.float64)
        roots = np.asarray(chunk.smooth_root_pos, dtype=np.float64)
        headings = np.asarray(chunk.global_root_heading, dtype=np.float64)
        fps = float(chunk.fps)

        indices = (
            self.head_index,
            self.left_index,
            self.right_index,
            self.hips_index,
            self.left_foot_index,
            self.right_foot_index,
            self.left_toe_index,
            self.right_toe_index,
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
            left_toe_index=self.left_toe_index,
            right_toe_index=self.right_toe_index,
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

        def stabilized_position(joint_index: int) -> np.ndarray:
            position = positions[frame_idx, joint_index]
            yaw_removed = root + remove_yaw @ (position - root)
            return yaw_removed - horizontal_root_delta

        def stabilized_rotation(joint_index: int) -> np.ndarray:
            return remove_yaw @ rotations[frame_idx, joint_index]

        hmd_base_vec = np.asarray(self.hmd_base, dtype=np.float64)

        def vmt_frame(joint_index: int) -> VmtFrame:
            position = stabilized_position(joint_index)
            room_position = hmd_base_vec + (position - self._base_head_position) * self._scale
            room_rotation = stabilized_rotation(joint_index) @ self._base_head_rotation.T
            return VmtFrame(
                position=tuple(float(v) for v in room_position),
                quaternion_xyzw=matrix_to_quaternion_xyzw(room_rotation),
                fps=fps,
            )

        head_position = stabilized_position(self.head_index)
        head_delta_position = (head_position - self._base_head_position) * self._scale
        head_rotation = stabilized_rotation(self.head_index) @ self._base_head_rotation.T
        head = OpenTrackFrame(
            xyz_cm=(
                -100.0 * float(head_delta_position[0]),
                -100.0 * float(head_delta_position[1]),
                100.0 * float(head_delta_position[2]),
            ),
            ypr_deg=rotation_matrix_to_opentrack_ypr(head_rotation),
            fps=fps,
        )

        return SixPointFrame(
            head=head,
            left=vmt_frame(self.left_index),
            right=vmt_frame(self.right_index),
            hips=vmt_frame(self.hips_index),
            left_foot=vmt_frame(self.left_foot_index),
            right_foot=vmt_frame(self.right_foot_index),
            fps=fps,
            scale=self._scale,
            locomotion_x=float(locomotion_x),
            locomotion_y=float(locomotion_y),
            locomotion_turn=float(locomotion_turn),
        )