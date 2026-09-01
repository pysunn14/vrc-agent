from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def compute_smoothed_heading_angles(
    headings: np.ndarray,
    *,
    fps: float,
    smoothing_seconds: float = 0.5,
) -> np.ndarray:
    headings = np.asarray(headings, dtype=np.float64)
    if headings.ndim != 2 or headings.shape[1] != 2:
        raise ValueError(f"unexpected global_root_heading shape: {headings.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if smoothing_seconds < 0:
        raise ValueError("smoothing_seconds must be non-negative")

    heading_norms = np.linalg.norm(headings, axis=1, keepdims=True)
    if np.any(heading_norms <= 1e-8):
        raise ValueError("global_root_heading contains a zero-length heading")
    normalized_heading = headings / heading_norms
    angles = np.unwrap(np.arctan2(normalized_heading[:, 1], normalized_heading[:, 0]))

    window = max(1, int(round(smoothing_seconds * fps)))
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(angles) <= 1:
        return angles

    half = window // 2
    padded = np.pad(angles, (half, half), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


@dataclass(frozen=True)
class LocomotionFrame:
    x: float
    y: float
    turn: float
    fps: float
    speed_mps: float
    turn_speed_dps: float


def iter_locomotion_frames(
    npz_path: str | Path,
    *,
    full_stick_speed_mps: float = 2.5,
    deadzone_mps: float = 0.08,
    full_turn_speed_dps: float = 180.0,
    turn_deadzone_dps: float = 12.0,
    heading_smoothing_seconds: float = 0.5,
) -> Iterator[LocomotionFrame]:
    """Convert ARDY root motion into left-stick strafe/forward values.

    ARDY stores heading as ``[cos(theta), sin(theta)]`` where theta=0 faces
    world +Z. Root velocity is projected into that heading-local frame so
    joystick +Y always means "move in the direction the generated body faces".
    """
    if full_stick_speed_mps <= 0:
        raise ValueError("full_stick_speed_mps must be positive")
    if deadzone_mps < 0:
        raise ValueError("deadzone_mps must be non-negative")
    if full_turn_speed_dps <= 0:
        raise ValueError("full_turn_speed_dps must be positive")
    if turn_deadzone_dps < 0:
        raise ValueError("turn_deadzone_dps must be non-negative")
    if heading_smoothing_seconds < 0:
        raise ValueError("heading_smoothing_seconds must be non-negative")

    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        root_key = "smooth_root_pos" if "smooth_root_pos" in data else "root_positions"
        if root_key not in data or "global_root_heading" not in data:
            raise ValueError("NPZ must contain root_positions/smooth_root_pos and global_root_heading")

        root_positions = np.asarray(data[root_key], dtype=np.float64)
        headings = np.asarray(data["global_root_heading"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data else 20.0

        if root_positions.ndim != 2 or root_positions.shape[1] != 3:
            raise ValueError(f"unexpected {root_key} shape: {root_positions.shape}")
        if headings.shape != (root_positions.shape[0], 2):
            raise ValueError(f"unexpected global_root_heading shape: {headings.shape}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        frame_count = root_positions.shape[0]
        planar_velocity = np.zeros((frame_count, 2), dtype=np.float64)
        if frame_count > 1:
            planar_velocity[:-1] = (
                root_positions[1:, [0, 2]] - root_positions[:-1, [0, 2]]
            ) * fps
            planar_velocity[-1] = planar_velocity[-2]

        heading_norms = np.linalg.norm(headings, axis=1, keepdims=True)
        if np.any(heading_norms <= 1e-8):
            raise ValueError("global_root_heading contains a zero-length heading")
        normalized_heading = headings / heading_norms
        heading_angles = compute_smoothed_heading_angles(
            headings,
            fps=fps,
            smoothing_seconds=heading_smoothing_seconds,
        )
        turn_velocity_dps = np.zeros(frame_count, dtype=np.float64)
        if frame_count > 1:
            turn_velocity_dps[:-1] = np.rad2deg(heading_angles[1:] - heading_angles[:-1]) * fps
            turn_velocity_dps[-1] = turn_velocity_dps[-2]

        for frame_idx in range(frame_count):
            vx, vz = planar_velocity[frame_idx]
            cos_theta, sin_theta = normalized_heading[frame_idx]

            # theta=0 faces +Z. The heading-local basis is therefore:
            #   right   = (+cos, -sin) in world (x,z)
            #   forward = (+sin, +cos) in world (x,z)
            strafe_mps = vx * cos_theta - vz * sin_theta
            forward_mps = vx * sin_theta + vz * cos_theta
            speed_mps = float(np.hypot(strafe_mps, forward_mps))

            if speed_mps < deadzone_mps:
                stick = np.zeros(2, dtype=np.float64)
            else:
                stick = np.array([strafe_mps, forward_mps], dtype=np.float64) / full_stick_speed_mps
                magnitude = float(np.linalg.norm(stick))
                if magnitude > 1.0:
                    stick /= magnitude

            turn_speed_dps = float(turn_velocity_dps[frame_idx])
            if abs(turn_speed_dps) < turn_deadzone_dps:
                turn = 0.0
            else:
                turn = float(np.clip(turn_speed_dps / full_turn_speed_dps, -1.0, 1.0))

            yield LocomotionFrame(
                x=float(stick[0]),
                y=float(stick[1]),
                turn=turn,
                fps=fps,
                speed_mps=speed_mps,
                turn_speed_dps=turn_speed_dps,
            )