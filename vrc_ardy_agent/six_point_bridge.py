from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import time
from typing import Iterable, Iterator

import numpy as np

from .locomotion import compute_smoothed_heading_angles, iter_locomotion_frames
from .opentrack_bridge import OpenTrackFrame, encode_opentrack_packet, rotation_matrix_to_opentrack_ypr
from .vmt_bridge import VmtFrame, encode_vmt_room_unity, matrix_to_quaternion_xyzw
from .vrchat_osc import encode_vrchat_axis


@dataclass(frozen=True)
class SixPointFrame:
    head: OpenTrackFrame
    left: VmtFrame
    right: VmtFrame
    hips: VmtFrame
    left_foot: VmtFrame
    right_foot: VmtFrame
    fps: float
    scale: float
    locomotion_x: float
    locomotion_y: float
    locomotion_turn: float


def _yaw_matrix(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _validate_motion_arrays(
    positions: np.ndarray,
    rotations: np.ndarray,
    indices: tuple[int, ...],
    fps: float,
) -> None:
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"unexpected posed_joints shape: {positions.shape}")
    if rotations.shape[:2] != positions.shape[:2] or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"unexpected global_rot_mats shape: {rotations.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    for index in indices:
        if not 0 <= index < positions.shape[1]:
            raise IndexError(f"joint index {index} outside 0..{positions.shape[1] - 1}")


def _resolve_scale(
    positions: np.ndarray,
    *,
    head_index: int,
    left_toe_index: int,
    right_toe_index: int,
    hmd_base: tuple[float, float, float],
    scale: float | None,
) -> float:
    if scale is not None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        return float(scale)

    hmd_height = float(hmd_base[1])
    if hmd_height <= 0:
        raise ValueError("automatic scale requires a positive HMD base height")

    floor_y = float((positions[0, left_toe_index, 1] + positions[0, right_toe_index, 1]) * 0.5)
    source_height = float(positions[0, head_index, 1] - floor_y)
    if source_height <= 0:
        raise ValueError(f"invalid source head-to-floor height: {source_height}")
    return hmd_height / source_height


def iter_six_point_frames(
    npz_path: str | Path,
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
    remove_horizontal_root_motion: bool = False,
    locomotion: bool = False,
    full_stick_speed_mps: float = 1.0,
    locomotion_deadzone_mps: float = 0.08,
    full_turn_speed_dps: float = 180.0,
    turn_deadzone_dps: float = 12.0,
    heading_smoothing_seconds: float = 0.5,
) -> Iterator[SixPointFrame]:
    """Yield synchronized ARDY head, hands, hips and feet in one SteamVR frame.

    Every VMT device shares the first ARDY head position as its room-space
    origin. When ``scale`` is omitted, the source head-to-floor height is
    scaled to the configured VRto3D HMD height. When horizontal root motion is
    removed, generated X/Z translation is reserved for the locomotion input
    layer instead of being applied a second time through tracker positions.
    """
    path = Path(npz_path)
    if len(hmd_base) != 3:
        raise ValueError("hmd_base must contain exactly three values")

    with np.load(path, allow_pickle=False) as data:
        if "posed_joints" not in data or "global_rot_mats" not in data:
            raise ValueError("NPZ must contain posed_joints and global_rot_mats")

        positions = np.asarray(data["posed_joints"], dtype=np.float64)
        rotations = np.asarray(data["global_rot_mats"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data else 20.0
        indices = (
            head_index,
            left_index,
            right_index,
            hips_index,
            left_foot_index,
            right_foot_index,
            left_toe_index,
            right_toe_index,
        )
        _validate_motion_arrays(positions, rotations, indices, fps)

        resolved_scale = _resolve_scale(
            positions,
            head_index=head_index,
            left_toe_index=left_toe_index,
            right_toe_index=right_toe_index,
            hmd_base=hmd_base,
            scale=scale,
        )
        base_head_position = positions[0, head_index].copy()
        base_head_rotation = rotations[0, head_index].copy()
        hmd_base_vec = np.asarray(hmd_base, dtype=np.float64)

        locomotion_frames = None
        if locomotion:
            locomotion_frames = list(
                iter_locomotion_frames(
                    path,
                    full_stick_speed_mps=full_stick_speed_mps,
                    deadzone_mps=locomotion_deadzone_mps,
                    full_turn_speed_dps=full_turn_speed_dps,
                    turn_deadzone_dps=turn_deadzone_dps,
                    heading_smoothing_seconds=heading_smoothing_seconds,
                )
            )
            if len(locomotion_frames) != positions.shape[0]:
                raise ValueError(
                    f"locomotion frame count {len(locomotion_frames)} does not match pose frame count {positions.shape[0]}"
                )
            remove_horizontal_root_motion = True

        heading_delta = np.zeros(positions.shape[0], dtype=np.float64)
        if locomotion:
            headings = np.asarray(data["global_root_heading"], dtype=np.float64)
            heading_angles = compute_smoothed_heading_angles(
                headings,
                fps=fps,
                smoothing_seconds=heading_smoothing_seconds,
            )
            heading_delta = heading_angles - heading_angles[0]

        horizontal_root_delta = np.zeros((positions.shape[0], 3), dtype=np.float64)
        root_positions = None
        if remove_horizontal_root_motion:
            root_key = "smooth_root_pos" if "smooth_root_pos" in data else "root_positions"
            if root_key not in data:
                raise ValueError(
                    "remove_horizontal_root_motion requires root_positions or smooth_root_pos"
                )
            root_positions = np.asarray(data[root_key], dtype=np.float64)
            if root_positions.shape != (positions.shape[0], 3):
                raise ValueError(f"unexpected {root_key} shape: {root_positions.shape}")
            horizontal_root_delta = root_positions - root_positions[0]
            horizontal_root_delta[:, 1] = 0.0

        def stabilized_position(frame_idx: int, joint_index: int) -> np.ndarray:
            position = positions[frame_idx, joint_index].copy()
            if locomotion:
                if root_positions is None:
                    raise RuntimeError("locomotion requires root positions")
                root = root_positions[frame_idx]
                position = root + _yaw_matrix(-heading_delta[frame_idx]) @ (position - root)
            return position - horizontal_root_delta[frame_idx]

        def stabilized_rotation(frame_idx: int, joint_index: int) -> np.ndarray:
            rotation = rotations[frame_idx, joint_index]
            if locomotion:
                rotation = _yaw_matrix(-heading_delta[frame_idx]) @ rotation
            return rotation

        def vmt_frame(frame_idx: int, joint_index: int) -> VmtFrame:
            position = stabilized_position(frame_idx, joint_index)
            room_position = hmd_base_vec + (position - base_head_position) * resolved_scale
            room_rotation = stabilized_rotation(frame_idx, joint_index) @ base_head_rotation.T
            return VmtFrame(
                position=tuple(float(v) for v in room_position),
                quaternion_xyzw=matrix_to_quaternion_xyzw(room_rotation),
                fps=fps,
            )

        for frame_idx in range(positions.shape[0]):
            stabilized_head = stabilized_position(frame_idx, head_index)
            head_delta_position = (stabilized_head - base_head_position) * resolved_scale
            head_delta_rotation = stabilized_rotation(frame_idx, head_index) @ base_head_rotation.T
            head = OpenTrackFrame(
                xyz_cm=(
                    -100.0 * float(head_delta_position[0]),
                    -100.0 * float(head_delta_position[1]),
                    100.0 * float(head_delta_position[2]),
                ),
                ypr_deg=rotation_matrix_to_opentrack_ypr(head_delta_rotation),
                fps=fps,
            )

            yield SixPointFrame(
                head=head,
                left=vmt_frame(frame_idx, left_index),
                right=vmt_frame(frame_idx, right_index),
                hips=vmt_frame(frame_idx, hips_index),
                left_foot=vmt_frame(frame_idx, left_foot_index),
                right_foot=vmt_frame(frame_idx, right_foot_index),
                fps=fps,
                scale=resolved_scale,
                locomotion_x=(locomotion_frames[frame_idx].x if locomotion_frames is not None else 0.0),
                locomotion_y=(locomotion_frames[frame_idx].y if locomotion_frames is not None else 0.0),
                locomotion_turn=(locomotion_frames[frame_idx].turn if locomotion_frames is not None else 0.0),
            )


def replay_six_point_frames(
    frames: Iterable[SixPointFrame],
    *,
    host: str,
    opentrack_port: int = 4242,
    vmt_port: int = 39570,
    left_tracker_index: int = 1,
    right_tracker_index: int = 2,
    hips_tracker_index: int = 3,
    left_foot_tracker_index: int = 4,
    right_foot_tracker_index: int = 5,
    left_enable: int = 5,
    right_enable: int = 6,
    body_enable: int = 7,
    send_locomotion: bool = False,
    vrchat_port: int = 9000,
    dry_run: bool = False,
    park_head_on_exit: bool = False,
) -> int:
    """Replay synchronized HMD, hands, hips and feet in real time."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.perf_counter()
    sent = 0

    try:
        for frame in frames:
            packets = [
                (encode_opentrack_packet(*frame.head.xyz_cm, *frame.head.ypr_deg), opentrack_port),
                (
                    encode_vmt_room_unity(
                        index=left_tracker_index,
                        enable=left_enable,
                        timeoffset=0.0,
                        position=frame.left.position,
                        quaternion_xyzw=frame.left.quaternion_xyzw,
                    ),
                    vmt_port,
                ),
                (
                    encode_vmt_room_unity(
                        index=right_tracker_index,
                        enable=right_enable,
                        timeoffset=0.0,
                        position=frame.right.position,
                        quaternion_xyzw=frame.right.quaternion_xyzw,
                    ),
                    vmt_port,
                ),
                (
                    encode_vmt_room_unity(
                        index=hips_tracker_index,
                        enable=body_enable,
                        timeoffset=0.0,
                        position=frame.hips.position,
                        quaternion_xyzw=frame.hips.quaternion_xyzw,
                    ),
                    vmt_port,
                ),
                (
                    encode_vmt_room_unity(
                        index=left_foot_tracker_index,
                        enable=body_enable,
                        timeoffset=0.0,
                        position=frame.left_foot.position,
                        quaternion_xyzw=frame.left_foot.quaternion_xyzw,
                    ),
                    vmt_port,
                ),
                (
                    encode_vmt_room_unity(
                        index=right_foot_tracker_index,
                        enable=body_enable,
                        timeoffset=0.0,
                        position=frame.right_foot.position,
                        quaternion_xyzw=frame.right_foot.quaternion_xyzw,
                    ),
                    vmt_port,
                ),
            ]
            if send_locomotion:
                packets.append((encode_vrchat_axis("Horizontal", frame.locomotion_x), vrchat_port))
                packets.append((encode_vrchat_axis("Vertical", frame.locomotion_y), vrchat_port))
                packets.append((encode_vrchat_axis("LookHorizontal", frame.locomotion_turn), vrchat_port))

            if not dry_run:
                for packet, port in packets:
                    sock.sendto(packet, (host, port))
            sent += 1

            target = start + sent / frame.fps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    finally:
        if not dry_run and send_locomotion:
            sock.sendto(encode_vrchat_axis("Horizontal", 0.0), (host, vrchat_port))
            sock.sendto(encode_vrchat_axis("Vertical", 0.0), (host, vrchat_port))
            sock.sendto(encode_vrchat_axis("LookHorizontal", 0.0), (host, vrchat_port))
        if park_head_on_exit and not dry_run:
            sock.sendto(encode_opentrack_packet(0, 0, 0, 0, 0, 0), (host, opentrack_port))
        sock.close()

    return sent