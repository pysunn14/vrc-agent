from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import time
from typing import Iterable, Iterator

import numpy as np

from .coordinate_space import ardy_to_unity_position, ardy_to_unity_rotation
from .opentrack_bridge import OpenTrackFrame, encode_opentrack_packet, rotation_matrix_to_opentrack_ypr
from .vmt_bridge import VmtFrame, encode_vmt_room_unity, matrix_to_quaternion_xyzw


@dataclass(frozen=True)
class ThreePointFrame:
    head: OpenTrackFrame
    left: VmtFrame
    right: VmtFrame
    fps: float


def _validate_motion_arrays(
    positions: np.ndarray,
    rotations: np.ndarray,
    indices: tuple[int, int, int],
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


def iter_three_point_frames(
    npz_path: str | Path,
    *,
    head_index: int = 6,
    left_index: int = 16,
    right_index: int = 10,
    hmd_base: tuple[float, float, float] = (0.0, 1.0, 0.0),
    scale: float = 1.0,
) -> Iterator[ThreePointFrame]:
    """Yield synchronized ARDY head + hands in one SteamVR room frame.

    VRto3D owns the actual HMD and adds OpenTrack as an offset around its
    configured base pose. VMT, however, consumes absolute room-space poses.
    Anchoring both hands to the first ARDY head position makes the two paths
    share one origin:

        HMD(t)  = hmd_base + (Head(t) - Head(0))
        Hand(t) = hmd_base + (Hand(t) - Head(0))

    This is equivalent to adding each current head-relative hand vector to the
    current HMD position, while preserving ARDY's global locomotion.
    """
    path = Path(npz_path)
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if len(hmd_base) != 3:
        raise ValueError("hmd_base must contain exactly three values")

    with np.load(path, allow_pickle=False) as data:
        if "posed_joints" not in data or "global_rot_mats" not in data:
            raise ValueError("NPZ must contain posed_joints and global_rot_mats")

        positions = np.asarray(data["posed_joints"], dtype=np.float64)
        rotations = np.asarray(data["global_rot_mats"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data else 20.0
        _validate_motion_arrays(positions, rotations, (head_index, left_index, right_index), fps)

        base_head_position = positions[0, head_index].copy()
        base_head_rotation = rotations[0, head_index].copy()
        hmd_base_vec = np.asarray(hmd_base, dtype=np.float64)

        for frame_idx in range(positions.shape[0]):
            head_delta_position = ardy_to_unity_position(
                (positions[frame_idx, head_index] - base_head_position) * scale
            )
            head_delta_rotation = ardy_to_unity_rotation(
                rotations[frame_idx, head_index] @ base_head_rotation.T
            )

            # VRto3D maps the OpenTrack packet to SteamVR as
            # {-X/100, -Y/100, Z/100}; pre-invert it here.
            head = OpenTrackFrame(
                xyz_cm=(
                    -100.0 * float(head_delta_position[0]),
                    -100.0 * float(head_delta_position[1]),
                    100.0 * float(head_delta_position[2]),
                ),
                ypr_deg=rotation_matrix_to_opentrack_ypr(head_delta_rotation),
                fps=fps,
            )

            def hand_frame(joint_index: int) -> VmtFrame:
                room_position = hmd_base_vec + ardy_to_unity_position(
                    (positions[frame_idx, joint_index] - base_head_position) * scale
                )
                # Use the same initial-head neutralization convention as the
                # HMD path so both devices start in one orientation frame.
                room_rotation = ardy_to_unity_rotation(
                    rotations[frame_idx, joint_index] @ base_head_rotation.T
                )
                return VmtFrame(
                    position=tuple(float(v) for v in room_position),
                    quaternion_xyzw=matrix_to_quaternion_xyzw(room_rotation),
                    fps=fps,
                )

            yield ThreePointFrame(
                head=head,
                left=hand_frame(left_index),
                right=hand_frame(right_index),
                fps=fps,
            )


def replay_three_point_frames(
    frames: Iterable[ThreePointFrame],
    *,
    host: str,
    opentrack_port: int = 4242,
    vmt_port: int = 39570,
    left_tracker_index: int = 1,
    right_tracker_index: int = 2,
    left_enable: int = 5,
    right_enable: int = 6,
    dry_run: bool = False,
    park_head_on_exit: bool = False,
) -> int:
    """Replay synchronized HMD + left/right controller poses in real time."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.perf_counter()
    sent = 0

    try:
        for frame in frames:
            head_packet = encode_opentrack_packet(*frame.head.xyz_cm, *frame.head.ypr_deg)
            left_packet = encode_vmt_room_unity(
                index=left_tracker_index,
                enable=left_enable,
                timeoffset=0.0,
                position=frame.left.position,
                quaternion_xyzw=frame.left.quaternion_xyzw,
            )
            right_packet = encode_vmt_room_unity(
                index=right_tracker_index,
                enable=right_enable,
                timeoffset=0.0,
                position=frame.right.position,
                quaternion_xyzw=frame.right.quaternion_xyzw,
            )

            if not dry_run:
                sock.sendto(head_packet, (host, opentrack_port))
                sock.sendto(left_packet, (host, vmt_port))
                sock.sendto(right_packet, (host, vmt_port))
            sent += 1

            target = start + sent / frame.fps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    finally:
        if park_head_on_exit and not dry_run:
            sock.sendto(encode_opentrack_packet(0, 0, 0, 0, 0, 0), (host, opentrack_port))
        sock.close()

    return sent
