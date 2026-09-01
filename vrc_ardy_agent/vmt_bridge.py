from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import socket
import struct
import time
from typing import Iterable, Iterator

import numpy as np


VMT_ROOM_UNITY = "/VMT/Room/Unity"


@dataclass(frozen=True)
class VmtFrame:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    fps: float


def _osc_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    padding = (-len(raw)) % 4
    return raw + (b"\x00" * padding)


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to an OSC/VMT quaternion in x,y,z,w order."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got {m.shape}")

    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("rotation matrix produced a zero-length quaternion")
    quat /= norm
    return tuple(float(v) for v in quat)


def encode_vmt_room_unity(
    *,
    index: int,
    enable: int,
    timeoffset: float,
    position: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
) -> bytes:
    """Encode VMT's recommended /VMT/Room/Unity OSC message."""
    x, y, z = position
    qx, qy, qz, qw = quaternion_xyzw
    return b"".join(
        [
            _osc_string(VMT_ROOM_UNITY),
            _osc_string(",iiffffffff"),
            struct.pack(">ii", int(index), int(enable)),
            struct.pack(">ffffffff", float(timeoffset), x, y, z, qx, qy, qz, qw),
        ]
    )


def iter_vmt_frames(npz_path: str | Path, *, joint_index: int) -> Iterator[VmtFrame]:
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        if "posed_joints" not in data or "global_rot_mats" not in data:
            raise ValueError("NPZ must contain posed_joints and global_rot_mats")

        positions = np.asarray(data["posed_joints"], dtype=np.float32)
        rotations = np.asarray(data["global_rot_mats"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data else 20.0

        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"unexpected posed_joints shape: {positions.shape}")
        if rotations.shape[:2] != positions.shape[:2] or rotations.shape[-2:] != (3, 3):
            raise ValueError(f"unexpected global_rot_mats shape: {rotations.shape}")
        if not 0 <= joint_index < positions.shape[1]:
            raise IndexError(f"joint index {joint_index} outside 0..{positions.shape[1] - 1}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        for frame_idx in range(positions.shape[0]):
            position = tuple(float(v) for v in positions[frame_idx, joint_index])
            quaternion = matrix_to_quaternion_xyzw(rotations[frame_idx, joint_index])
            yield VmtFrame(position=position, quaternion_xyzw=quaternion, fps=fps)


def replay_frames(
    frames: Iterable[VmtFrame],
    *,
    host: str,
    port: int = 39570,
    tracker_index: int = 0,
    enable: int = 1,
    scale: float = 1.0,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dry_run: bool = False,
) -> int:
    """Replay frames in real time. Returns the number of transmitted frames."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.perf_counter()
    sent = 0

    try:
        for frame in frames:
            x = frame.position[0] * scale + offset[0]
            y = frame.position[1] * scale + offset[1]
            z = frame.position[2] * scale + offset[2]
            packet = encode_vmt_room_unity(
                index=tracker_index,
                enable=enable,
                timeoffset=0.0,
                position=(x, y, z),
                quaternion_xyzw=frame.quaternion_xyzw,
            )
            if not dry_run:
                sock.sendto(packet, (host, port))
            sent += 1

            target = start + sent / frame.fps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    finally:
        sock.close()

    return sent