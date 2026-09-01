from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import struct
import time
from typing import Iterable, Iterator

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class OpenTrackFrame:
    xyz_cm: tuple[float, float, float]
    ypr_deg: tuple[float, float, float]
    fps: float


def encode_opentrack_packet(
    x: float,
    y: float,
    z: float,
    yaw: float,
    pitch: float,
    roll: float,
) -> bytes:
    """Encode OpenTrack's UDP-over-network packet: 6 little-endian doubles."""
    return struct.pack("<6d", float(x), float(y), float(z), float(yaw), float(pitch), float(roll))


def rotation_matrix_to_opentrack_ypr(matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a SteamVR-space rotation matrix to the Y/P/R values VRto3D expects.

    VRto3D reconstructs its quaternion with an extrinsic Z-X-Y Euler sequence using
    (Roll, Pitch, -Yaw), so this function performs the exact inverse mapping.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got {m.shape}")

    roll_z, pitch_x, yaw_y = Rotation.from_matrix(m).as_euler("zxy", degrees=True)
    return float(-yaw_y), float(pitch_x), float(roll_z)


def iter_opentrack_frames(npz_path: str | Path, *, joint_index: int = 6) -> Iterator[OpenTrackFrame]:
    """Yield ARDY head pose as OpenTrack offsets relative to the first frame.

    VRto3D adds OpenTrack data to its configured base HMD pose. Sending relative
    motion avoids applying the avatar's absolute ~1.6 m head height twice.
    """
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        if "posed_joints" not in data or "global_rot_mats" not in data:
            raise ValueError("NPZ must contain posed_joints and global_rot_mats")

        positions = np.asarray(data["posed_joints"], dtype=np.float64)
        rotations = np.asarray(data["global_rot_mats"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data else 20.0

        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"unexpected posed_joints shape: {positions.shape}")
        if rotations.shape[:2] != positions.shape[:2] or rotations.shape[-2:] != (3, 3):
            raise ValueError(f"unexpected global_rot_mats shape: {rotations.shape}")
        if not 0 <= joint_index < positions.shape[1]:
            raise IndexError(f"joint index {joint_index} outside 0..{positions.shape[1] - 1}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        base_position = positions[0, joint_index].copy()
        base_rotation = rotations[0, joint_index].copy()

        for frame_idx in range(positions.shape[0]):
            delta_position = positions[frame_idx, joint_index] - base_position
            # ARDY matrices describe the joint frame in world space. Remove the
            # initial head orientation so VRto3D receives motion around neutral.
            delta_rotation = rotations[frame_idx, joint_index] @ base_rotation.T

            # VRto3D converts OpenTrack cm to SteamVR meters as:
            #   {-X / 100, -Y / 100, Z / 100}
            # Invert that conversion so the final SteamVR delta equals ARDY's delta.
            x_cm = -100.0 * float(delta_position[0])
            y_cm = -100.0 * float(delta_position[1])
            z_cm = 100.0 * float(delta_position[2])
            ypr = rotation_matrix_to_opentrack_ypr(delta_rotation)
            yield OpenTrackFrame(xyz_cm=(x_cm, y_cm, z_cm), ypr_deg=ypr, fps=fps)


def replay_frames(
    frames: Iterable[OpenTrackFrame],
    *,
    host: str,
    port: int = 4242,
    dry_run: bool = False,
    park_on_exit: bool = True,
) -> int:
    """Replay OpenTrack frames in real time and return the number sent."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.perf_counter()
    sent = 0

    try:
        for frame in frames:
            packet = encode_opentrack_packet(*frame.xyz_cm, *frame.ypr_deg)
            if not dry_run:
                sock.sendto(packet, (host, port))
            sent += 1

            target = start + sent / frame.fps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    finally:
        if park_on_exit and not dry_run:
            sock.sendto(encode_opentrack_packet(0, 0, 0, 0, 0, 0), (host, port))
        sock.close()

    return sent