from __future__ import annotations

import socket
import struct
from enum import Enum
from typing import Callable, Any

from .asset_contract import face_values
from .opentrack_bridge import encode_opentrack_packet
from .six_point_bridge import SixPointFrame
from .vmt_bridge import encode_vmt_room_unity
from .vrchat_osc import (
    _osc_string,
    encode_vrchat_axis,
    encode_vrchat_head_tracker_position,
    encode_vrchat_tracker_position,
    encode_vrchat_tracker_rotation,
)


class BodyPoseTransport(str, Enum):
    VMT = "vmt"
    VRCHAT_OSC = "vrchat-osc"


class SixPointUdpSink:
    """Persistent UDP sink for one live ARDY -> VRChat session."""

    def __init__(
        self,
        *,
        host: str,
        opentrack_port: int = 4242,
        vmt_port: int = 39570,
        vrchat_port: int = 9000,
        left_tracker_index: int = 1,
        right_tracker_index: int = 2,
        hips_tracker_index: int = 3,
        left_foot_tracker_index: int = 4,
        right_foot_tracker_index: int = 5,
        left_enable: int = 5,
        right_enable: int = 6,
        body_enable: int = 7,
        body_pose_transport: BodyPoseTransport | str = BodyPoseTransport.VMT,
        osc_head_position: tuple[float, float, float] | None = None,
        send_locomotion: bool = True,
        send_face: bool = False,
        lock_head_rotation: bool = False,
        park_head_on_close: bool = False,
        disable_trackers_on_close: bool = False,
        dry_run: bool = False,
        socket_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.host = host
        self.opentrack_port = int(opentrack_port)
        self.vmt_port = int(vmt_port)
        self.vrchat_port = int(vrchat_port)
        self.left_tracker_index = int(left_tracker_index)
        self.right_tracker_index = int(right_tracker_index)
        self.hips_tracker_index = int(hips_tracker_index)
        self.left_foot_tracker_index = int(left_foot_tracker_index)
        self.right_foot_tracker_index = int(right_foot_tracker_index)
        self.left_enable = int(left_enable)
        self.right_enable = int(right_enable)
        self.body_enable = int(body_enable)
        self.body_pose_transport = BodyPoseTransport(body_pose_transport)
        self.osc_head_position = osc_head_position
        if self.body_pose_transport is BodyPoseTransport.VRCHAT_OSC:
            if osc_head_position is None:
                raise ValueError(
                    "osc_head_position is required for VRChat OSC body tracking"
                )
            # Build once at startup so malformed alignment data fails before the
            # long-running device plane begins streaming.
            encode_vrchat_head_tracker_position(osc_head_position)
        self.send_locomotion = bool(send_locomotion)
        self.send_face = bool(send_face)
        self._face_parameters: set[str] = set()
        self.lock_head_rotation = bool(lock_head_rotation)
        self.park_head_on_close = bool(park_head_on_close)
        self.disable_trackers_on_close = bool(disable_trackers_on_close)
        self.dry_run = bool(dry_run)
        self._closed = False

        if self.dry_run:
            self._sock = None
        else:
            if socket_factory is None:
                socket_factory = lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock = socket_factory()

    def send(self, frame: SixPointFrame) -> None:
        if self._closed:
            raise RuntimeError("cannot send through a closed SixPointUdpSink")
        if self.dry_run:
            return
        if self._sock is None:
            raise RuntimeError("UDP socket is unavailable")

        head_ypr = (0.0, 0.0, 0.0) if self.lock_head_rotation else frame.head.ypr_deg
        direct_body = self.body_pose_transport is BodyPoseTransport.VRCHAT_OSC
        packets = [
            (
                encode_opentrack_packet(*frame.head.xyz_cm, *head_ypr),
                self.opentrack_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.left_tracker_index,
                    enable=(
                        self.left_enable if frame.tracker_activation.left else 0
                    ),
                    timeoffset=0.0,
                    position=frame.left.position,
                    quaternion_xyzw=frame.left.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.right_tracker_index,
                    enable=(
                        self.right_enable if frame.tracker_activation.right else 0
                    ),
                    timeoffset=0.0,
                    position=frame.right.position,
                    quaternion_xyzw=frame.right.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.hips_tracker_index,
                    enable=(
                        0
                        if direct_body
                        else self.body_enable if frame.tracker_activation.hips else 0
                    ),
                    timeoffset=0.0,
                    position=frame.hips.position,
                    quaternion_xyzw=frame.hips.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.left_foot_tracker_index,
                    enable=(
                        0
                        if direct_body
                        else self.body_enable
                        if frame.tracker_activation.left_foot
                        else 0
                    ),
                    timeoffset=0.0,
                    position=frame.left_foot.position,
                    quaternion_xyzw=frame.left_foot.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.right_foot_tracker_index,
                    enable=(
                        0
                        if direct_body
                        else self.body_enable
                        if frame.tracker_activation.right_foot
                        else 0
                    ),
                    timeoffset=0.0,
                    position=frame.right_foot.position,
                    quaternion_xyzw=frame.right_foot.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
        ]

        if direct_body:
            body_trackers = (
                (1, frame.hips, frame.tracker_activation.hips),
                (2, frame.left_foot, frame.tracker_activation.left_foot),
                (3, frame.right_foot, frame.tracker_activation.right_foot),
            )
            for tracker_index, tracker, active in body_trackers:
                if not active:
                    continue
                packets.extend(
                    [
                        (
                            encode_vrchat_tracker_position(
                                tracker_index,
                                tracker.position,
                            ),
                            self.vrchat_port,
                        ),
                        (
                            # The current lower-body runtime is deliberately
                            # position-only. Bone-space rotations are not valid
                            # tracker mount rotations, so keep them room-aligned
                            # until a calibrated rotation retargeter exists.
                            encode_vrchat_tracker_rotation(
                                tracker_index,
                                (0.0, 0.0, 0.0),
                            ),
                            self.vrchat_port,
                        ),
                    ]
                )
            if self.osc_head_position is None:
                raise RuntimeError("VRChat OSC head alignment position is unavailable")
            packets.append(
                (
                    encode_vrchat_head_tracker_position(self.osc_head_position),
                    self.vrchat_port,
                )
            )

        if self.send_locomotion:
            packets.extend(
                [
                    (encode_vrchat_axis("Horizontal", frame.locomotion_x), self.vrchat_port),
                    (encode_vrchat_axis("Vertical", frame.locomotion_y), self.vrchat_port),
                    (encode_vrchat_axis("LookHorizontal", frame.locomotion_turn), self.vrchat_port),
                ]
            )

        if self.send_face:
            values = face_values(dict(frame.face))
            # Every previously driven parameter gets an explicit neutral value
            # when a new action omits it, including completion and cancellation.
            self._face_parameters.update(values)
            packets[0:0] = [(self._face_packet(key, values.get(key, 0.)), self.vrchat_port)
                            for key in sorted(self._face_parameters)]
        for packet, port in packets:
            self._sock.sendto(packet, (self.host, port))

    @staticmethod
    def _face_packet(parameter, value):
        face_values({parameter: value})
        return _osc_string("/avatar/parameters/" + parameter) + _osc_string(",f") + struct.pack(">f", value)

    def neutralize_inputs(self) -> None:
        if self.send_face and not self._closed and not self.dry_run and self._sock is not None:
            for key in sorted(self._face_parameters):
                self._sock.sendto(self._face_packet(key, 0.), (self.host, self.vrchat_port))
        if self._closed or self.dry_run or self._sock is None or not self.send_locomotion:
            return
        self._sock.sendto(
            encode_vrchat_axis("Horizontal", 0.0),
            (self.host, self.vrchat_port),
        )
        self._sock.sendto(
            encode_vrchat_axis("Vertical", 0.0),
            (self.host, self.vrchat_port),
        )
        self._sock.sendto(
            encode_vrchat_axis("LookHorizontal", 0.0),
            (self.host, self.vrchat_port),
        )

    def neutralize_pose(self) -> None:
        if self._closed or self.dry_run or self._sock is None:
            return
        self.neutralize_inputs()
        self._sock.sendto(
            encode_opentrack_packet(0, 0, 0, 0, 0, 0),
            (self.host, self.opentrack_port),
        )
        for tracker_index in (
            self.left_tracker_index,
            self.right_tracker_index,
            self.hips_tracker_index,
            self.left_foot_tracker_index,
            self.right_foot_tracker_index,
        ):
            self._sock.sendto(
                encode_vmt_room_unity(
                    index=tracker_index,
                    enable=0,
                    timeoffset=0.0,
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                (self.host, self.vmt_port),
            )

    def close(self) -> None:
        if self._closed:
            return
        if self.dry_run or self._sock is None:
            self._closed = True
            return

        try:
            if self.disable_trackers_on_close:
                self.neutralize_pose()
            else:
                self.neutralize_inputs()
                if self.park_head_on_close:
                    self._sock.sendto(
                        encode_opentrack_packet(0, 0, 0, 0, 0, 0),
                        (self.host, self.opentrack_port),
                    )
        finally:
            self._closed = True
            self._sock.close()

    def __enter__(self) -> "SixPointUdpSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
