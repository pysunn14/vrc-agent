from __future__ import annotations

import socket
from typing import Callable, Any

from .opentrack_bridge import encode_opentrack_packet
from .six_point_bridge import SixPointFrame
from .vmt_bridge import encode_vmt_room_unity
from .vrchat_osc import encode_vrchat_axis


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
        send_locomotion: bool = True,
        park_head_on_close: bool = False,
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
        self.send_locomotion = bool(send_locomotion)
        self.park_head_on_close = bool(park_head_on_close)
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

        packets = [
            (
                encode_opentrack_packet(*frame.head.xyz_cm, *frame.head.ypr_deg),
                self.opentrack_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.left_tracker_index,
                    enable=self.left_enable,
                    timeoffset=0.0,
                    position=frame.left.position,
                    quaternion_xyzw=frame.left.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.right_tracker_index,
                    enable=self.right_enable,
                    timeoffset=0.0,
                    position=frame.right.position,
                    quaternion_xyzw=frame.right.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.hips_tracker_index,
                    enable=self.body_enable,
                    timeoffset=0.0,
                    position=frame.hips.position,
                    quaternion_xyzw=frame.hips.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.left_foot_tracker_index,
                    enable=self.body_enable,
                    timeoffset=0.0,
                    position=frame.left_foot.position,
                    quaternion_xyzw=frame.left_foot.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
            (
                encode_vmt_room_unity(
                    index=self.right_foot_tracker_index,
                    enable=self.body_enable,
                    timeoffset=0.0,
                    position=frame.right_foot.position,
                    quaternion_xyzw=frame.right_foot.quaternion_xyzw,
                ),
                self.vmt_port,
            ),
        ]

        if self.send_locomotion:
            packets.extend(
                [
                    (encode_vrchat_axis("Horizontal", frame.locomotion_x), self.vrchat_port),
                    (encode_vrchat_axis("Vertical", frame.locomotion_y), self.vrchat_port),
                    (encode_vrchat_axis("LookHorizontal", frame.locomotion_turn), self.vrchat_port),
                ]
            )

        for packet, port in packets:
            self._sock.sendto(packet, (self.host, port))

    def neutralize_inputs(self) -> None:
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

    def close(self) -> None:
        if self._closed:
            return
        if self.dry_run or self._sock is None:
            self._closed = True
            return

        try:
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