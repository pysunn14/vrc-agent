from __future__ import annotations

import unittest

from vrc_ardy_agent.live_sink import SixPointUdpSink
from vrc_ardy_agent.opentrack_bridge import OpenTrackFrame
from vrc_ardy_agent.six_point_bridge import SixPointFrame
from vrc_ardy_agent.vmt_bridge import VmtFrame


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def sendto(self, packet: bytes, target: tuple[str, int]) -> None:
        self.sent.append((packet, target))

    def close(self) -> None:
        self.closed = True


def _frame() -> SixPointFrame:
    tracker = VmtFrame(
        position=(0.0, 1.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        fps=20.0,
    )
    return SixPointFrame(
        head=OpenTrackFrame(xyz_cm=(0.0, 0.0, 0.0), ypr_deg=(0.0, 0.0, 0.0), fps=20.0),
        left=tracker,
        right=tracker,
        hips=tracker,
        left_foot=tracker,
        right_foot=tracker,
        fps=20.0,
        scale=1.0,
        locomotion_x=0.25,
        locomotion_y=0.75,
        locomotion_turn=-0.5,
    )


class SixPointUdpSinkTests(unittest.TestCase):
    def test_send_routes_pose_and_locomotion_to_expected_ports(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(
            host="192.0.2.10",
            socket_factory=lambda: sock,
        )

        sink.send(_frame())

        ports = [target[1] for _, target in sock.sent]
        self.assertEqual(ports.count(4242), 1)
        self.assertEqual(ports.count(39570), 5)
        self.assertEqual(ports.count(9000), 3)
        self.assertEqual(len(sock.sent), 9)

    def test_close_neutralizes_vrchat_input_and_closes_socket(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(host="192.0.2.10", socket_factory=lambda: sock)
        sink.send(_frame())
        before = len(sock.sent)

        sink.close()

        self.assertEqual(len(sock.sent) - before, 3)
        self.assertTrue(sock.closed)
        sink.close()  # idempotent
        self.assertEqual(len(sock.sent) - before, 3)


if __name__ == "__main__":
    unittest.main()