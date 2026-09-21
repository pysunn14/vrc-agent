from __future__ import annotations

import struct
import unittest

from vrc_ardy_agent.live_sink import BodyPoseTransport, SixPointUdpSink
from vrc_ardy_agent.opentrack_bridge import OpenTrackFrame
from vrc_ardy_agent.six_point_bridge import SixPointFrame, TrackerActivation
from vrc_ardy_agent.vmt_bridge import VmtFrame


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def sendto(self, packet: bytes, target: tuple[str, int]) -> None:
        self.sent.append((packet, target))

    def close(self) -> None:
        self.closed = True


def _frame(
    *,
    head_ypr: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tracker_activation: TrackerActivation | None = None,
) -> SixPointFrame:
    tracker = VmtFrame(
        position=(0.0, 1.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        fps=20.0,
    )
    return SixPointFrame(
        head=OpenTrackFrame(xyz_cm=(1.0, 2.0, 3.0), ypr_deg=head_ypr, fps=20.0),
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
        tracker_activation=tracker_activation or TrackerActivation(),
    )


class SixPointUdpSinkTests(unittest.TestCase):
    def test_direct_vrchat_body_output_disables_vmt_body_and_sends_osc_trackers(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(
            host="192.0.2.10",
            body_pose_transport=BodyPoseTransport.VRCHAT_OSC,
            osc_head_position=(0.0, 1.0, 0.25),
            socket_factory=lambda: sock,
        )

        sink.send(_frame())

        vmt_packets = [packet for packet, target in sock.sent if target[1] == 39570]
        decoded_vmt = [struct.unpack(">ii8f", packet[28:]) for packet in vmt_packets]
        self.assertEqual([values[1] for values in decoded_vmt], [5, 6, 0, 0, 0])

        osc_addresses = [
            packet[:packet.index(b"\x00")].decode("utf-8")
            for packet, target in sock.sent
            if target[1] == 9000
        ]
        self.assertEqual(
            osc_addresses,
            [
                "/tracking/trackers/1/position",
                "/tracking/trackers/1/rotation",
                "/tracking/trackers/2/position",
                "/tracking/trackers/2/rotation",
                "/tracking/trackers/3/position",
                "/tracking/trackers/3/rotation",
                "/tracking/trackers/head/position",
                "/input/Horizontal",
                "/input/Vertical",
                "/input/LookHorizontal",
            ],
        )

    def test_direct_vrchat_body_output_requires_head_alignment_position(self):
        with self.assertRaises(ValueError):
            SixPointUdpSink(
                host="192.0.2.10",
                body_pose_transport=BodyPoseTransport.VRCHAT_OSC,
                socket_factory=_FakeSocket,
            )

    def test_frame_can_disable_hands_without_disabling_body_trackers(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(host="192.0.2.10", socket_factory=lambda: sock)

        sink.send(
            _frame(
                tracker_activation=TrackerActivation(left=False, right=False),
            )
        )

        packets = [packet for packet, target in sock.sent if target[1] == 39570]
        decoded = [struct.unpack(">ii8f", packet[28:]) for packet in packets]
        self.assertEqual([values[1] for values in decoded], [0, 0, 7, 7, 7])

    def test_disabled_body_trackers_are_explicitly_sent_as_disabled(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(
            host="192.0.2.10",
            body_enable=0,
            socket_factory=lambda: sock,
        )

        sink.send(_frame())

        packets = [packet for packet, target in sock.sent if target[1] == 39570]
        decoded = [struct.unpack(">ii8f", packet[28:]) for packet in packets]
        self.assertEqual([values[1] for values in decoded], [5, 6, 0, 0, 0])

    def test_head_rotation_lock_keeps_translation_and_sends_neutral_rotation(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(
            host="192.0.2.10",
            lock_head_rotation=True,
            socket_factory=lambda: sock,
        )

        sink.send(_frame(head_ypr=(40.0, -30.0, 20.0)))

        packet = next(packet for packet, target in sock.sent if target[1] == 4242)
        self.assertEqual(struct.unpack("<6d", packet), (1.0, 2.0, 3.0, 0.0, 0.0, 0.0))

    def test_head_rotation_is_preserved_without_lock(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(host="192.0.2.10", socket_factory=lambda: sock)

        sink.send(_frame(head_ypr=(40.0, -30.0, 20.0)))

        packet = next(packet for packet, target in sock.sent if target[1] == 4242)
        self.assertEqual(
            struct.unpack("<6d", packet),
            (1.0, 2.0, 3.0, 40.0, -30.0, 20.0),
        )

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

    def test_close_can_explicitly_disable_all_pose_trackers(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(
            host="192.0.2.10",
            disable_trackers_on_close=True,
            socket_factory=lambda: sock,
        )
        sink.send(_frame())
        before = len(sock.sent)

        sink.close()

        close_packets = sock.sent[before:]
        ports = [target[1] for _, target in close_packets]
        self.assertEqual(ports.count(4242), 1)
        self.assertEqual(ports.count(39570), 5)
        self.assertEqual(ports.count(9000), 3)
        vmt_packets = [packet for packet, target in close_packets if target[1] == 39570]
        decoded = [struct.unpack(">ii8f", packet[28:]) for packet in vmt_packets]
        self.assertEqual([values[1] for values in decoded], [0, 0, 0, 0, 0])

    def test_neutralize_pose_disables_trackers_and_locomotion(self):
        sock = _FakeSocket()
        sink = SixPointUdpSink(host="192.0.2.10", socket_factory=lambda: sock)

        sink.neutralize_pose()

        ports = [target[1] for _, target in sock.sent]
        self.assertEqual(ports.count(4242), 1)
        self.assertEqual(ports.count(39570), 5)
        self.assertEqual(ports.count(9000), 3)
        vmt_packets = [packet for packet, target in sock.sent if target[1] == 39570]
        decoded = [struct.unpack(">ii8f", packet[28:]) for packet in vmt_packets]
        self.assertEqual([values[1] for values in decoded], [0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
