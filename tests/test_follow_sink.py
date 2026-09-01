from __future__ import annotations

import unittest

from vrc_ardy_agent.follow_control import FollowDecision, FollowState
from vrc_ardy_agent.follow_sink import VrchatLocomotionSink
from vrc_ardy_agent.vrchat_osc import encode_vrchat_axis


class _Socket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def sendto(self, packet: bytes, destination: tuple[str, int]) -> None:
        self.sent.append((packet, destination))

    def close(self) -> None:
        self.closed = True


class VrchatLocomotionSinkTests(unittest.TestCase):
    def test_decision_sends_all_axes_without_waiting_for_an_ardy_frame(self):
        sock = _Socket()
        sink = VrchatLocomotionSink(
            host="windows",
            port=9000,
            socket_factory=lambda: sock,
        )
        decision = FollowDecision(
            state=FollowState.ALIGN,
            horizontal=0.0,
            vertical=0.3,
            look_horizontal=-0.2,
            observation_age_seconds=0.01,
            target_center_error=-0.2,
            target_height=0.3,
        )

        sink.send(decision)

        self.assertEqual(
            sock.sent,
            [
                (encode_vrchat_axis("Horizontal", 0.0), ("windows", 9000)),
                (encode_vrchat_axis("Vertical", 0.3), ("windows", 9000)),
                (encode_vrchat_axis("LookHorizontal", -0.2), ("windows", 9000)),
            ],
        )

    def test_close_neutralizes_axes_and_is_idempotent(self):
        sock = _Socket()
        sink = VrchatLocomotionSink(
            host="windows",
            port=9000,
            socket_factory=lambda: sock,
        )

        sink.close()
        sink.close()

        self.assertEqual(len(sock.sent), 3)
        self.assertEqual(
            [packet for packet, _destination in sock.sent],
            [
                encode_vrchat_axis("Horizontal", 0.0),
                encode_vrchat_axis("Vertical", 0.0),
                encode_vrchat_axis("LookHorizontal", 0.0),
            ],
        )
        self.assertTrue(sock.closed)


if __name__ == "__main__":
    unittest.main()
