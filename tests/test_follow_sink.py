from __future__ import annotations

import unittest

from vrc_ardy_agent.follow_control import FollowDecision, FollowState
from vrc_ardy_agent.follow_sink import VmtLocomotionSink, VrchatLocomotionSink
from vrc_ardy_agent.vmt_input import encode_vmt_joystick
from vrc_ardy_agent.vrchat_osc import encode_vrchat_axis, encode_vrchat_button


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

    def test_vr_turn_mode_pulses_direction_buttons_with_release_edges(self):
        sock = _Socket()
        now = [10.0]
        sink = VrchatLocomotionSink(
            host="windows",
            port=9000,
            turn_buttons=True,
            turn_pulse_interval_seconds=0.5,
            clock=lambda: now[0],
            socket_factory=lambda: sock,
        )
        decision = FollowDecision(
            state=FollowState.ALIGN,
            horizontal=0.0,
            vertical=0.0,
            look_horizontal=0.3,
            observation_age_seconds=0.01,
            target_center_error=0.3,
            target_height=0.3,
        )

        sink.send(decision)
        now[0] = 10.05
        sink.send(decision)
        now[0] = 10.5
        sink.send(decision)

        first = [packet for packet, _destination in sock.sent[0:5]]
        released = [packet for packet, _destination in sock.sent[5:10]]
        second = [packet for packet, _destination in sock.sent[10:15]]
        self.assertEqual(
            first[-2:],
            [
                encode_vrchat_button("LookLeft", False),
                encode_vrchat_button("LookRight", True),
            ],
        )
        self.assertEqual(
            released[-2:],
            [
                encode_vrchat_button("LookLeft", False),
                encode_vrchat_button("LookRight", False),
            ],
        )
        self.assertEqual(second[-1], encode_vrchat_button("LookRight", True))


class VmtLocomotionSinkTests(unittest.TestCase):
    def test_decision_drives_left_stick_and_pulses_right_stick(self):
        sock = _Socket()
        now = [10.0]
        sink = VmtLocomotionSink(
            host="windows",
            port=39570,
            left_controller_index=1,
            right_controller_index=2,
            turn_pulse_interval_seconds=0.5,
            clock=lambda: now[0],
            socket_factory=lambda: sock,
        )
        decision = FollowDecision(
            state=FollowState.FOLLOW,
            horizontal=-0.1,
            vertical=0.3,
            look_horizontal=0.2,
            observation_age_seconds=0.01,
            target_center_error=0.2,
            target_height=0.3,
        )

        sink.send(decision)
        now[0] = 10.05
        sink.send(decision)
        now[0] = 10.5
        sink.send(decision)

        packets = [packet for packet, _destination in sock.sent]
        self.assertEqual(
            packets,
            [
                encode_vmt_joystick(
                    index=1, joystick_index=0, timeoffset=0.0, x=-0.1, y=0.3
                ),
                encode_vmt_joystick(
                    index=2, joystick_index=0, timeoffset=0.0, x=1.0, y=0.0
                ),
                encode_vmt_joystick(
                    index=1, joystick_index=0, timeoffset=0.0, x=-0.1, y=0.3
                ),
                encode_vmt_joystick(
                    index=2, joystick_index=0, timeoffset=0.0, x=0.0, y=0.0
                ),
                encode_vmt_joystick(
                    index=1, joystick_index=0, timeoffset=0.0, x=-0.1, y=0.3
                ),
                encode_vmt_joystick(
                    index=2, joystick_index=0, timeoffset=0.0, x=1.0, y=0.0
                ),
            ],
        )
        self.assertTrue(all(destination == ("windows", 39570) for _, destination in sock.sent))

    def test_close_neutralizes_both_controller_sticks(self):
        sock = _Socket()
        sink = VmtLocomotionSink(
            host="windows",
            socket_factory=lambda: sock,
        )

        sink.close()
        sink.close()

        self.assertEqual(
            [packet for packet, _destination in sock.sent],
            [
                encode_vmt_joystick(
                    index=1, joystick_index=0, timeoffset=0.0, x=0.0, y=0.0
                ),
                encode_vmt_joystick(
                    index=2, joystick_index=0, timeoffset=0.0, x=0.0, y=0.0
                ),
            ],
        )
        self.assertTrue(sock.closed)


if __name__ == "__main__":
    unittest.main()
