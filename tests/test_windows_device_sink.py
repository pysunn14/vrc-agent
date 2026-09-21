from __future__ import annotations

import unittest

from vrc_ardy_agent.action_contracts import ControlResource, OutputEnvelope
from vrc_ardy_agent.device_payloads import encode_pose_payload, encode_wav_payload
from vrc_ardy_agent.stream_protocol import OutputCompletedMessage
from vrc_ardy_agent.windows_device_sink import WindowsDeviceSink
from tests.test_device_payloads import _frame


class _PoseSink:
    def __init__(self) -> None:
        self.frames = []
        self.neutralized = 0
        self.closed = 0

    def send(self, frame) -> None:
        self.frames.append(frame)

    def neutralize_pose(self) -> None:
        self.neutralized += 1

    def close(self) -> None:
        self.closed += 1


class _WavePlayer:
    def __init__(self) -> None:
        self.plays = []
        self.stop_calls = 0

    def play(self, wav_bytes, completed) -> None:
        self.plays.append((wav_bytes, completed))

    def stop(self) -> None:
        self.stop_calls += 1


def _envelope(resource, payload, *, action_id, lease_token) -> OutputEnvelope:
    return OutputEnvelope(
        session_id="session-a",
        action_id=action_id,
        resource=resource,
        lease_token=lease_token,
        sequence=0,
        ttl_ms=1000,
        payload=payload,
    )


class WindowsDeviceSinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = _PoseSink()
        self.player = _WavePlayer()
        self.completions = []
        self.sink = WindowsDeviceSink(
            pose_sink=self.pose,
            wave_player=self.player,
            completion_sender=self.completions.append,
        )

    def test_pose_payload_is_decoded_and_sent_to_local_vrchat_ports(self) -> None:
        frame = _frame()

        self.sink.apply(
            _envelope(
                ControlResource.FULL_BODY_POSE,
                encode_pose_payload(frame),
                action_id="motion-a",
                lease_token=2,
            )
        )

        self.assertEqual(self.pose.frames, [frame])

    def test_voice_completion_keeps_the_action_identity(self) -> None:
        self.sink.apply(
            _envelope(
                ControlResource.VOICE_OUTPUT,
                encode_wav_payload(b"RIFF-wave"),
                action_id="speech-a",
                lease_token=7,
            )
        )
        self.assertEqual(self.player.plays[0][0], b"RIFF-wave")

        self.player.plays[0][1]("completed", None)

        self.assertEqual(
            self.completions,
            [
                OutputCompletedMessage(
                    session_id="session-a",
                    action_id="speech-a",
                    resource=ControlResource.VOICE_OUTPUT,
                    lease_token=7,
                    state="completed",
                    error=None,
                )
            ],
        )

    def test_neutralize_routes_to_the_owned_device(self) -> None:
        self.sink.neutralize(ControlResource.FULL_BODY_POSE)
        self.sink.neutralize(ControlResource.VOICE_OUTPUT)

        self.assertEqual(self.pose.neutralized, 1)
        self.assertEqual(self.player.stop_calls, 1)

    def test_resource_payload_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pose"):
            self.sink.apply(
                _envelope(
                    ControlResource.FULL_BODY_POSE,
                    encode_wav_payload(b"RIFF-wave"),
                    action_id="motion-a",
                    lease_token=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
