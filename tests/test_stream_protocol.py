from __future__ import annotations

import unittest

from vrc_ardy_agent.action_contracts import (
    ControlResource,
    OutputEnvelope,
    ResourceLease,
)
from vrc_ardy_agent.stream_protocol import (
    AckMessage,
    AudioChunkMessage,
    AuthorizeMessage,
    HeartbeatMessage,
    HelloMessage,
    OutputCompletedMessage,
    OutputDataMessage,
    OutputNeutralizeMessage,
    ProtocolError,
    ScreenshotRequestMessage,
    ScreenshotResponseMessage,
    SessionOpenMessage,
    decode_message,
    encode_message,
)


class StreamProtocolTests(unittest.TestCase):
    def test_control_and_output_messages_round_trip(self) -> None:
        messages = (
            HelloMessage(client_instance_id="windows-a"),
            SessionOpenMessage(session_id="session-a"),
            AuthorizeMessage(
                message_id="message-1",
                session_id="session-a",
                lease=ResourceLease(
                    resource=ControlResource.FULL_BODY_POSE,
                    token=9,
                    action_id="motion-a",
                ),
            ),
            OutputNeutralizeMessage(
                message_id="message-2",
                session_id="session-a",
                resource=ControlResource.FULL_BODY_POSE,
                lease_token=10,
            ),
            OutputDataMessage(
                envelope=OutputEnvelope(
                    session_id="session-a",
                    action_id="motion-a",
                    resource=ControlResource.FULL_BODY_POSE,
                    lease_token=9,
                    sequence=17,
                    ttl_ms=250,
                    payload={"kind": "pose", "frame": {"x": 1.0}},
                )
            ),
            AckMessage(message_id="message-2", accepted=True, reason=None),
            OutputCompletedMessage(
                session_id="session-a",
                action_id="speech-a",
                resource=ControlResource.VOICE_OUTPUT,
                lease_token=4,
                state="completed",
                error=None,
            ),
            AudioChunkMessage(
                session_id="session-a",
                sequence=18,
                captured_monotonic_ns=123456,
                pcm=b"\x01\x00\x02\x00",
            ),
            ScreenshotRequestMessage(
                request_id="screen-1",
                session_id="session-a",
            ),
            ScreenshotResponseMessage(
                request_id="screen-1",
                session_id="session-a",
                jpeg=b"\xff\xd8image\xff\xd9",
                error=None,
            ),
            HeartbeatMessage(
                session_id="session-a",
                sender="windows",
                sequence=3,
            ),
        )

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(decode_message(encode_message(message)), message)

    def test_output_message_preserves_fencing_fields(self) -> None:
        decoded = decode_message(
            '{"type":"output.data","version":2,"session_id":"s",'
            '"action_id":"a","resource":"VOICE_OUTPUT","lease_token":3,'
            '"sequence":2,"ttl_ms":500,"payload":{"kind":"wav","data":"UklGRg=="}}'
        )

        self.assertIsInstance(decoded, OutputDataMessage)
        self.assertEqual(decoded.envelope.session_id, "s")
        self.assertEqual(decoded.envelope.lease_token, 3)
        self.assertEqual(decoded.envelope.sequence, 2)
        self.assertEqual(decoded.envelope.ttl_ms, 500)

    def test_unknown_or_extra_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "unsupported message type"):
            decode_message('{"type":"mystery","version":2}')
        with self.assertRaisesRegex(ProtocolError, "fields"):
            decode_message(
                '{"type":"hello","version":2,'
                '"client_instance_id":"windows-a","extra":true}'
            )

    def test_invalid_protocol_version_and_completion_state_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "version"):
            decode_message(
                '{"type":"hello","version":999,'
                '"client_instance_id":"windows-a"}'
            )
        with self.assertRaisesRegex(ProtocolError, "completion state"):
            decode_message(
                '{"type":"output.completed","version":2,'
                '"session_id":"s","action_id":"a",'
                '"resource":"VOICE_OUTPUT","lease_token":1,'
                '"state":"running","error":null}'
            )

    def test_audio_contract_is_fixed_to_16khz_mono_pcm(self) -> None:
        raw = encode_message(
            AudioChunkMessage(
                session_id="session-a",
                sequence=0,
                captured_monotonic_ns=1,
                pcm=b"\x00\x00" * 320,
            )
        )

        self.assertIn('"sample_rate":16000', raw)
        self.assertIn('"channels":1', raw)
        self.assertIn('"encoding":"pcm_s16le"', raw)

    def test_screenshot_response_has_exactly_one_result(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "exactly one"):
            encode_message(
                ScreenshotResponseMessage(
                    request_id="screen-1",
                    session_id="session-a",
                    jpeg=b"\xff\xd8image\xff\xd9",
                    error="capture failed",
                )
            )


if __name__ == "__main__":
    unittest.main()
