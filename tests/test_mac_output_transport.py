from __future__ import annotations

from io import BytesIO
import threading
import time
import unittest
import wave

from vrc_ardy_agent.action_contracts import (
    ControlResource,
    ExecutionCommand,
    ResourceLease,
    SayAction,
)
from vrc_ardy_agent.device_payloads import decode_pose_payload
from vrc_ardy_agent.mac_output_transport import (
    MacOutputTransport,
    OutputTransportError,
)
from vrc_ardy_agent.stream_protocol import (
    AckMessage,
    AuthorizeMessage,
    OutputCompletedMessage,
    OutputDataMessage,
)
from tests.test_device_payloads import _frame


class _RecordingSender:
    def __init__(self) -> None:
        self.messages = []
        self.condition = threading.Condition()

    def __call__(self, message) -> None:
        with self.condition:
            self.messages.append(message)
            self.condition.notify_all()

    def wait_for(self, count: int, timeout: float = 1.0) -> bool:
        with self.condition:
            return self.condition.wait_for(
                lambda: len(self.messages) >= count,
                timeout=timeout,
            )


def _wav(duration_seconds: float = 0.05) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * int(16000 * duration_seconds))
    return output.getvalue()


def _speech_command() -> ExecutionCommand:
    return ExecutionCommand(
        action_id="speech-a",
        turn_id="turn-a",
        action=SayAction(text="안녕."),
        resource=ControlResource.VOICE_OUTPUT,
        lease_token=4,
    )


class MacOutputTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sender = _RecordingSender()
        self.disconnections = []
        ids = (f"message-{index}" for index in range(20))
        self.transport = MacOutputTransport(
            control_timeout_seconds=1.0,
            message_id_factory=ids.__next__,
            disconnected=lambda reason: self.disconnections.append(reason),
        )
        self.transport.open_session("session-a", self.sender)

    def test_authorize_waits_for_the_device_ack(self) -> None:
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                self.transport.authorize(
                    ResourceLease(
                        resource=ControlResource.FULL_BODY_POSE,
                        token=7,
                        action_id="motion-a",
                    )
                )
            )
        )
        thread.start()
        self.assertTrue(self.sender.wait_for(1))

        sent = self.sender.messages[0]
        self.assertEqual(
            sent,
            AuthorizeMessage(
                message_id="message-0",
                session_id="session-a",
                lease=ResourceLease(
                    resource=ControlResource.FULL_BODY_POSE,
                    token=7,
                    action_id="motion-a",
                ),
            ),
        )
        self.transport.receive(
            AckMessage(message_id="message-0", accepted=True, reason=None)
        )
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])

    def test_pose_output_carries_the_current_session_and_fencing_values(self) -> None:
        lease = ResourceLease(
            action_id="motion-a",
            resource=ControlResource.FULL_BODY_POSE,
            token=7,
        )

        self.transport.send_pose(
            lease,
            _frame(),
            sequence=12,
            ttl_ms=250,
        )

        sent = self.sender.messages[-1]
        self.assertIsInstance(sent, OutputDataMessage)
        self.assertEqual(sent.envelope.session_id, "session-a")
        self.assertEqual(sent.envelope.action_id, "motion-a")
        self.assertEqual(sent.envelope.lease_token, 7)
        self.assertEqual(sent.envelope.sequence, 12)
        self.assertEqual(decode_pose_payload(sent.envelope.payload), _frame())

    def test_speech_waits_for_authoritative_playback_completion(self) -> None:
        outcome = []
        thread = threading.Thread(
            target=lambda: outcome.append(
                self.transport.play_wav(
                    _speech_command(),
                    _wav(),
                    threading.Event(),
                )
            )
        )
        thread.start()
        self.assertTrue(self.sender.wait_for(1))
        self.assertTrue(thread.is_alive())

        sent = self.sender.messages[0]
        self.assertIsInstance(sent, OutputDataMessage)
        self.transport.receive(
            OutputCompletedMessage(
                session_id="session-a",
                action_id="speech-a",
                resource=ControlResource.VOICE_OUTPUT,
                lease_token=4,
                state="completed",
                error=None,
            )
        )
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, [None])

    def test_disconnect_wakes_waiters_and_reports_the_reason(self) -> None:
        errors = []
        thread = threading.Thread(
            target=lambda: self._capture_authorize_error(errors)
        )
        thread.start()
        self.assertTrue(self.sender.wait_for(1))

        self.transport.close_session("session-a", reason="socket closed")
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("socket closed", str(errors[0]))
        self.assertEqual(self.disconnections, ["socket closed"])
        self.assertFalse(self.transport.snapshot().connected)

    def _capture_authorize_error(self, errors: list[BaseException]) -> None:
        try:
            self.transport.authorize(
                ResourceLease(
                    resource=ControlResource.VOICE_OUTPUT,
                    token=1,
                    action_id="speech-a",
                )
            )
        except BaseException as exc:
            errors.append(exc)


if __name__ == "__main__":
    unittest.main()
