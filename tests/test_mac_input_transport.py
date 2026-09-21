from __future__ import annotations

import threading
import time
import unittest

from vrc_ardy_agent.mac_input_transport import (
    InputTransportError,
    MacInputTransport,
)
from vrc_ardy_agent.stream_protocol import (
    AudioChunkMessage,
    ScreenshotRequestMessage,
    ScreenshotResponseMessage,
)


class MacInputTransportTests(unittest.TestCase):
    def test_screenshot_request_waits_for_matching_response(self) -> None:
        sent = []
        transport = MacInputTransport(
            audio_handler=lambda _message: None,
            request_id_factory=lambda: "screen-1",
            screenshot_timeout_seconds=1.0,
        )
        transport.open_session("session-a", sent.append)
        result = []

        thread = threading.Thread(
            target=lambda: result.append(transport.request_screenshot())
        )
        thread.start()
        deadline = time.monotonic() + 1.0
        while not sent and time.monotonic() < deadline:
            time.sleep(0.001)

        self.assertEqual(
            sent,
            [ScreenshotRequestMessage("screen-1", "session-a")],
        )
        self.assertTrue(
            transport.receive(
                ScreenshotResponseMessage(
                    request_id="screen-1",
                    session_id="session-a",
                    jpeg=b"\xff\xd8image\xff\xd9",
                    error=None,
                )
            )
        )
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [b"\xff\xd8image\xff\xd9"])

    def test_audio_gaps_and_old_chunks_are_observable(self) -> None:
        received = []
        transport = MacInputTransport(audio_handler=received.append)
        transport.open_session("session-a", lambda _message: None)

        for sequence in (3, 5, 4):
            transport.receive(
                AudioChunkMessage(
                    session_id="session-a",
                    sequence=sequence,
                    captured_monotonic_ns=sequence,
                    pcm=b"\x00\x00",
                )
            )

        self.assertEqual([item.sequence for item in received], [3, 5])
        snapshot = transport.snapshot()
        self.assertEqual(snapshot.last_audio_sequence, 5)
        self.assertEqual(snapshot.missing_audio_chunks, 1)
        self.assertEqual(snapshot.stale_messages_received, 1)

    def test_disconnect_wakes_pending_screenshot_without_fallback(self) -> None:
        sent = []
        transport = MacInputTransport(
            audio_handler=lambda _message: None,
            request_id_factory=lambda: "screen-1",
            screenshot_timeout_seconds=1.0,
        )
        transport.open_session("session-a", sent.append)
        errors = []
        thread = threading.Thread(
            target=lambda: self._capture_error(transport, errors)
        )
        thread.start()
        deadline = time.monotonic() + 1.0
        while not sent and time.monotonic() < deadline:
            time.sleep(0.001)

        transport.close_session("session-a", reason="link lost")

        thread.join(timeout=1.0)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InputTransportError)
        self.assertIn("link lost", str(errors[0]))

    @staticmethod
    def _capture_error(transport, errors) -> None:
        try:
            transport.request_screenshot()
        except BaseException as exc:
            errors.append(exc)


if __name__ == "__main__":
    unittest.main()
