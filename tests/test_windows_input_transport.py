from __future__ import annotations

import unittest

from vrc_ardy_agent.stream_protocol import (
    AudioChunkMessage,
    ScreenshotRequestMessage,
    ScreenshotResponseMessage,
)
from vrc_ardy_agent.windows_input_transport import WindowsInputTransport


class WindowsInputTransportTests(unittest.TestCase):
    def test_audio_is_not_buffered_while_disconnected_and_sequence_resets(self) -> None:
        sent = []
        transport = WindowsInputTransport(
            screenshot_provider=lambda: b"\xff\xd8image\xff\xd9"
        )

        self.assertFalse(
            transport.publish_audio(
                b"\x00\x00",
                captured_monotonic_ns=1,
            )
        )
        transport.open_session("session-a", sent.append)
        self.assertTrue(
            transport.publish_audio(
                b"\x01\x00",
                captured_monotonic_ns=2,
            )
        )
        transport.close_session("session-a")
        transport.open_session("session-b", sent.append)
        self.assertTrue(
            transport.publish_audio(
                b"\x02\x00",
                captured_monotonic_ns=3,
            )
        )

        chunks = [item for item in sent if isinstance(item, AudioChunkMessage)]
        self.assertEqual([item.session_id for item in chunks], ["session-a", "session-b"])
        self.assertEqual([item.sequence for item in chunks], [0, 0])
        self.assertEqual(transport.snapshot().audio_chunks_dropped, 1)

    def test_screenshot_request_captures_at_request_time(self) -> None:
        calls = []
        sent = []

        def capture() -> bytes:
            calls.append("capture")
            return b"\xff\xd8image\xff\xd9"

        transport = WindowsInputTransport(screenshot_provider=capture)
        transport.open_session("session-a", sent.append)

        transport.handle_screenshot_request(
            ScreenshotRequestMessage(
                request_id="screen-1",
                session_id="session-a",
            )
        )

        self.assertEqual(calls, ["capture"])
        self.assertEqual(
            sent,
            [
                ScreenshotResponseMessage(
                    request_id="screen-1",
                    session_id="session-a",
                    jpeg=b"\xff\xd8image\xff\xd9",
                    error=None,
                )
            ],
        )

    def test_capture_failure_is_reported_instead_of_using_an_old_frame(self) -> None:
        sent = []

        def capture() -> bytes:
            raise RuntimeError("window minimized")

        transport = WindowsInputTransport(screenshot_provider=capture)
        transport.open_session("session-a", sent.append)

        transport.handle_screenshot_request(
            ScreenshotRequestMessage("screen-1", "session-a")
        )

        self.assertEqual(sent[0].jpeg, None)
        self.assertIn("window minimized", sent[0].error)


if __name__ == "__main__":
    unittest.main()
