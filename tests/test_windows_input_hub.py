from __future__ import annotations

import unittest

from vrc_ardy_agent.stream_protocol import ScreenshotRequestMessage
from vrc_ardy_agent.windows_input_hub import WindowsInputHub


class WindowsInputHubTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sent = []
        self.hub = WindowsInputHub()
        self.hub.open_session("bridge-a", self.sent.append)

    def test_screenshot_fails_explicitly_without_an_attached_sensor(self) -> None:
        self.hub.handle_screenshot_request(
            ScreenshotRequestMessage(request_id="shot-a", session_id="bridge-a")
        )

        response = self.sent[-1]
        self.assertIsNone(response.jpeg)
        self.assertIn("sensor_not_attached", response.error)

    def test_replacing_sensor_fences_stale_audio_and_uses_new_screenshot(self) -> None:
        self.hub.attach_sensor("sensor-a", lambda: b"old")
        self.hub.attach_sensor("sensor-b", lambda: b"new")

        stale = self.hub.publish_audio(
            "sensor-a",
            b"\x00\x00",
            captured_monotonic_ns=1,
        )
        current = self.hub.publish_audio(
            "sensor-b",
            b"\x01\x00",
            captured_monotonic_ns=2,
        )
        self.hub.handle_screenshot_request(
            ScreenshotRequestMessage(request_id="shot-b", session_id="bridge-a")
        )

        self.assertFalse(stale)
        self.assertTrue(current)
        self.assertEqual(self.sent[-1].jpeg, b"new")
        snapshot = self.hub.snapshot()
        self.assertEqual(snapshot.sensor_id, "sensor-b")
        self.assertEqual(snapshot.sensor_generation, 2)
        self.assertEqual(snapshot.stale_audio_drops, 1)

    def test_stale_detach_cannot_remove_the_current_sensor(self) -> None:
        self.hub.attach_sensor("sensor-a", lambda: b"old")
        self.hub.attach_sensor("sensor-b", lambda: b"new")

        self.assertFalse(self.hub.detach_sensor("sensor-a"))
        self.assertTrue(self.hub.snapshot().sensor_attached)
        self.assertTrue(self.hub.detach_sensor("sensor-b"))
        self.assertFalse(self.hub.snapshot().sensor_attached)


if __name__ == "__main__":
    unittest.main()
