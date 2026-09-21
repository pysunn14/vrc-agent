from __future__ import annotations

import unittest

from vrc_ardy_agent.windows_sensor_session import VrchatSensorSession


class _Component:
    def __init__(
        self, name: str, events: list[str], *, fail_start: bool = False
    ) -> None:
        self.name = name
        self.events = events
        self.fail_start = fail_start
        self.running = False

    def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} failed")
        self.running = True

    def close(self) -> None:
        self.events.append(f"close:{self.name}")
        self.running = False

    def snapshot(self):
        return {"running": self.running}


class _InputHub:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.sensor_id = None

    def attach_sensor(self, sensor_id, _screenshot_provider) -> None:
        self.events.append(f"attach:{sensor_id}")
        self.sensor_id = sensor_id

    def detach_sensor(self, sensor_id) -> bool:
        self.events.append(f"detach:{sensor_id}")
        if self.sensor_id != sensor_id:
            return False
        self.sensor_id = None
        return True


class VrchatSensorSessionTests(unittest.TestCase):
    def test_sensor_stop_does_not_touch_the_device_plane(self) -> None:
        events = []
        hub = _InputHub(events)
        capture = _Component("capture", events)
        audio = _Component("audio", events)
        session = VrchatSensorSession(
            sensor_id="vrchat-1",
            capture=capture,
            screenshot_provider=lambda: b"jpeg",
            audio=audio,
            input_hub=hub,
            status_sources={"capture": capture, "audio": audio},
        )

        session.start()
        session.stop()

        self.assertEqual(
            events,
            [
                "start:capture",
                "attach:vrchat-1",
                "start:audio",
                "detach:vrchat-1",
                "close:audio",
                "close:capture",
            ],
        )
        self.assertFalse(session.snapshot().running)

    def test_audio_start_failure_detaches_sensor_and_closes_capture(self) -> None:
        events = []
        session = VrchatSensorSession(
            sensor_id="vrchat-1",
            capture=_Component("capture", events),
            screenshot_provider=lambda: b"jpeg",
            audio=_Component("audio", events, fail_start=True),
            input_hub=_InputHub(events),
        )

        with self.assertRaisesRegex(RuntimeError, "audio failed"):
            session.start()

        self.assertEqual(
            events,
            [
                "start:capture",
                "attach:vrchat-1",
                "start:audio",
                "detach:vrchat-1",
                "close:audio",
                "close:capture",
            ],
        )


if __name__ == "__main__":
    unittest.main()
