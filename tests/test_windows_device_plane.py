from __future__ import annotations

import unittest

from vrc_ardy_agent.windows_device_plane import WindowsDevicePlane


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

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")
        self.running = False

    def close(self) -> None:
        self.events.append(f"close:{self.name}")
        self.running = False

    def snapshot(self):
        return {"running": self.running}


class WindowsDevicePlaneTests(unittest.TestCase):
    def test_tracker_stream_starts_before_bridge_and_closes_only_at_service_stop(
        self,
    ) -> None:
        events = []
        tracker = _Component("tracker", events)
        bridge = _Component("bridge", events)
        devices = _Component("devices", events)
        plane = WindowsDevicePlane(
            tracker_controller=tracker,
            bridge=bridge,
            devices=devices,
            status_sources={"tracker": tracker, "bridge": bridge},
            wall_time=lambda: 100.0,
        )

        plane.start()
        plane.stop()

        self.assertEqual(
            events,
            ["start:tracker", "start:bridge", "stop:bridge", "close:devices"],
        )
        snapshot = plane.snapshot()
        self.assertFalse(snapshot.running)
        self.assertEqual(snapshot.streaming_started_wall_time, 100.0)
        self.assertIsNone(snapshot.vmt_alive)
        self.assertFalse(snapshot.registration_verified)


if __name__ == "__main__":
    unittest.main()
