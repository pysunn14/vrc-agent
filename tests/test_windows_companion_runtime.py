from __future__ import annotations

import unittest
from dataclasses import dataclass

from vrc_ardy_agent.windows_companion_runtime import WindowsCompanionRuntime


@dataclass(frozen=True)
class _DeviceSnapshot:
    running: bool
    streaming_started_wall_time: float | None


@dataclass(frozen=True)
class _SensorSnapshot:
    running: bool
    target: dict[str, object] | None


class _DevicePlane:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.running = False
        self.started_at = None

    def start(self) -> None:
        self.events.append("start:device_plane")
        self.running = True
        self.started_at = 100.0

    def stop(self) -> None:
        self.events.append("stop:device_plane")
        self.running = False

    def snapshot(self) -> _DeviceSnapshot:
        return _DeviceSnapshot(self.running, self.started_at)


class _SensorSupervisor:
    def __init__(self, events: list[str], *, target_started_at: float = 120.0) -> None:
        self.events = events
        self.running = False
        self.target_started_at = target_started_at

    def start(self) -> None:
        self.events.append("start:sensors")
        self.running = True

    def stop(self) -> None:
        self.events.append("stop:sensors")
        self.running = False

    def snapshot(self) -> _SensorSnapshot:
        return _SensorSnapshot(
            self.running,
            {"process_started_at": self.target_started_at} if self.running else None,
        )


class WindowsCompanionRuntimeTests(unittest.TestCase):
    def test_device_plane_starts_before_sensors_and_stops_after_them(self) -> None:
        events = []
        runtime = WindowsCompanionRuntime(
            device_plane=_DevicePlane(events),
            sensor_supervisor=_SensorSupervisor(events),
        )

        runtime.start()
        runtime.stop()

        self.assertEqual(
            events,
            [
                "start:device_plane",
                "start:sensors",
                "stop:sensors",
                "stop:device_plane",
            ],
        )
        self.assertFalse(runtime.snapshot().running)

    def test_late_registration_compares_authoritative_process_start_time(self) -> None:
        events = []
        runtime = WindowsCompanionRuntime(
            device_plane=_DevicePlane(events),
            sensor_supervisor=_SensorSupervisor(events, target_started_at=90.0),
        )
        self.addCleanup(runtime.stop)
        runtime.start()

        self.assertTrue(runtime.snapshot().late_registration)

        runtime.sensor_supervisor.target_started_at = 120.0
        self.assertFalse(runtime.snapshot().late_registration)

    def test_sensor_start_failure_closes_the_already_started_device_plane(self) -> None:
        events = []

        class FailingSensors(_SensorSupervisor):
            def start(self) -> None:
                self.events.append("start:sensors")
                raise RuntimeError("sensor start failed")

        runtime = WindowsCompanionRuntime(
            device_plane=_DevicePlane(events),
            sensor_supervisor=FailingSensors(events),
        )

        with self.assertRaisesRegex(RuntimeError, "sensor start failed"):
            runtime.start()

        self.assertEqual(
            events,
            ["start:device_plane", "start:sensors", "stop:device_plane"],
        )


if __name__ == "__main__":
    unittest.main()
