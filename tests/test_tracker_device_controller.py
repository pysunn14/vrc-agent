from __future__ import annotations

import threading
import time
import unittest
from dataclasses import replace

from tests.test_device_payloads import _frame
from vrc_ardy_agent.tracker_device_controller import (
    TrackerDeviceController,
    TrackerOutputMode,
)


class _PoseSink:
    def __init__(self) -> None:
        self.frames = []
        self.inputs_neutralized = 0
        self.trackers_disabled = 0
        self.closed = 0
        self.frame_arrived = threading.Event()

    def send(self, frame) -> None:
        self.frames.append(frame)
        self.frame_arrived.set()

    def neutralize_inputs(self) -> None:
        self.inputs_neutralized += 1

    def neutralize_pose(self) -> None:
        self.trackers_disabled += 1

    def close(self) -> None:
        self.closed += 1


class TrackerDeviceControllerTests(unittest.TestCase):
    def test_starts_with_safe_pose_before_accepting_action_targets(self) -> None:
        sink = _PoseSink()
        safe = _frame()
        action = replace(safe, locomotion_x=0.2)
        controller = TrackerDeviceController(
            sink=sink,
            safe_frame=safe,
            rate_hz=200.0,
        )
        self.addCleanup(controller.close)

        controller.start()
        self.assertEqual(sink.frames[0], safe)
        controller.send(action)
        self._wait_until(lambda: action in sink.frames)

        snapshot = controller.snapshot()
        self.assertTrue(snapshot.running)
        self.assertEqual(snapshot.mode, TrackerOutputMode.ACTION)
        self.assertGreaterEqual(snapshot.frames_sent, 2)

    def test_neutralize_returns_to_safe_pose_without_disabling_trackers(self) -> None:
        sink = _PoseSink()
        safe = _frame()
        action = replace(safe, locomotion_x=0.2)
        controller = TrackerDeviceController(
            sink=sink,
            safe_frame=safe,
            rate_hz=200.0,
        )
        self.addCleanup(controller.close)
        controller.start()
        controller.send(action)
        self._wait_until(lambda: action in sink.frames)
        before = len(sink.frames)

        controller.neutralize_pose()
        self._wait_until(lambda: safe in sink.frames[before:])

        self.assertEqual(controller.snapshot().mode, TrackerOutputMode.SAFE)
        self.assertEqual(sink.trackers_disabled, 0)
        self.assertEqual(sink.inputs_neutralized, 1)

    def test_close_is_the_only_path_that_disables_trackers(self) -> None:
        sink = _PoseSink()
        controller = TrackerDeviceController(
            sink=sink,
            safe_frame=_frame(),
            rate_hz=200.0,
        )
        controller.start()

        controller.close()
        controller.close()

        self.assertEqual(sink.trackers_disabled, 1)
        self.assertEqual(sink.closed, 1)
        self.assertFalse(controller.snapshot().running)

    @staticmethod
    def _wait_until(predicate, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not predicate():
            raise AssertionError("condition was not reached before timeout")


if __name__ == "__main__":
    unittest.main()
