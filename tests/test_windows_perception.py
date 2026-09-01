from __future__ import annotations

import unittest

from vrc_ardy_agent.windows_perception import (
    SingleTargetSelector,
    TrackedDetection,
    WindowsPerceptionRunner,
)


def _detection(
    *,
    track_id: int | None,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> TrackedDetection:
    return TrackedDetection(track_id=track_id, bbox_xyxy=bbox, confidence=confidence)


class SingleTargetSelectorTests(unittest.TestCase):
    def test_selector_locks_track_id_and_reacquires_only_after_grace_frames(self):
        selector = SingleTargetSelector(reacquire_after_missed_frames=2)
        first = _detection(track_id=7, bbox=(0, 0, 80, 100))
        other = _detection(track_id=8, bbox=(0, 0, 30, 50))

        selected = selector.select([other, first])
        missing_once = selector.select([other])
        reacquired = selector.select([other])

        self.assertEqual(selected.track_id, 7)  # type: ignore[union-attr]
        self.assertIsNone(missing_once)
        self.assertEqual(reacquired.track_id, 8)  # type: ignore[union-attr]

    def test_selector_without_track_ids_uses_largest_detection(self):
        selector = SingleTargetSelector()
        small = _detection(track_id=None, bbox=(0, 0, 20, 20), confidence=0.99)
        large = _detection(track_id=None, bbox=(0, 0, 50, 80), confidence=0.8)

        self.assertEqual(selector.select([small, large]), large)


class _Frame:
    shape = (100, 200, 3)


class _Capture:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def read(self, *, timeout_seconds: float):
        return _Frame()

    def close(self) -> None:
        self.closed = True


class _Tracker:
    def __init__(self) -> None:
        self.calls = 0

    def track(self, _frame):
        self.calls += 1
        if self.calls == 1:
            return [_detection(track_id=4, bbox=(50, 10, 150, 90))]
        return []


class _Sender:
    def __init__(self) -> None:
        self.observations = []
        self.closed = False

    def send(self, observation) -> None:
        self.observations.append(observation)

    def close(self) -> None:
        self.closed = True


class WindowsPerceptionRunnerTests(unittest.TestCase):
    def test_runner_sends_visible_then_explicit_invisible_full_state(self):
        capture = _Capture()
        sender = _Sender()
        runner = WindowsPerceptionRunner(
            capture=capture,
            tracker=_Tracker(),
            selector=SingleTargetSelector(reacquire_after_missed_frames=1),
            sender=sender,
        )

        status = runner.run(max_frames=2, heartbeat_interval_seconds=1.0)

        self.assertTrue(capture.started)
        self.assertTrue(capture.closed)
        self.assertTrue(sender.closed)
        self.assertEqual(status.frames_processed, 2)
        self.assertEqual(len(sender.observations), 2)
        self.assertTrue(sender.observations[0].visible)
        self.assertEqual(sender.observations[0].bbox, (0.25, 0.1, 0.75, 0.9))
        self.assertFalse(sender.observations[1].visible)
        self.assertIsNone(sender.observations[1].bbox)


if __name__ == "__main__":
    unittest.main()
