from __future__ import annotations

from dataclasses import dataclass
import unittest

from vrc_ardy_agent.windows_perception import (
    TrackedDetection,
    WindowsPerceptionRunner,
)
from vrc_ardy_agent.follow_protocol import TargetSource
from vrc_ardy_agent.target_fusion import TargetFusionSelector


def _detection(
    *,
    track_id: int | None,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> TrackedDetection:
    return TrackedDetection(track_id=track_id, bbox_xyxy=bbox, confidence=confidence)


class TargetFusionBodyLockTests(unittest.TestCase):
    def test_selector_locks_track_id_and_reacquires_only_after_grace_frames(self):
        selector = TargetFusionSelector(reacquire_after_missed_frames=2)
        first = _detection(track_id=7, bbox=(0, 0, 80, 100))
        other = _detection(track_id=8, bbox=(0, 0, 30, 50))

        selected = selector.select(
            [other, first], nameplate=None, frame_shape=(100, 100, 3)
        )
        missing_once = selector.select(
            [other], nameplate=None, frame_shape=(100, 100, 3)
        )
        reacquired = selector.select(
            [other], nameplate=None, frame_shape=(100, 100, 3)
        )

        self.assertEqual(selected.source, TargetSource.BODY)  # type: ignore[union-attr]
        self.assertIsNone(missing_once)
        self.assertEqual(reacquired.source, TargetSource.BODY)  # type: ignore[union-attr]
        self.assertEqual(selector.target_track_id, 8)

    def test_selector_without_track_ids_uses_largest_detection(self):
        selector = TargetFusionSelector()
        small = _detection(track_id=None, bbox=(0, 0, 20, 20), confidence=0.99)
        large = _detection(track_id=None, bbox=(0, 0, 50, 80), confidence=0.8)

        selected = selector.select(
            [small, large],
            nameplate=None,
            frame_shape=(100, 100, 3),
        )
        self.assertAlmostEqual(selected.proximity, 0.8)  # type: ignore[union-attr]


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


@dataclass
class _NameplateStatus:
    running: bool = True
    scanning: bool = True
    frames_submitted: int = 1
    scans_completed: int = 0
    matches_found: int = 0
    visual_updates: int = 0
    visual_matches_found: int = 0
    last_visual_score: float | None = None
    last_scan_seconds: float | None = None
    last_error: str | None = None


class _InitialNameplateTracker:
    def __init__(self) -> None:
        self.status = _NameplateStatus()

    def start(self) -> None:
        pass

    def submit(self, _frame, *, observed_monotonic: float) -> bool:
        return True

    def locate(self, _frame, *, now_monotonic: float):
        return None

    def stop(self) -> None:
        self.status.running = False


class WindowsPerceptionRunnerTests(unittest.TestCase):
    def test_runner_sends_visible_then_explicit_invisible_full_state(self):
        capture = _Capture()
        sender = _Sender()
        runner = WindowsPerceptionRunner(
            capture=capture,
            tracker=_Tracker(),
            selector=TargetFusionSelector(reacquire_after_missed_frames=1),
            sender=sender,
        )

        status = runner.run(max_frames=2, heartbeat_interval_seconds=1.0)

        self.assertTrue(capture.started)
        self.assertTrue(capture.closed)
        self.assertTrue(sender.closed)
        self.assertEqual(status.frames_processed, 2)
        self.assertEqual(len(sender.observations), 2)
        self.assertTrue(sender.observations[0].visible)
        self.assertEqual(sender.observations[0].source, TargetSource.BODY)
        self.assertEqual(sender.observations[0].center_x, 0.5)
        self.assertEqual(sender.observations[0].proximity, 0.8)
        self.assertFalse(sender.observations[1].visible)
        self.assertIsNone(sender.observations[1].source)

    def test_runner_gives_initial_nameplate_scan_priority_over_person_inference(self):
        capture = _Capture()
        tracker = _Tracker()
        sender = _Sender()
        runner = WindowsPerceptionRunner(
            capture=capture,
            tracker=tracker,
            selector=TargetFusionSelector(),
            sender=sender,
            nameplate_tracker=_InitialNameplateTracker(),  # type: ignore[arg-type]
        )

        status = runner.run(max_frames=1, heartbeat_interval_seconds=1.0)

        self.assertEqual(tracker.calls, 0)
        self.assertEqual(status.person_inference_skipped, 1)
        self.assertFalse(sender.observations[0].visible)


if __name__ == "__main__":
    unittest.main()
