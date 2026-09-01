from __future__ import annotations

import time
import unittest

from vrc_ardy_agent.nameplate import (
    AsyncNameplateTracker,
    NameplateMatcher,
    OcrTextRegion,
)


def _region(
    text: str,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> OcrTextRegion:
    return OcrTextRegion(text=text, bbox_xyxy=bbox, confidence=confidence)


class NameplateMatcherTests(unittest.TestCase):
    def test_matcher_combines_adjacent_ocr_regions_and_tolerates_one_wrong_character(self):
        matcher = NameplateMatcher("TargetUser 28", minimum_score=0.72)

        match = matcher.match(
            [
                _region("TargetUser", (100, 20, 210, 50)),
                _region("2B", (218, 21, 282, 51)),
                _region("Public", (400, 100, 480, 130)),
            ]
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.recognized_text, "TargetUser 2B")
        self.assertEqual(match.bbox_xyxy, (100.0, 20.0, 282.0, 51.0))
        self.assertGreater(match.match_score, 0.85)

    def test_matcher_rejects_unrelated_world_text(self):
        matcher = NameplateMatcher("TargetUser 28", minimum_score=0.72)

        match = matcher.match(
            [
                _region("LORA PROGRAM 2025", (10, 10, 240, 40)),
                _region("Public", (10, 50, 90, 80)),
            ]
        )

        self.assertIsNone(match)

    def test_matcher_combines_name_fragments_when_other_text_is_between_them(self):
        matcher = NameplateMatcher("TargetUser 28", minimum_score=0.72)

        match = matcher.match(
            [
                _region("TargetUser", (100, 20, 210, 50)),
                _region("Public", (150, 100, 230, 130)),
                _region("28", (218, 21, 282, 51)),
            ]
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.recognized_text, "TargetUser 28")  # type: ignore[union-attr]


class _Frame:
    def copy(self):
        return self


class _Reader:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        started=None,
        release=None,
    ) -> None:
        self.error = error
        self.started = started
        self.release = release

    def read(self, _frame):
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=1.0)
        if self.error is not None:
            raise self.error
        return [_region("TargetUser 28", (100, 20, 280, 50))]


class _VisualLocator:
    def __init__(self) -> None:
        self.anchors = []
        self.result = ((120.0, 30.0, 300.0, 60.0), 0.8)

    def reset(self) -> None:
        self.anchors.clear()

    def anchor(self, frame, bbox_xyxy) -> None:
        self.anchors.append((frame, bbox_xyxy))

    def locate(self, _frame):
        return self.result


class AsyncNameplateTrackerTests(unittest.TestCase):
    def test_slow_ocr_anchor_is_relocated_on_the_current_frame(self):
        locator = _VisualLocator()
        tracker = AsyncNameplateTracker(
            reader=_Reader(),
            matcher=NameplateMatcher("TargetUser 28"),
            visual_locator=locator,
            scan_interval_seconds=0.01,
            max_anchor_age_seconds=0.1,
        )
        tracker.start()
        try:
            self.assertTrue(tracker.submit(_Frame(), observed_monotonic=10.0))
            deadline = time.monotonic() + 1.0
            while tracker.status.scans_completed == 0 and time.monotonic() < deadline:
                time.sleep(0.01)

            now = time.monotonic()
            observed = tracker.locate(_Frame(), now_monotonic=now)

            self.assertIsNotNone(observed)
            assert observed is not None
            self.assertEqual(observed.match.bbox_xyxy, (120.0, 30.0, 300.0, 60.0))
            self.assertEqual(observed.observed_monotonic, now)
            self.assertEqual(len(locator.anchors), 1)
            self.assertIsNone(
                tracker.locate(_Frame(), now_monotonic=now + 0.11)
            )
            self.assertEqual(tracker.status.matches_found, 1)
            self.assertEqual(tracker.status.visual_matches_found, 1)
            self.assertIsNone(tracker.status.last_error)
        finally:
            tracker.stop()

    def test_worker_does_not_queue_more_frames_while_ocr_is_scanning(self):
        import threading

        started = threading.Event()
        release = threading.Event()
        tracker = AsyncNameplateTracker(
            reader=_Reader(started=started, release=release),
            matcher=NameplateMatcher("TargetUser 28"),
            visual_locator=_VisualLocator(),
            scan_interval_seconds=0.01,
            max_anchor_age_seconds=1.0,
        )
        tracker.start()
        try:
            self.assertTrue(tracker.submit(_Frame(), observed_monotonic=10.0))
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(tracker.submit(_Frame(), observed_monotonic=11.0))
        finally:
            release.set()
            tracker.stop()

    def test_worker_exposes_reader_failure_and_stops(self):
        tracker = AsyncNameplateTracker(
            reader=_Reader(error=RuntimeError("ocr failed")),
            matcher=NameplateMatcher("TargetUser 28"),
            visual_locator=_VisualLocator(),
            scan_interval_seconds=0.01,
            max_anchor_age_seconds=0.1,
        )
        tracker.start()
        tracker.submit(_Frame(), observed_monotonic=10.0)
        deadline = time.monotonic() + 1.0
        while tracker.status.running and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(tracker.status.running)
        self.assertEqual(tracker.status.last_error, "ocr failed")
        tracker.stop()


if __name__ == "__main__":
    unittest.main()
