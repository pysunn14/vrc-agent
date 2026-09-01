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
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    def read(self, _frame):
        if self.error is not None:
            raise self.error
        return [_region("TargetUser 28", (100, 20, 280, 50))]


class AsyncNameplateTrackerTests(unittest.TestCase):
    def test_worker_publishes_match_without_blocking_submit_and_expires_stale_result(self):
        tracker = AsyncNameplateTracker(
            reader=_Reader(),
            matcher=NameplateMatcher("TargetUser 28"),
            scan_interval_seconds=0.01,
            max_result_age_seconds=0.1,
        )
        tracker.start()
        try:
            self.assertTrue(tracker.submit(_Frame(), observed_monotonic=10.0))
            deadline = time.monotonic() + 1.0
            while tracker.status.scans_completed == 0 and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertIsNotNone(tracker.snapshot(now_monotonic=10.05))
            self.assertIsNone(tracker.snapshot(now_monotonic=10.11))
            self.assertEqual(tracker.status.matches_found, 1)
            self.assertIsNone(tracker.status.last_error)
        finally:
            tracker.stop()

    def test_worker_exposes_reader_failure_and_stops(self):
        tracker = AsyncNameplateTracker(
            reader=_Reader(error=RuntimeError("ocr failed")),
            matcher=NameplateMatcher("TargetUser 28"),
            scan_interval_seconds=0.01,
            max_result_age_seconds=0.1,
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
