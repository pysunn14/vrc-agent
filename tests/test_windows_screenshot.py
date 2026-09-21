from __future__ import annotations

from types import SimpleNamespace
import unittest

from vrc_ardy_agent.windows_screenshot import (
    VrchatObserverScreenshotProvider,
    WindowsJpegScreenshotProvider,
)


class _FrameSource:
    def __init__(self, frame=object()) -> None:
        self.frame = frame
        self.calls = []

    def read_after(self, *, after_monotonic_ns: int, timeout_seconds: float):
        self.calls.append((after_monotonic_ns, timeout_seconds))
        return self.frame


class WindowsJpegScreenshotProviderTests(unittest.TestCase):
    def test_encodes_only_a_frame_captured_after_the_request(self) -> None:
        source = _FrameSource(frame="current-frame")
        encoder_calls = []

        def encode(frame, quality):
            encoder_calls.append((frame, quality))
            return b"\xff\xd8current\xff\xd9"

        provider = WindowsJpegScreenshotProvider(
            source=source,
            jpeg_quality=82,
            timeout_seconds=1.5,
            encoder=encode,
            monotonic_ns=lambda: 123,
        )

        jpeg = provider()

        self.assertEqual(jpeg, b"\xff\xd8current\xff\xd9")
        self.assertEqual(source.calls, [(123, 1.5)])
        self.assertEqual(encoder_calls, [("current-frame", 82)])
        self.assertEqual(provider.snapshot().screenshots_encoded, 1)

    def test_capture_failure_is_observable_and_propagated(self) -> None:
        class FailingSource(_FrameSource):
            def read_after(self, **_kwargs):
                raise TimeoutError("window produced no fresh frame")

        provider = WindowsJpegScreenshotProvider(
            source=FailingSource(),
            encoder=lambda _frame, _quality: b"unused",
        )

        with self.assertRaisesRegex(TimeoutError, "fresh frame"):
            provider()

        snapshot = provider.snapshot()
        self.assertEqual(snapshot.failures, 1)
        self.assertIn("fresh frame", snapshot.last_error)

    def test_invalid_encoder_output_is_rejected(self) -> None:
        provider = WindowsJpegScreenshotProvider(
            source=_FrameSource(),
            encoder=lambda _frame, _quality: b"not-a-jpeg",
        )

        with self.assertRaisesRegex(RuntimeError, "valid JPEG"):
            provider()


class VrchatObserverScreenshotProviderTests(unittest.TestCase):
    def test_resolves_captures_and_closes_the_observer_window_per_request(self) -> None:
        events = []

        class Source(_FrameSource):
            def start(self) -> None:
                events.append("start")

            def close(self) -> None:
                events.append("close")

        target = SimpleNamespace(
            window=SimpleNamespace(hwnd=42, process_id=7),
            identity=SimpleNamespace(display_name="Observer"),
        )
        provider = VrchatObserverScreenshotProvider(
            resolve_target=lambda: target,
            source_factory=lambda hwnd: (
                events.append(("source", hwnd)) or Source(frame="observer-frame")
            ),
            encoder=lambda frame, _quality: (
                events.append(("encode", frame))
                or b"\xff\xd8observer\xff\xd9"
            ),
        )

        jpeg = provider()

        self.assertEqual(jpeg, b"\xff\xd8observer\xff\xd9")
        self.assertEqual(
            events,
            [("source", 42), "start", ("encode", "observer-frame"), "close"],
        )
        snapshot = provider.snapshot()
        self.assertEqual(snapshot.captures, 1)
        self.assertEqual(snapshot.target_process_id, 7)
        self.assertEqual(snapshot.target_display_name, "Observer")

    def test_missing_observer_is_reported_without_a_stale_fallback(self) -> None:
        provider = VrchatObserverScreenshotProvider(
            resolve_target=lambda: None,
            source_factory=lambda _hwnd: self.fail("capture must not start"),
            resolution_diagnostics=lambda: {
                "window_count": 2,
                "matching_processes": 0,
            },
        )

        with self.assertRaisesRegex(RuntimeError, "not running"):
            provider()

        snapshot = provider.snapshot()
        self.assertEqual(snapshot.failures, 1)
        self.assertEqual(snapshot.resolution_diagnostics["window_count"], 2)


if __name__ == "__main__":
    unittest.main()
