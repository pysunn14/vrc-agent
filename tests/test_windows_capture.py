from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

from vrc_ardy_agent.windows_capture import (
    LatestFrameQueue,
    WindowInfo,
    WindowsGraphicsCaptureSource,
    choose_window,
    wait_for_window,
)


class WindowSelectionTests(unittest.TestCase):
    def test_choose_window_prefers_visible_exact_title_then_largest_area(self):
        windows = [
            WindowInfo(1, "VRChat", 10, 1920, 1080, True),
            WindowInfo(2, "VRChat overlay", 11, 2560, 1440, False),
            WindowInfo(3, "VRChat", 12, 1280, 720, False),
            WindowInfo(4, "VRChat", 13, 1600, 900, False),
        ]

        selected = choose_window(windows, title="VRChat")

        self.assertEqual(selected, windows[3])

    def test_wait_for_window_retries_and_reports_heartbeat(self):
        target = WindowInfo(7, "VRChat", 20, 1920, 1080, False)
        responses = iter([[], [], [target]])
        now = 0.0
        heartbeats: list[float] = []

        def enumerate_windows(*, title_filter: str | None = None):
            self.assertEqual(title_filter, "VRChat")
            return next(responses)

        def monotonic() -> float:
            return now

        def sleep(seconds: float) -> None:
            nonlocal now
            now += seconds

        selected = wait_for_window(
            title="VRChat",
            poll_interval_seconds=0.5,
            heartbeat_interval_seconds=0.5,
            enumerate_windows=enumerate_windows,
            monotonic=monotonic,
            sleep=sleep,
            heartbeat=heartbeats.append,
        )

        self.assertEqual(selected, target)
        self.assertEqual(heartbeats, [0.5])

    def test_wait_for_window_times_out(self):
        now = 0.0

        def monotonic() -> float:
            return now

        def sleep(seconds: float) -> None:
            nonlocal now
            now += seconds

        with self.assertRaisesRegex(TimeoutError, "VRChat"):
            wait_for_window(
                title="VRChat",
                timeout_seconds=1.0,
                poll_interval_seconds=0.5,
                enumerate_windows=lambda **_kwargs: [],
                monotonic=monotonic,
                sleep=sleep,
            )


class _CopyableFrame:
    def __init__(self, value: int) -> None:
        self.value = value

    def copy(self):
        return _CopyableFrame(self.value)


class LatestFrameQueueTests(unittest.TestCase):
    def test_publish_drops_old_frame_instead_of_growing_backlog(self):
        frames = LatestFrameQueue()

        frames.publish(_CopyableFrame(1))
        frames.publish(_CopyableFrame(2))

        self.assertEqual(frames.read(timeout_seconds=0.01).value, 2)

    def test_closed_empty_queue_reports_end_of_capture(self):
        frames = LatestFrameQueue()
        frames.close()

        with self.assertRaises(EOFError):
            frames.read(timeout_seconds=0.01)


class _CaptureControl:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _RawFrame:
    def convert_to_bgr(self):
        return SimpleNamespace(frame_buffer=_CopyableFrame(42))


class _WindowsCaptureBackend:
    latest = None

    def __init__(self, **options) -> None:
        self.options = options
        self.events = {}
        self.control = _CaptureControl()
        type(self).latest = self

    def event(self, callback):
        self.events[callback.__name__] = callback
        return callback

    def start_free_threaded(self):
        self.events["on_frame_arrived"](_RawFrame(), self.control)
        return self.control


class WindowsGraphicsCaptureSourceTests(unittest.TestCase):
    def test_source_registers_callbacks_and_copies_bgr_frame(self):
        fake_module = SimpleNamespace(WindowsCapture=_WindowsCaptureBackend)
        with patch("vrc_ardy_agent.windows_capture.sys.platform", "win32"), patch.dict(
            "sys.modules", {"windows_capture": fake_module}
        ):
            source = WindowsGraphicsCaptureSource(hwnd=123, minimum_update_interval_ms=50)
            source.start()
            frame = source.read(timeout_seconds=0.01)
            source.close()

        backend = _WindowsCaptureBackend.latest
        self.assertEqual(frame.value, 42)
        self.assertEqual(backend.options["window_hwnd"], 123)
        self.assertFalse(backend.options["cursor_capture"])
        self.assertEqual(source.frames_captured, 1)
        self.assertTrue(backend.control.stopped)


if __name__ == "__main__":
    unittest.main()
