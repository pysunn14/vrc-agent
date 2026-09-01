from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

from vrc_ardy_agent.windows_capture import LatestFrameQueue, WindowsGraphicsCaptureSource


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
