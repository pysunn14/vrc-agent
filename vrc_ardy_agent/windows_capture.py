from __future__ import annotations

from dataclasses import dataclass
import queue
import sys
import threading
from typing import Any


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    process_id: int
    width: int
    height: int
    minimized: bool


class LatestFrameQueue:
    """Keep only the freshest frame so slow inference cannot build latency."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._closed = False

    def publish(self, frame: Any) -> None:
        copied = frame.copy()
        with self._lock:
            if self._closed:
                return
            try:
                self._queue.put_nowait(copied)
                return
            except queue.Full:
                pass
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(copied)

    def read(self, *, timeout_seconds: float) -> Any:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        with self._lock:
            if self._closed and self._queue.empty():
                raise EOFError("capture is closed")
        try:
            return self._queue.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            with self._lock:
                if self._closed:
                    raise EOFError("capture is closed") from exc
            raise TimeoutError("no captured frame arrived before the timeout") from exc

    def close(self) -> None:
        with self._lock:
            self._closed = True


def list_windows(*, title_filter: str | None = None) -> list[WindowInfo]:
    if sys.platform != "win32":
        raise RuntimeError("window enumeration is only available on Windows")

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    windows: list[WindowInfo] = []
    normalized_filter = title_filter.casefold() if title_filter else None

    def visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title_length = int(user32.GetWindowTextLengthW(hwnd))
        if title_length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, buffer, len(buffer))
        title = buffer.value.strip()
        if not title:
            return True
        if normalized_filter is not None and normalized_filter not in title.casefold():
            return True

        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                process_id=int(process_id.value),
                width=max(0, int(rect.right - rect.left)),
                height=max(0, int(rect.bottom - rect.top)),
                minimized=bool(user32.IsIconic(hwnd)),
            )
        )
        return True

    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    callback = callback_type(visit)
    if not user32.EnumWindows(callback, 0):
        raise ctypes.WinError()
    return sorted(windows, key=lambda window: (window.title.casefold(), window.hwnd))


class WindowsGraphicsCaptureSource:
    """Capture a single HWND through Windows Graphics Capture."""

    def __init__(self, *, hwnd: int, minimum_update_interval_ms: int = 50) -> None:
        if hwnd <= 0:
            raise ValueError("hwnd must be positive")
        if minimum_update_interval_ms < 0:
            raise ValueError("minimum_update_interval_ms must be non-negative")
        self.hwnd = int(hwnd)
        self.minimum_update_interval_ms = int(minimum_update_interval_ms)
        self._frames = LatestFrameQueue()
        self._capture: Any | None = None
        self._control: Any | None = None
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self.frames_captured = 0

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("capture is already started")
            if self._closed:
                raise RuntimeError("capture is closed")
        if sys.platform != "win32":
            raise RuntimeError("Windows Graphics Capture is only available on Windows")
        try:
            from windows_capture import WindowsCapture
        except ImportError as exc:
            raise RuntimeError(
                "windows-capture is missing; install requirements-windows.txt on Windows"
            ) from exc

        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            minimum_update_interval=self.minimum_update_interval_ms,
            window_hwnd=self.hwnd,
        )

        @capture.event
        def on_frame_arrived(frame: Any, _capture_control: Any) -> None:
            bgr_frame = frame.convert_to_bgr().frame_buffer
            self._frames.publish(bgr_frame)
            with self._lock:
                self.frames_captured += 1

        @capture.event
        def on_closed() -> None:
            self._frames.close()

        with self._lock:
            self._capture = capture
            self._started = True
        try:
            control = capture.start_free_threaded()
        except BaseException:
            with self._lock:
                self._capture = None
                self._started = False
            raise
        with self._lock:
            closed_while_starting = self._closed
            if not closed_while_starting:
                self._control = control
        if closed_while_starting:
            control.stop()
            raise RuntimeError("capture was closed while starting")

    def read(self, *, timeout_seconds: float) -> Any:
        return self._frames.read(timeout_seconds=timeout_seconds)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            control = self._control
            self._control = None
        try:
            if control is not None:
                control.stop()
        finally:
            self._frames.close()
