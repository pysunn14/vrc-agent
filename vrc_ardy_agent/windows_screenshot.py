from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Protocol

from .stream_protocol import MAX_SCREENSHOT_BYTES


class FreshFrameSource(Protocol):
    def read_after(
        self,
        *,
        after_monotonic_ns: int,
        timeout_seconds: float,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ScreenshotSnapshot:
    requests: int
    screenshots_encoded: int
    failures: int
    heartbeat_monotonic: float
    last_error: str | None


@dataclass(frozen=True, slots=True)
class ObserverScreenshotSnapshot:
    requests: int
    captures: int
    failures: int
    target_process_id: int | None
    target_display_name: str | None
    heartbeat_monotonic: float
    last_error: str | None
    resolution_diagnostics: object | None


class WindowsJpegScreenshotProvider:
    """Captures and encodes a frame newer than each incoming request."""

    def __init__(
        self,
        *,
        source: FreshFrameSource,
        jpeg_quality: int = 85,
        timeout_seconds: float = 2.0,
        encoder: Callable[[Any, int], bytes] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
            raise ValueError("jpeg_quality must be an integer")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._source = source
        self._jpeg_quality = jpeg_quality
        self._timeout_seconds = float(timeout_seconds)
        self._encoder = encoder or _encode_jpeg
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns
        self._lock = threading.Lock()
        self._status = ScreenshotSnapshot(
            requests=0,
            screenshots_encoded=0,
            failures=0,
            heartbeat_monotonic=self._monotonic(),
            last_error=None,
        )

    def __call__(self) -> bytes:
        requested_at = self._monotonic_ns()
        with self._lock:
            self._status = replace(
                self._status,
                requests=self._status.requests + 1,
                heartbeat_monotonic=self._monotonic(),
            )
        try:
            frame = self._source.read_after(
                after_monotonic_ns=requested_at,
                timeout_seconds=self._timeout_seconds,
            )
            jpeg = self._encoder(frame, self._jpeg_quality)
            if (
                not isinstance(jpeg, bytes)
                or len(jpeg) < 4
                or not jpeg.startswith(b"\xff\xd8")
                or not jpeg.endswith(b"\xff\xd9")
            ):
                raise RuntimeError("screenshot encoder did not return a valid JPEG")
            if len(jpeg) > MAX_SCREENSHOT_BYTES:
                raise RuntimeError("encoded screenshot exceeds the protocol size limit")
        except Exception as exc:
            with self._lock:
                self._status = replace(
                    self._status,
                    failures=self._status.failures + 1,
                    heartbeat_monotonic=self._monotonic(),
                    last_error=f"{type(exc).__name__}: {exc}",
                )
            raise
        with self._lock:
            self._status = replace(
                self._status,
                screenshots_encoded=self._status.screenshots_encoded + 1,
                heartbeat_monotonic=self._monotonic(),
                last_error=None,
            )
        return jpeg

    def snapshot(self) -> ScreenshotSnapshot:
        with self._lock:
            return replace(self._status)


class VrchatObserverScreenshotProvider:
    """Resolve and capture a separate VRChat observer without stale fallback.

    The controlled avatar's capture remains the VLM point of view. This
    provider opens the observer window only for an explicit diagnostic request,
    so visual testing does not silently replace the character's perception.
    """

    def __init__(
        self,
        *,
        resolve_target: Callable[[], Any | None],
        source_factory: Callable[[int], Any],
        jpeg_quality: int = 85,
        timeout_seconds: float = 2.0,
        encoder: Callable[[Any, int], bytes] | None = None,
        resolution_diagnostics: Callable[[], object] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(resolve_target):
            raise TypeError("resolve_target must be callable")
        if not callable(source_factory):
            raise TypeError("source_factory must be callable")
        self._resolve_target = resolve_target
        self._source_factory = source_factory
        self._jpeg_quality = jpeg_quality
        self._timeout_seconds = timeout_seconds
        self._encoder = encoder
        self._resolution_diagnostics = resolution_diagnostics
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._status = ObserverScreenshotSnapshot(
            requests=0,
            captures=0,
            failures=0,
            target_process_id=None,
            target_display_name=None,
            heartbeat_monotonic=self._monotonic(),
            last_error=None,
            resolution_diagnostics=None,
        )

    def __call__(self) -> bytes:
        with self._lock:
            self._status = replace(
                self._status,
                requests=self._status.requests + 1,
                heartbeat_monotonic=self._monotonic(),
            )
            try:
                target = self._resolve_target()
                if target is None:
                    raise RuntimeError("observer VRChat account is not running")
                window = target.window
                source = self._source_factory(window.hwnd)
                try:
                    source.start()
                    jpeg = WindowsJpegScreenshotProvider(
                        source=source,
                        jpeg_quality=self._jpeg_quality,
                        timeout_seconds=self._timeout_seconds,
                        encoder=self._encoder,
                        monotonic=self._monotonic,
                    )()
                finally:
                    source.close()
            except Exception as exc:
                diagnostics: object | None = None
                if self._resolution_diagnostics is not None:
                    try:
                        diagnostics = self._resolution_diagnostics()
                    except Exception as diagnostic_exc:
                        diagnostics = {
                            "error": (
                                f"{type(diagnostic_exc).__name__}: "
                                f"{diagnostic_exc}"
                            )
                        }
                self._status = replace(
                    self._status,
                    failures=self._status.failures + 1,
                    heartbeat_monotonic=self._monotonic(),
                    last_error=f"{type(exc).__name__}: {exc}",
                    resolution_diagnostics=diagnostics,
                )
                raise
            self._status = replace(
                self._status,
                captures=self._status.captures + 1,
                target_process_id=int(window.process_id),
                target_display_name=str(target.identity.display_name),
                heartbeat_monotonic=self._monotonic(),
                last_error=None,
                resolution_diagnostics=None,
            )
            return jpeg

    def snapshot(self) -> ObserverScreenshotSnapshot:
        with self._lock:
            return replace(self._status)


def _encode_jpeg(frame: Any, quality: int) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is missing; install requirements-windows.txt on Windows"
        ) from exc
    succeeded, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not succeeded:
        raise RuntimeError("OpenCV failed to encode the screenshot")
    return encoded.tobytes()
