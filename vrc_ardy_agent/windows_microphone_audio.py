from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any

import numpy as np
from scipy import signal


SOURCE_SAMPLE_RATE = 48_000
SOURCE_CHANNELS = 1
TARGET_SAMPLE_RATE = 16_000
TARGET_FRAME_SAMPLES = 320


class Pcm48kMonoTo16kMono:
    """Convert native 48 kHz microphone PCM into fixed 20 ms STT frames."""

    def __init__(self) -> None:
        self._filter = signal.butter(
            8,
            7_200,
            btype="lowpass",
            fs=SOURCE_SAMPLE_RATE,
            output="sos",
        )
        self._filter_state = np.zeros((self._filter.shape[0], 2), dtype=np.float64)
        self._source_samples_seen = 0
        self._pending = np.empty(0, dtype="<i2")

    def push(self, pcm_s16le_mono: bytes) -> tuple[bytes, ...]:
        if not isinstance(pcm_s16le_mono, bytes) or not pcm_s16le_mono:
            raise ValueError("source PCM must not be empty")
        if len(pcm_s16le_mono) % np.dtype("<i2").itemsize:
            raise ValueError("source PCM must contain complete mono int16 samples")

        samples = np.frombuffer(pcm_s16le_mono, dtype="<i2").astype(np.float64)
        filtered, self._filter_state = signal.sosfilt(
            self._filter,
            samples,
            zi=self._filter_state,
        )
        first = (-self._source_samples_seen) % 3
        downsampled = filtered[first::3]
        self._source_samples_seen += samples.size
        quantized = np.rint(np.clip(downsampled, -32_768.0, 32_767.0)).astype(
            "<i2"
        )
        if self._pending.size:
            quantized = np.concatenate((self._pending, quantized))

        complete = quantized.size // TARGET_FRAME_SAMPLES
        frames = tuple(
            quantized[
                index * TARGET_FRAME_SAMPLES : (index + 1) * TARGET_FRAME_SAMPLES
            ].tobytes()
            for index in range(complete)
        )
        self._pending = quantized[complete * TARGET_FRAME_SAMPLES :].copy()
        return frames


@dataclass(frozen=True, slots=True)
class WasapiMicrophoneAudioSnapshot:
    running: bool
    device: str | int
    chunks_received: int
    source_frames_received: int
    frames_published: int
    overflow_events: int
    conversion_failures: int
    publish_failures: int
    monitor_ticks: int
    peak_sample: int
    rms_sample: int
    heartbeat_monotonic: float
    last_error: str | None


class WasapiMicrophoneAudioSource:
    """Capture one native 48 kHz mono WASAPI microphone endpoint."""

    def __init__(
        self,
        *,
        device: str | int,
        frame_handler: Callable[[bytes, int], object],
        stream_factory: Callable[[str | int, Callable[..., None]], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        monitor_interval_seconds: float = 1.0,
    ) -> None:
        if isinstance(device, str) and not device.strip():
            raise ValueError("microphone device must not be empty")
        if isinstance(device, bool) or not isinstance(device, (str, int)):
            raise TypeError("microphone device must be a name or index")
        if isinstance(device, int) and device < 0:
            raise ValueError("microphone device index must be non-negative")
        if not callable(frame_handler):
            raise TypeError("frame_handler must be callable")
        if monitor_interval_seconds <= 0:
            raise ValueError("monitor_interval_seconds must be positive")
        self.device = device
        self._frame_handler = frame_handler
        self._stream_factory = stream_factory or _create_microphone_stream
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns
        self._monitor_interval_seconds = float(monitor_interval_seconds)
        self._converter = Pcm48kMonoTo16kMono()
        self._callback_lock = threading.Lock()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._stream: Any | None = None
        self._closed = False
        self._status = WasapiMicrophoneAudioSnapshot(
            running=False,
            device=device,
            chunks_received=0,
            source_frames_received=0,
            frames_published=0,
            overflow_events=0,
            conversion_failures=0,
            publish_failures=0,
            monitor_ticks=0,
            peak_sample=0,
            rms_sample=0,
            heartbeat_monotonic=self._monotonic(),
            last_error=None,
        )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("microphone audio source is closed")
            if self._stream is not None:
                raise RuntimeError("microphone audio source is already started")

        stream: Any | None = None
        try:
            stream = self._stream_factory(self.device, self._on_audio)
            stream.start()
            if not bool(stream.active):
                raise RuntimeError("microphone stream did not enter the active state")
        except Exception as exc:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self._record_start_failure(exc)
            raise

        with self._lock:
            self._stream = stream
            self._stop_event.clear()
            self._status = replace(
                self._status,
                running=True,
                heartbeat_monotonic=self._monotonic(),
                last_error=None,
            )
            monitor = threading.Thread(
                target=self._monitor,
                name="windows-microphone-audio-monitor",
                daemon=True,
            )
            self._monitor_thread = monitor
            monitor.start()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            stream = self._stream
            monitor = self._monitor_thread
            self._stop_event.set()

        close_error: BaseException | None = None
        if stream is not None:
            try:
                if bool(stream.active):
                    stream.stop()
                stream.close()
            except BaseException as exc:
                close_error = exc
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=self._monitor_interval_seconds + 1.0)
            if monitor.is_alive() and close_error is None:
                close_error = TimeoutError("microphone monitor did not stop")

        with self._lock:
            self._stream = None
            self._monitor_thread = None
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=self._monotonic(),
                last_error=(
                    self._status.last_error
                    if close_error is None
                    else f"close failed: {type(close_error).__name__}: {close_error}"
                ),
            )
        if close_error is not None:
            raise close_error

    def snapshot(self) -> WasapiMicrophoneAudioSnapshot:
        with self._lock:
            return replace(self._status)

    def _on_audio(
        self,
        input_data: object,
        _reported_frames: int,
        _time_info: object,
        capture_status: object,
    ) -> None:
        pcm = bytes(input_data)
        with self._callback_lock:
            samples = np.frombuffer(pcm, dtype="<i2") if len(pcm) % 2 == 0 else None
            peak = int(np.max(np.abs(samples.astype(np.int32)))) if samples is not None and samples.size else 0
            rms = (
                math.isqrt(
                    int(np.mean(np.square(samples.astype(np.int64))))
                )
                if samples is not None and samples.size
                else 0
            )
            overflowed = bool(
                getattr(capture_status, "input_overflow", capture_status)
            )
            with self._lock:
                self._status = replace(
                    self._status,
                    chunks_received=self._status.chunks_received + 1,
                    source_frames_received=(
                        self._status.source_frames_received
                        + (0 if samples is None else samples.size)
                    ),
                    overflow_events=self._status.overflow_events + overflowed,
                    peak_sample=peak,
                    rms_sample=rms,
                    heartbeat_monotonic=self._monotonic(),
                    last_error=(
                        f"microphone capture status: {capture_status}"
                        if capture_status
                        else self._status.last_error
                    ),
                )
            try:
                frames = self._converter.push(pcm)
            except Exception as exc:
                with self._lock:
                    self._status = replace(
                        self._status,
                        conversion_failures=self._status.conversion_failures + 1,
                        heartbeat_monotonic=self._monotonic(),
                        last_error=(
                            f"audio conversion failed: {type(exc).__name__}: {exc}"
                        ),
                    )
                return

            captured_at = self._monotonic_ns()
            for frame in frames:
                try:
                    self._frame_handler(frame, captured_at)
                except Exception as exc:
                    with self._lock:
                        self._status = replace(
                            self._status,
                            publish_failures=self._status.publish_failures + 1,
                            heartbeat_monotonic=self._monotonic(),
                            last_error=(
                                f"audio publish failed: {type(exc).__name__}: {exc}"
                            ),
                        )
                    continue
                with self._lock:
                    self._status = replace(
                        self._status,
                        frames_published=self._status.frames_published + 1,
                        heartbeat_monotonic=self._monotonic(),
                    )

    def _monitor(self) -> None:
        while not self._stop_event.wait(self._monitor_interval_seconds):
            with self._lock:
                stream = self._stream
                if stream is None:
                    return
            if not bool(stream.active):
                with self._lock:
                    self._status = replace(
                        self._status,
                        running=False,
                        monitor_ticks=self._status.monitor_ticks + 1,
                        heartbeat_monotonic=self._monotonic(),
                        last_error="microphone stream stopped unexpectedly",
                    )
                return
            with self._lock:
                self._status = replace(
                    self._status,
                    monitor_ticks=self._status.monitor_ticks + 1,
                    heartbeat_monotonic=self._monotonic(),
                )

    def _record_start_failure(self, exc: BaseException) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=self._monotonic(),
                last_error=f"start failed: {type(exc).__name__}: {exc}",
            )


def _create_microphone_stream(
    device: str | int,
    callback: Callable[..., None],
) -> Any:
    try:
        import sounddevice
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is missing; install requirements-windows.txt on Windows"
        ) from exc
    sounddevice.check_input_settings(
        device=device,
        channels=SOURCE_CHANNELS,
        dtype="int16",
        samplerate=SOURCE_SAMPLE_RATE,
    )
    return sounddevice.RawInputStream(
        device=device,
        samplerate=SOURCE_SAMPLE_RATE,
        channels=SOURCE_CHANNELS,
        dtype="int16",
        blocksize=SOURCE_SAMPLE_RATE // 50,
        callback=callback,
    )
