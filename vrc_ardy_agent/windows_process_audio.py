from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import BinaryIO

import numpy as np
from scipy import signal


TARGET_SAMPLE_RATE = 16_000
TARGET_FRAME_SAMPLES = 320
_READ_SIZE = 64 * 1024
_DEFAULT_HELPER = (
    Path(__file__).resolve().parents[1]
    / "native"
    / "windows_process_audio_probe"
    / "src"
    / "bin"
    / "Release"
    / "net10.0-windows10.0.19041.0"
    / "vrc-ardy-process-audio-probe.dll"
)


class StreamingFloatPcmTo16kMono:
    """Convert arbitrary callback chunks without resetting filter or sample phase."""

    def __init__(self, *, sample_rate: int, channels: int) -> None:
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate < TARGET_SAMPLE_RATE
        ):
            raise ValueError("sample_rate must be an integer of at least 16000 Hz")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        self.sample_rate = sample_rate
        self.channels = channels
        self._bytes_per_source_frame = channels * np.dtype("<f4").itemsize
        self._filter = signal.butter(
            8,
            7_200,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )
        self._filter_state = np.zeros(
            (self._filter.shape[0], 2),
            dtype=np.float64,
        )
        self._source_byte_pending = b""
        self._previous_filtered_sample: float | None = None
        self._next_output_index = 0
        self._pending_output = np.empty(0, dtype="<i2")
        self.source_frames_seen = 0

    def push(self, pcm_f32le: bytes) -> tuple[bytes, ...]:
        if not isinstance(pcm_f32le, bytes) or not pcm_f32le:
            raise ValueError("source PCM must be non-empty bytes")
        combined = self._source_byte_pending + pcm_f32le
        complete_bytes = (
            len(combined) // self._bytes_per_source_frame
        ) * self._bytes_per_source_frame
        self._source_byte_pending = combined[complete_bytes:]
        if complete_bytes == 0:
            return ()

        samples = np.frombuffer(combined[:complete_bytes], dtype="<f4")
        if not bool(np.isfinite(samples).all()):
            raise ValueError("source PCM samples must be finite")
        source_start = self.source_frames_seen
        source = samples.reshape(-1, self.channels)
        mono = source.mean(axis=1, dtype=np.float64)
        filtered, self._filter_state = signal.sosfilt(
            self._filter,
            mono,
            zi=self._filter_state,
        )
        self.source_frames_seen += mono.size

        if self._previous_filtered_sample is None:
            interpolation_samples = filtered
            interpolation_start = source_start
        else:
            interpolation_samples = np.concatenate(
                ([self._previous_filtered_sample], filtered)
            )
            interpolation_start = source_start - 1
        self._previous_filtered_sample = float(filtered[-1])

        final_source_index = self.source_frames_seen - 1
        final_output_index = int(
            np.floor(
                final_source_index * TARGET_SAMPLE_RATE / self.sample_rate + 1e-12
            )
        )
        if final_output_index < self._next_output_index:
            return ()

        output_indices = np.arange(
            self._next_output_index,
            final_output_index + 1,
            dtype=np.int64,
        )
        source_positions = (
            output_indices.astype(np.float64) * self.sample_rate / TARGET_SAMPLE_RATE
        )
        interpolation_positions = interpolation_start + np.arange(
            interpolation_samples.size,
            dtype=np.float64,
        )
        resampled = np.interp(
            source_positions,
            interpolation_positions,
            interpolation_samples,
        )
        self._next_output_index = final_output_index + 1

        quantized = np.rint(np.clip(resampled, -1.0, 1.0) * 32_767.0).astype(
            "<i2"
        )
        if self._pending_output.size:
            quantized = np.concatenate((self._pending_output, quantized))
        complete_frames = quantized.size // TARGET_FRAME_SAMPLES
        frames = tuple(
            quantized[
                index * TARGET_FRAME_SAMPLES : (index + 1) * TARGET_FRAME_SAMPLES
            ].tobytes()
            for index in range(complete_frames)
        )
        self._pending_output = quantized[
            complete_frames * TARGET_FRAME_SAMPLES :
        ].copy()
        return frames


@dataclass(frozen=True, slots=True)
class NativeProcessAudioSnapshot:
    running: bool
    pid: int
    sample_rate: int | None
    channels: int | None
    chunks_received: int
    source_frames_received: int
    frames_published: int
    conversion_failures: int
    publish_failures: int
    native_heartbeats: int
    bytes_streamed: int
    heartbeat_monotonic: float
    last_error: str | None


class NativeProcessAudioSource:
    """Own the strict native process-loopback helper and publish 20 ms speech frames."""

    def __init__(
        self,
        *,
        pid: int,
        frame_handler: Callable[[bytes, int], object],
        helper_path: str | Path | None = None,
        helper_command: Sequence[str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        startup_timeout_seconds: float = 10.0,
        heartbeat_timeout_seconds: float = 5.0,
    ) -> None:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        if not callable(frame_handler):
            raise TypeError("frame_handler must be callable")
        if helper_path is not None and helper_command is not None:
            raise ValueError("helper_path and helper_command are mutually exclusive")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be positive")

        self.pid = pid
        self._frame_handler = frame_handler
        self._helper_path = Path(helper_path) if helper_path is not None else None
        self._helper_command = tuple(helper_command) if helper_command is not None else None
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns
        self._startup_timeout_seconds = float(startup_timeout_seconds)
        self._heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._startup_event = threading.Event()
        self._startup_error: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._converter: StreamingFloatPcmTo16kMono | None = None
        self._closed = False
        now = self._monotonic()
        self._status = NativeProcessAudioSnapshot(
            running=False,
            pid=pid,
            sample_rate=None,
            channels=None,
            chunks_received=0,
            source_frames_received=0,
            frames_published=0,
            conversion_failures=0,
            publish_failures=0,
            native_heartbeats=0,
            bytes_streamed=0,
            heartbeat_monotonic=now,
            last_error=None,
        )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("native process audio source is closed")
            if self._process is not None:
                raise RuntimeError("native process audio source is already started")

        command = [*self._resolve_helper_command(), "stream", "--pid", str(self.pid)]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            self._record_start_failure(exc)
            raise
        if process.stdout is None or process.stderr is None:
            process.terminate()
            error = RuntimeError("native helper pipes were not created")
            self._record_start_failure(error)
            raise error

        with self._lock:
            self._process = process
        stderr_thread = threading.Thread(
            target=self._read_status,
            args=(process.stderr,),
            name="vrchat-process-audio-status",
            daemon=True,
        )
        with self._lock:
            self._stderr_thread = stderr_thread
        stderr_thread.start()

        if not self._startup_event.wait(self._startup_timeout_seconds):
            error = RuntimeError("native process audio helper startup timed out")
            self._fail(error)
            self.close()
            raise error
        with self._lock:
            startup_error = self._startup_error
            converter = self._converter
        if startup_error is not None or converter is None:
            error = RuntimeError(startup_error or "native helper did not publish its format")
            self.close()
            raise error
        if process.poll() is not None:
            error = RuntimeError(
                f"native process audio helper exited during startup with code {process.returncode}"
            )
            self._fail(error)
            self.close()
            raise error

        with self._lock:
            self._status = replace(
                self._status,
                running=True,
                heartbeat_monotonic=self._monotonic(),
                last_error=None,
            )
        stdout_thread = threading.Thread(
            target=self._read_audio,
            args=(process.stdout,),
            name="vrchat-process-audio-stream",
            daemon=True,
        )
        monitor_thread = threading.Thread(
            target=self._monitor,
            name="vrchat-process-audio-monitor",
            daemon=True,
        )
        with self._lock:
            self._stdout_thread = stdout_thread
            self._monitor_thread = monitor_thread
        stdout_thread.start()
        monitor_thread.start()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            process = self._process
            threads = (
                self._stdout_thread,
                self._stderr_thread,
                self._monitor_thread,
            )

        close_error: BaseException | None = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired as exc:
                    close_error = exc
        for thread in threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
                if thread.is_alive() and close_error is None:
                    close_error = TimeoutError(f"{thread.name} did not stop")
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

        with self._lock:
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

    def snapshot(self) -> NativeProcessAudioSnapshot:
        with self._lock:
            return replace(self._status)

    def _resolve_helper_command(self) -> tuple[str, ...]:
        if self._helper_command is not None:
            if not self._helper_command:
                raise ValueError("helper_command must not be empty")
            return self._helper_command
        helper = self._helper_path or _DEFAULT_HELPER
        if not helper.is_file():
            raise RuntimeError(
                f"native process audio helper is not built: {helper}; "
                "build native/windows_process_audio_probe/src in Release mode"
            )
        if helper.suffix.lower() == ".dll":
            dotnet = shutil.which("dotnet")
            if dotnet is None:
                raise RuntimeError("dotnet is required to run the process audio helper")
            return (dotnet, str(helper))
        return (str(helper),)

    def _read_status(self, stream: BinaryIO) -> None:
        try:
            while not self._stop_event.is_set():
                line = stream.readline()
                if not line:
                    break
                try:
                    event = json.loads(line.decode("utf-8"))
                    self._handle_status_event(event)
                except Exception as exc:
                    self._fail(
                        RuntimeError(
                            f"invalid native helper status: {type(exc).__name__}: {exc}"
                        )
                    )
                    return
        finally:
            if not self._stop_event.is_set():
                with self._lock:
                    process = self._process
                return_code = None if process is None else process.poll()
                self._fail(
                    RuntimeError(
                        "native process audio helper status stream closed"
                        + ("" if return_code is None else f" with code {return_code}")
                    )
                )

    def _handle_status_event(self, event: object) -> None:
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise ValueError("status event must be an object with an event name")
        event_name = event["event"]
        if event_name == "started":
            if event.get("pid") != self.pid:
                raise ValueError(
                    f"native helper started for pid {event.get('pid')!r}, expected {self.pid}"
                )
            sample_rate, channels = _require_float32_format(event.get("format"))
            converter = StreamingFloatPcmTo16kMono(
                sample_rate=sample_rate,
                channels=channels,
            )
            with self._lock:
                if self._converter is not None:
                    raise ValueError("native helper emitted started more than once")
                self._converter = converter
                self._status = replace(
                    self._status,
                    sample_rate=sample_rate,
                    channels=channels,
                    heartbeat_monotonic=self._monotonic(),
                )
            self._startup_event.set()
            return
        if event_name == "heartbeat":
            bytes_streamed = event.get("bytes_streamed")
            if isinstance(bytes_streamed, bool) or not isinstance(bytes_streamed, int):
                raise ValueError("heartbeat bytes_streamed must be an integer")
            with self._lock:
                self._status = replace(
                    self._status,
                    native_heartbeats=self._status.native_heartbeats + 1,
                    bytes_streamed=bytes_streamed,
                    heartbeat_monotonic=self._monotonic(),
                )
            return
        if event_name == "error":
            message = event.get("message")
            if not isinstance(message, str) or not message:
                raise ValueError("native error event must contain a message")
            self._fail(RuntimeError(f"native process audio helper failed: {message}"))
            return
        raise ValueError(f"unknown native helper status event: {event_name}")

    def _read_audio(self, stream: BinaryIO) -> None:
        while not self._stop_event.is_set():
            chunk = stream.read(_READ_SIZE)
            if not chunk:
                return
            with self._lock:
                converter = self._converter
                if converter is None:
                    self._fail(RuntimeError("audio arrived before the native format"))
                    return
                previous_source_frames = converter.source_frames_seen
            try:
                frames = converter.push(chunk)
            except Exception as exc:
                with self._lock:
                    self._status = replace(
                        self._status,
                        conversion_failures=self._status.conversion_failures + 1,
                    )
                self._fail(
                    RuntimeError(
                        f"audio conversion failed: {type(exc).__name__}: {exc}"
                    )
                )
                return
            received = converter.source_frames_seen - previous_source_frames
            with self._lock:
                self._status = replace(
                    self._status,
                    chunks_received=self._status.chunks_received + 1,
                    source_frames_received=(
                        self._status.source_frames_received + received
                    ),
                    heartbeat_monotonic=self._monotonic(),
                )

            for frame in frames:
                captured_at = self._monotonic_ns()
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
        interval = min(1.0, self._heartbeat_timeout_seconds / 2.0)
        while not self._stop_event.wait(interval):
            with self._lock:
                process = self._process
                status = self._status
            if process is None:
                return
            return_code = process.poll()
            if return_code is not None:
                self._fail(
                    RuntimeError(
                        f"native process audio helper exited unexpectedly with code {return_code}"
                    )
                )
                return
            if self._monotonic() - status.heartbeat_monotonic > self._heartbeat_timeout_seconds:
                self._fail(RuntimeError("native process audio helper heartbeat is stale"))
                return

    def _fail(self, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        with self._lock:
            if self._status.last_error is not None and not self._status.running:
                self._startup_event.set()
                return
            if self._startup_error is None:
                self._startup_error = message
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=self._monotonic(),
                last_error=message,
            )
            process = self._process
        self._startup_event.set()
        if process is not None and process.poll() is None:
            process.terminate()

    def _record_start_failure(self, exc: BaseException) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=self._monotonic(),
                last_error=f"start failed: {type(exc).__name__}: {exc}",
            )


def _require_float32_format(value: object) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise RuntimeError("native helper format must be an object")
    sample_rate = value.get("sample_rate")
    channels = value.get("channels")
    bits_per_sample = value.get("bits_per_sample")
    sample_format = value.get("sample_format")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or isinstance(channels, bool)
        or not isinstance(channels, int)
        or bits_per_sample != 32
        or sample_format != "float32"
    ):
        raise RuntimeError(f"expected native float32 PCM format, got {value!r}")
    return sample_rate, channels
