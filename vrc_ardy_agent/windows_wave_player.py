from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
import threading
from typing import Any
import wave


CompletionCallback = Callable[[str, str | None], None]


@dataclass(slots=True)
class _Playback:
    frames: bytes
    sample_rate: int
    channels: int
    dtype: str
    bytes_per_frame: int
    stop_event: threading.Event
    completed: CompletionCallback
    thread: threading.Thread | None = None
    stream: Any | None = None


class WaveAudioPlayer:
    """Owns one cancellable WAV playback worker for a virtual microphone."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        chunk_frames: int = 1600,
        stop_timeout_seconds: float = 2.0,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if stop_timeout_seconds <= 0:
            raise ValueError("stop_timeout_seconds must be positive")
        self.device = device
        self.chunk_frames = int(chunk_frames)
        self.stop_timeout_seconds = float(stop_timeout_seconds)
        self._stream_factory = stream_factory or _raw_output_stream
        self._condition = threading.Condition(threading.RLock())
        self._current: _Playback | None = None
        self._last_error: str | None = None

    def play(self, wav_bytes: bytes, completed: CompletionCallback) -> None:
        if not callable(completed):
            raise TypeError("completed must be callable")
        playback = _parse_wav(wav_bytes, completed=completed)
        with self._condition:
            if self._current is not None:
                raise RuntimeError("WAV playback is already active")
            thread = threading.Thread(
                target=self._run,
                args=(playback,),
                name="windows-wave-playback",
                daemon=True,
            )
            playback.thread = thread
            self._current = playback
            self._last_error = None
            thread.start()

    def stop(self) -> None:
        with self._condition:
            playback = self._current
            if playback is None:
                return
            playback.stop_event.set()
            stream = playback.stream
            thread = playback.thread
        if stream is not None:
            abort = getattr(stream, "abort", None)
            if callable(abort):
                abort()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.stop_timeout_seconds)
            if thread.is_alive():
                raise TimeoutError("WAV playback worker did not stop")

    def wait_until_idle(self, *, timeout: float | None = None) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._current is None,
                timeout=timeout,
            )

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    def _run(self, playback: _Playback) -> None:
        state = "completed"
        error: str | None = None
        try:
            with self._stream_factory(
                samplerate=playback.sample_rate,
                channels=playback.channels,
                dtype=playback.dtype,
                device=self.device,
            ) as stream:
                with self._condition:
                    playback.stream = stream
                chunk_bytes = self.chunk_frames * playback.bytes_per_frame
                for offset in range(0, len(playback.frames), chunk_bytes):
                    if playback.stop_event.is_set():
                        state = "cancelled"
                        break
                    stream.write(playback.frames[offset : offset + chunk_bytes])
                if playback.stop_event.is_set():
                    state = "cancelled"
        except Exception as exc:
            if playback.stop_event.is_set():
                state = "cancelled"
            else:
                state = "failed"
                error = f"{type(exc).__name__}: {exc}"
                with self._condition:
                    self._last_error = error
        finally:
            try:
                playback.completed(state, error)
            except Exception as exc:
                with self._condition:
                    self._last_error = (
                        f"completion callback failed: {type(exc).__name__}: {exc}"
                    )
            with self._condition:
                playback.stream = None
                if self._current is playback:
                    self._current = None
                self._condition.notify_all()


def _parse_wav(
    wav_bytes: bytes,
    *,
    completed: CompletionCallback,
) -> _Playback:
    if not isinstance(wav_bytes, bytes) or not wav_bytes:
        raise ValueError("wav_bytes must not be empty")
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as reader:
            channels = reader.getnchannels()
            sample_rate = reader.getframerate()
            sample_width = reader.getsampwidth()
            compression = reader.getcomptype()
            frames = reader.readframes(reader.getnframes())
    except (EOFError, wave.Error) as exc:
        raise ValueError("wav_bytes is not a valid WAV") from exc
    if compression != "NONE":
        raise ValueError("compressed WAV audio is not supported")
    if channels <= 0 or sample_rate <= 0 or not frames:
        raise ValueError("WAV has no playable frames")
    dtypes = {1: "uint8", 2: "int16", 3: "int24", 4: "int32"}
    try:
        dtype = dtypes[sample_width]
    except KeyError as exc:
        raise ValueError(f"unsupported WAV sample width: {sample_width}") from exc
    return _Playback(
        frames=frames,
        sample_rate=sample_rate,
        channels=channels,
        dtype=dtype,
        bytes_per_frame=channels * sample_width,
        stop_event=threading.Event(),
        completed=completed,
    )


def _raw_output_stream(**options: object) -> Any:
    try:
        import sounddevice
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required for Windows virtual-microphone output"
        ) from exc
    return sounddevice.RawOutputStream(**options)
