from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import struct
import threading
import time


@dataclass(frozen=True, slots=True)
class EnergyVadSnapshot:
    active: bool
    frames_received: int
    utterances_emitted: int
    utterances_discarded: int
    resets: int
    last_rms: int
    heartbeat_monotonic: float
    last_reset_reason: str | None


class EnergyVadSegmenter:
    """A deterministic 20 ms energy gate for the first interaction PoC."""

    SAMPLE_RATE = 16000
    FRAME_MS = 20
    FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
    FRAME_BYTES = FRAME_SAMPLES * 2

    def __init__(
        self,
        *,
        rms_threshold: int = 500,
        start_trigger_ms: int = 60,
        end_silence_ms: int = 600,
        pre_roll_ms: int = 200,
        minimum_speech_ms: int = 200,
        maximum_utterance_ms: int = 15000,
    ) -> None:
        if rms_threshold <= 0:
            raise ValueError("rms_threshold must be positive")
        for name, value, allow_zero in (
            ("start_trigger_ms", start_trigger_ms, False),
            ("end_silence_ms", end_silence_ms, False),
            ("pre_roll_ms", pre_roll_ms, True),
            ("minimum_speech_ms", minimum_speech_ms, False),
            ("maximum_utterance_ms", maximum_utterance_ms, False),
        ):
            if value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"{name} must be positive")
            if value % self.FRAME_MS:
                raise ValueError(f"{name} must be a multiple of {self.FRAME_MS} ms")
        if maximum_utterance_ms < start_trigger_ms:
            raise ValueError("maximum_utterance_ms must cover the start trigger")
        self.rms_threshold = int(rms_threshold)
        self.start_trigger_ms = int(start_trigger_ms)
        self.end_silence_ms = int(end_silence_ms)
        self.pre_roll_ms = int(pre_roll_ms)
        self.minimum_speech_ms = int(minimum_speech_ms)
        self.maximum_utterance_ms = int(maximum_utterance_ms)
        self._lock = threading.Lock()
        self._pre_roll: deque[bytes] = deque(
            maxlen=pre_roll_ms // self.FRAME_MS
        )
        self._candidate: list[bytes] = []
        self._candidate_ms = 0
        self._active_frames: list[bytes] | None = None
        self._active_voiced_ms = 0
        self._active_duration_ms = 0
        self._trailing_silence_ms = 0
        self._frames_received = 0
        self._utterances_emitted = 0
        self._utterances_discarded = 0
        self._resets = 0
        self._last_rms = 0
        self._heartbeat_monotonic = time.monotonic()
        self._last_reset_reason: str | None = None

    def push(self, pcm_frame: bytes) -> tuple[bytes, ...]:
        if not isinstance(pcm_frame, bytes) or len(pcm_frame) != self.FRAME_BYTES:
            raise ValueError(
                f"pcm_frame must be one {self.FRAME_MS} ms PCM s16le frame"
            )
        rms = _pcm_s16le_rms(pcm_frame)
        voiced = rms >= self.rms_threshold
        emitted: list[bytes] = []
        with self._lock:
            self._frames_received += 1
            self._last_rms = rms
            self._heartbeat_monotonic = time.monotonic()
            if self._active_frames is None:
                self._push_inactive_locked(pcm_frame, voiced)
            else:
                self._push_active_locked(pcm_frame, voiced)

            if self._active_frames is not None:
                if self._trailing_silence_ms >= self.end_silence_ms:
                    result = self._finish_locked(keep_silence_tail=True)
                    if result is not None:
                        emitted.append(result)
                elif self._active_duration_ms >= self.maximum_utterance_ms:
                    result = self._finish_locked(keep_silence_tail=False)
                    if result is not None:
                        emitted.append(result)
        return tuple(emitted)

    def reset(self, *, reason: str) -> None:
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must not be empty")
        with self._lock:
            self._clear_buffers_locked()
            self._resets += 1
            self._last_reset_reason = reason
            self._heartbeat_monotonic = time.monotonic()

    def snapshot(self) -> EnergyVadSnapshot:
        with self._lock:
            return EnergyVadSnapshot(
                active=self._active_frames is not None,
                frames_received=self._frames_received,
                utterances_emitted=self._utterances_emitted,
                utterances_discarded=self._utterances_discarded,
                resets=self._resets,
                last_rms=self._last_rms,
                heartbeat_monotonic=self._heartbeat_monotonic,
                last_reset_reason=self._last_reset_reason,
            )

    def _push_inactive_locked(self, frame: bytes, voiced: bool) -> None:
        if not voiced:
            self._candidate.clear()
            self._candidate_ms = 0
            self._pre_roll.append(frame)
            return
        self._candidate.append(frame)
        self._candidate_ms += self.FRAME_MS
        if self._candidate_ms < self.start_trigger_ms:
            return
        self._active_frames = [*self._pre_roll, *self._candidate]
        self._active_voiced_ms = self._candidate_ms
        self._active_duration_ms = len(self._active_frames) * self.FRAME_MS
        self._trailing_silence_ms = 0
        self._pre_roll.clear()
        self._candidate.clear()
        self._candidate_ms = 0

    def _push_active_locked(self, frame: bytes, voiced: bool) -> None:
        assert self._active_frames is not None
        self._active_frames.append(frame)
        self._active_duration_ms += self.FRAME_MS
        if voiced:
            self._active_voiced_ms += self.FRAME_MS
            self._trailing_silence_ms = 0
        else:
            self._trailing_silence_ms += self.FRAME_MS

    def _finish_locked(self, *, keep_silence_tail: bool) -> bytes | None:
        assert self._active_frames is not None
        frames = self._active_frames
        voiced_ms = self._active_voiced_ms
        tail = []
        if keep_silence_tail and self._pre_roll.maxlen:
            tail = frames[-self._pre_roll.maxlen :]
        self._clear_buffers_locked()
        self._pre_roll.extend(tail)
        if voiced_ms < self.minimum_speech_ms:
            self._utterances_discarded += 1
            return None
        self._utterances_emitted += 1
        return b"".join(frames)

    def _clear_buffers_locked(self) -> None:
        self._pre_roll.clear()
        self._candidate.clear()
        self._candidate_ms = 0
        self._active_frames = None
        self._active_voiced_ms = 0
        self._active_duration_ms = 0
        self._trailing_silence_ms = 0


def _pcm_s16le_rms(frame: bytes) -> int:
    squares = sum(
        sample * sample for (sample,) in struct.iter_unpack("<h", frame)
    )
    return math.isqrt(squares // (len(frame) // 2))
