from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import threading
import time
from typing import Protocol

from .dialogue_turns import TurnToken
from .energy_vad import EnergyVadSegmenter
from .interaction_runtime import TurnOutcome, TurnOutcomeStatus
from .stream_protocol import AudioChunkMessage


class SpeechToText(Protocol):
    def transcribe_pcm(self, pcm_s16le: bytes) -> str: ...


class ReservedTurnRuntime(Protocol):
    def begin_utterance(self) -> TurnToken: ...

    def handle_reserved_utterance(
        self,
        token: TurnToken,
        *,
        transcript: str,
        screenshot: bytes | None,
    ) -> TurnOutcome: ...

    def fail_reserved_utterance(
        self,
        token: TurnToken,
        *,
        error: str,
    ) -> TurnOutcome: ...


@dataclass(frozen=True, slots=True)
class AudioPipelineSnapshot:
    closed: bool = False
    active_jobs: int = 0
    utterances_detected: int = 0
    utterances_completed: int = 0
    utterances_failed: int = 0
    utterances_superseded: int = 0
    audio_discontinuities: int = 0
    last_transcript: str | None = None
    last_outcome: str | None = None
    last_utterance_audio_ms: int | None = None
    last_stt_latency_ms: float | None = None
    last_turn_handling_latency_ms: float | None = None
    last_pipeline_latency_ms: float | None = None
    heartbeat_monotonic: float = 0.0
    last_error: str | None = None


class AudioInteractionPipeline:
    """Turns ordered PCM frames into independently fenced interaction jobs."""

    def __init__(
        self,
        *,
        vad: EnergyVadSegmenter,
        stt: SpeechToText,
        runtime: ReservedTurnRuntime,
        monotonic: Callable[[], float] = time.monotonic,
        on_input_activity: Callable[[], None] | None = None,
    ) -> None:
        self.vad = vad
        self.stt = stt
        self.runtime = runtime
        self._monotonic = monotonic
        self._on_input_activity = on_input_activity
        self._ingest_lock = threading.Lock()
        self._condition = threading.Condition(threading.RLock())
        self._session_id: str | None = None
        self._last_sequence: int | None = None
        self._next_job_id = 1
        self._latest_admitted_job_id = 0
        self._latest_reported_job_id = 0
        self._jobs: dict[int, threading.Thread] = {}
        self._status = AudioPipelineSnapshot(heartbeat_monotonic=self._monotonic())

    def accept_audio(self, message: AudioChunkMessage) -> None:
        if not isinstance(message, AudioChunkMessage):
            raise TypeError("message must be AudioChunkMessage")
        with self._ingest_lock:
            with self._condition:
                if self._status.closed:
                    raise RuntimeError("audio interaction pipeline is closed")
                session_changed = message.session_id != self._session_id
                discontinuous = (
                    not session_changed
                    and self._last_sequence is not None
                    and message.sequence != self._last_sequence + 1
                )
                if session_changed:
                    self.vad.reset(reason="audio session changed")
                    self._session_id = message.session_id
                    self._last_sequence = None
                elif discontinuous:
                    self.vad.reset(reason="audio discontinuity")
                    self._status = replace(
                        self._status,
                        audio_discontinuities=(self._status.audio_discontinuities + 1),
                    )
                self._last_sequence = message.sequence
                self._touch_locked()
            utterances = self.vad.push(message.pcm)
            observed_vad = self.vad.snapshot()
            if self._on_input_activity is not None and (observed_vad.active or observed_vad.last_rms >= self.vad.rms_threshold):
                self._on_input_activity()
            for pcm in utterances:
                self._launch(pcm)

    def tick_autonomy(self):
        # Admission uses the same ingest lock as VAD and turn reservation. Speech
        # cannot begin between checking quiet input and submitting an idle action.
        with self._ingest_lock:
            with self._condition:
                quiet = not self._status.closed and not self._jobs and not self.vad.snapshot().active
            self.runtime.tick_autonomy(input_quiet=quiet)

    def wait_until_idle(self, *, timeout: float | None = None) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: not self._jobs, timeout=timeout)

    def close(self, *, timeout: float = 5.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._condition:
            self._status = replace(
                self._status,
                closed=True,
                heartbeat_monotonic=self._monotonic(),
            )
            threads = tuple(self._jobs.values())
        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        with self._condition:
            if self._jobs:
                raise TimeoutError("audio interaction jobs did not stop")

    def snapshot(self) -> AudioPipelineSnapshot:
        with self._condition:
            return replace(self._status, active_jobs=len(self._jobs))

    def _launch(self, pcm: bytes) -> None:
        with self._condition:
            job_id = self._next_job_id
            self._next_job_id += 1
            token = self.runtime.begin_utterance()
            self._latest_admitted_job_id = job_id
            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, pcm, token),
                name=f"interaction-utterance-{job_id}",
                daemon=True,
            )
            self._jobs[job_id] = thread
            self._status = replace(
                self._status,
                active_jobs=len(self._jobs),
                utterances_detected=self._status.utterances_detected + 1,
                heartbeat_monotonic=self._monotonic(),
            )
            thread.start()

    def _run_job(self, job_id: int, pcm: bytes, token: TurnToken) -> None:
        job_started_at = self._monotonic()
        outcome: TurnOutcome | None = None
        status: TurnOutcomeStatus | None = None
        transcript: str | None = None
        error: str | None = None
        stt_started_at = self._monotonic()
        stt_latency_ms: float | None = None
        turn_handling_latency_ms: float | None = None
        try:
            try:
                transcript = self.stt.transcribe_pcm(pcm)
            except Exception as exc:
                error = f"STT failed: {type(exc).__name__}: {exc}"
                outcome = self._report_failure(token, error)
                status = outcome.status
            finally:
                stt_latency_ms = (self._monotonic() - stt_started_at) * 1000.0
            if transcript is not None:
                try:
                    token = self._admit_transcript(job_id, token)
                    if token is None:
                        status = TurnOutcomeStatus.SUPERSEDED
                    else:
                        handling_started_at = self._monotonic()
                        outcome = self.runtime.handle_reserved_utterance(
                            token,
                            transcript=transcript,
                            screenshot=None,
                        )
                        turn_handling_latency_ms = (
                            self._monotonic() - handling_started_at
                        ) * 1000.0
                    if outcome is not None:
                        status = outcome.status
                except Exception as exc:
                    error = f"turn handling failed: {type(exc).__name__}: {exc}"
                    if token is None:
                        status = TurnOutcomeStatus.FAILED
                    else:
                        outcome = self._report_failure(token, error)
                        status = outcome.status
        finally:
            finished_at = self._monotonic()
            with self._condition:
                self._jobs.pop(job_id, None)
                latest_fields: dict[str, object] = {}
                if job_id > self._latest_reported_job_id:
                    self._latest_reported_job_id = job_id
                    latest_fields = {
                        "last_transcript": transcript,
                        "last_outcome": (None if status is None else status.value),
                        "last_utterance_audio_ms": (
                            len(pcm) * 1000 // (EnergyVadSegmenter.SAMPLE_RATE * 2)
                        ),
                        "last_stt_latency_ms": stt_latency_ms,
                        "last_turn_handling_latency_ms": (turn_handling_latency_ms),
                        "last_pipeline_latency_ms": (finished_at - job_started_at)
                        * 1000.0,
                        "last_error": error,
                    }
                self._status = replace(
                    self._status,
                    active_jobs=len(self._jobs),
                    utterances_completed=(
                        self._status.utterances_completed
                        + (
                            1
                            if status
                            in {
                                TurnOutcomeStatus.APPLIED,
                                TurnOutcomeStatus.FAST_STOP_REQUESTED,
                            }
                            else 0
                        )
                    ),
                    utterances_failed=(
                        self._status.utterances_failed
                        + (1 if status == TurnOutcomeStatus.FAILED else 0)
                    ),
                    utterances_superseded=(
                        self._status.utterances_superseded
                        + (1 if status == TurnOutcomeStatus.SUPERSEDED else 0)
                    ),
                    heartbeat_monotonic=finished_at,
                    **latest_fields,
                )
                self._condition.notify_all()

    def _admit_transcript(self, job_id: int, token: TurnToken) -> TurnToken | None:
        with self._condition:
            if job_id != self._latest_admitted_job_id:
                return None
            return token

    def _report_failure(self, token: TurnToken, error: str) -> TurnOutcome:
        try:
            return self.runtime.fail_reserved_utterance(token, error=error)
        except Exception as exc:
            # A failed status report must still terminate the local job and be
            # observable; otherwise wait_until_idle can hang behind a dead worker.
            combined = f"{error}; failure reporting failed: {type(exc).__name__}: {exc}"
            return TurnOutcome(
                turn_id=token.turn_id,
                turn_version=token.version,
                status=TurnOutcomeStatus.FAILED,
                error=combined,
            )

    def _touch_locked(self) -> None:
        self._status = replace(
            self._status,
            heartbeat_monotonic=self._monotonic(),
        )
