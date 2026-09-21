from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from .live_session import PromptPlaybackStarted


class CuePhase(StrEnum):
    PREPARING = "PREPARING"
    PLAYING = "PLAYING"
    RETURNING_IDLE = "RETURNING_IDLE"


class CuePlaybackObservation(StrEnum):
    IGNORED = "IGNORED"
    CUE_STARTED = "CUE_STARTED"
    IDLE_STARTED = "IDLE_STARTED"


@dataclass(slots=True)
class MotionCue:
    action_id: str
    turn_id: str
    prompt: str
    duration_seconds: float
    phase: CuePhase = CuePhase.PREPARING
    prompt_revision: int | None = None
    prepare_deadline_monotonic: float | None = None
    started_monotonic: float | None = None
    playback_deadline_monotonic: float | None = None
    idle_revision: int | None = None
    idle_deadline_monotonic: float | None = None
    completion_driven: bool = False

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must not be empty")
        if not self.turn_id:
            raise ValueError("turn_id must not be empty")
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive")

    def bind_prompt_revision(
        self,
        revision: int,
        *,
        now: float,
        timeout_seconds: float,
    ) -> None:
        _validate_revision(revision)
        _validate_clock(now, "now")
        _validate_timeout(timeout_seconds)
        self.phase = CuePhase.PREPARING
        self.prompt_revision = revision
        self.prepare_deadline_monotonic = now + timeout_seconds
        self.started_monotonic = None
        self.playback_deadline_monotonic = None
        self.idle_revision = None
        self.idle_deadline_monotonic = None

    def begin_return_to_idle(
        self,
        revision: int,
        *,
        now: float,
        timeout_seconds: float,
    ) -> None:
        if self.phase is not CuePhase.PLAYING:
            raise RuntimeError("cue must be playing before returning to idle")
        _validate_revision(revision)
        _validate_clock(now, "now")
        _validate_timeout(timeout_seconds)
        self.phase = CuePhase.RETURNING_IDLE
        self.idle_revision = revision
        self.idle_deadline_monotonic = now + timeout_seconds

    def observe_playback(
        self,
        event: PromptPlaybackStarted,
    ) -> CuePlaybackObservation:
        if (
            self.phase is CuePhase.PREPARING
            and event.revision == self.prompt_revision
        ):
            self.phase = CuePhase.PLAYING
            self.prepare_deadline_monotonic = None
            self.started_monotonic = event.started_monotonic
            self.playback_deadline_monotonic = (
                event.started_monotonic + self.duration_seconds
            )
            return CuePlaybackObservation.CUE_STARTED
        if (
            self.phase is CuePhase.RETURNING_IDLE
            and event.revision == self.idle_revision
        ):
            return CuePlaybackObservation.IDLE_STARTED
        return CuePlaybackObservation.IGNORED

    def timeout_kind(self, now: float) -> str | None:
        _validate_clock(now, "now")
        if (
            self.phase is CuePhase.PREPARING
            and self.prepare_deadline_monotonic is not None
            and now >= self.prepare_deadline_monotonic
        ):
            return "cue playback"
        if (
            self.phase is CuePhase.RETURNING_IDLE
            and self.idle_deadline_monotonic is not None
            and now >= self.idle_deadline_monotonic
        ):
            return "idle playback"
        return None

    def playback_expired(self, now: float) -> bool:
        _validate_clock(now, "now")
        return (
            self.phase is CuePhase.PLAYING
            and not self.completion_driven
            and self.playback_deadline_monotonic is not None
            and now >= self.playback_deadline_monotonic
        )

    def next_deadline(self) -> float | None:
        if self.phase is CuePhase.PREPARING:
            return self.prepare_deadline_monotonic
        if self.phase is CuePhase.PLAYING:
            return None if self.completion_driven else self.playback_deadline_monotonic
        return self.idle_deadline_monotonic


def _validate_revision(revision: int) -> None:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
        raise ValueError("prompt revision must be a positive integer")


def _validate_clock(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_timeout(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
