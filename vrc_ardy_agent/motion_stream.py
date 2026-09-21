from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .action_contracts import ResourceLease


CALIBRATED_IDLE = "idle:calibrated"
# ARDY may briefly spend more than one frame interval generating the next
# horizon. Keep the last valid pose alive across that expected gap while still
# neutralizing promptly when the writer actually disappears.
DEFAULT_POSE_TTL_MS = 2000


class MotionDirectorState(StrEnum):
    STOPPED = "STOPPED"
    WAITING_OUTPUT = "WAITING_OUTPUT"
    STARTING = "STARTING"
    IDLE = "IDLE"
    PREPARING_CUE = "PREPARING_CUE"
    CUE = "CUE"
    RETURNING_IDLE = "RETURNING_IDLE"
    STOPPING = "STOPPING"
    HALTED = "HALTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class MotionDirectorSnapshot:
    state: MotionDirectorState = MotionDirectorState.STOPPED
    running: bool = False
    output_session_id: str | None = None
    stream_action_id: str | None = None
    lease_token: int = 0
    active_prompt: str | None = None
    active_cue_id: str | None = None
    active_turn_id: str | None = None
    cue_prompt_revision: int | None = None
    cue_prepare_deadline_monotonic: float | None = None
    cue_started_monotonic: float | None = None
    cue_deadline_monotonic: float | None = None
    idle_prompt_revision: int | None = None
    playing_prompt: str | None = None
    playing_prompt_revision: int = 0
    played_frames: int = 0
    generated_chunks: int = 0
    underruns: int = 0
    last_generation_seconds: float | None = None
    heartbeat_monotonic: float = 0.0
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class MotionHaltResult:
    reason: str
    lease_token: int


class DirectorPoseSink:
    """Send one persistent pose stream under a session-scoped lease."""

    def __init__(
        self,
        *,
        output: Any,
        lease: ResourceLease,
        ttl_ms: int,
    ) -> None:
        self._output = output
        self._lease = lease
        self._ttl_ms = ttl_ms
        self._sequence = 0

    def send(self, frame: object) -> None:
        sequence = self._sequence
        self._output.send_pose(
            self._lease,
            frame,
            sequence=sequence,
            ttl_ms=self._ttl_ms,
        )
        self._sequence += 1

    def neutralize_inputs(self) -> None:
        # The director fences and neutralizes the complete output resource.
        return

    def close(self) -> None:
        # The shared transport belongs to the companion application.
        return
