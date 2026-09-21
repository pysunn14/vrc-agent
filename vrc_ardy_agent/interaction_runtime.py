from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from enum import StrEnum
import threading
import time
from types import MappingProxyType
from typing import Protocol

from .action_contracts import (
    ActionBundle,
    ActionType,
    ArdyMotionAction,
    PresetMotionAction,
    ControlResource,
    PlanValidationError,
    validate_action_bundle,
)
from .character_supervisor import CharacterSupervisor, HaltResult, SupervisorSnapshot
from .dialogue_turns import DialogueTurnManager, DialogueTurnSnapshot, TurnToken
from .idle_behavior_policy import IdleBehaviorPolicy
from .fast_stop import is_fast_stop
from .observation_loop import ObservationLoop, SupersededObservation
from .motion_stream import MotionDirectorSnapshot, MotionHaltResult, MotionDirectorState


class BodyActivity(StrEnum):
    IDLE = "IDLE"
    ARDY_MOTION = "ARDY_MOTION"


class SpeechActivity(StrEnum):
    IDLE = "IDLE"
    SPEAKING = "SPEAKING"


class TurnOutcomeStatus(StrEnum):
    APPLIED = "APPLIED"
    PLANNED = "PLANNED"
    FAST_STOP_REQUESTED = "FAST_STOP_REQUESTED"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class BrainRequest:
    turn_id: str
    turn_version: int
    transcript: str
    screenshot: bytes | None
    body: BodyActivity
    speech: SpeechActivity
    current_action_ids: tuple[str, ...]
    last_error: str | None
    round_index: int = 0
    observations: tuple = ()
    conversation: tuple = ()
    execution_feedback: dict | None = None
    deadline_monotonic: float | None = None
    call_id: str | None = None
    record_model_metadata: object | None = None
    available_motions: tuple[dict, ...] = ()


class Brain(Protocol):
    def propose(self, request: BrainRequest) -> object: ...


class MotionController(Protocol):
    def submit_cue(self, action: ArdyMotionAction | PresetMotionAction, *, turn_id: str) -> str: ...

    def halt(self, *, reason: str) -> MotionHaltResult: ...

    def snapshot(self) -> MotionDirectorSnapshot: ...


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    turn_id: str
    turn_version: int
    status: TurnOutcomeStatus
    action_ids: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    sequence: int
    event_type: str
    turn_id: str | None
    detail: str | None
    monotonic: float


@dataclass(frozen=True, slots=True)
class InteractionSnapshot:
    turns: DialogueTurnSnapshot
    supervisor: SupervisorSnapshot
    motion: MotionDirectorSnapshot
    thinking_turn_ids: tuple[str, ...]
    recent_events: tuple[RuntimeEvent, ...]
    heartbeat_monotonic: float
    last_error: str | None


class InteractionRuntime:
    """Connects completed utterances to validated, fenced action execution."""

    def __init__(
        self,
        *,
        brain: Brain,
        supervisor: CharacterSupervisor,
        motion_director: MotionController,
        turn_manager: DialogueTurnManager | None = None,
        event_history_limit: int = 100,
        trace_factory=None,
        plan_recorder=None,
        behaviors=None,
        autonomy=None,
    ) -> None:
        if event_history_limit <= 0:
            raise ValueError("event_history_limit must be positive")
        self.behaviors = MappingProxyType(dict(behaviors or {}))
        self._activity_lock = threading.RLock()
        self._autonomy_output = None
        self._idle_policy = IdleBehaviorPolicy(autonomy or {
            'enabled': False, 'interval_seconds': 30, 'behaviors': []}, self.behaviors)
        self._trace_factory = trace_factory
        self._plan_recorder = plan_recorder
        self._observation = ObservationLoop()
        self._brain = brain
        self._supervisor = supervisor
        self._motion = motion_director
        self._turns = turn_manager or DialogueTurnManager()
        self._lock = threading.Lock()
        self._thinking_turn_ids: set[str] = set()
        self._active_turns: set[TurnToken] = set()
        self._events: deque[RuntimeEvent] = deque(maxlen=event_history_limit)
        self._event_sequence = 0
        self._last_error: str | None = None
        self._heartbeat_monotonic = time.monotonic()

    def tick_autonomy(self, *, input_quiet, now=None):
        with self._activity_lock:
            motion = self._motion.snapshot()
            identity = (motion.output_session_id, motion.lease_token)
            if identity != self._autonomy_output:
                self._idle_policy.interrupt()
                self._autonomy_output = identity
            speech = self._supervisor.snapshot()
            turns = self._turns.snapshot()
            with self._lock:
                thinking = bool(self._thinking_turn_ids or self._active_turns)
            eligible = (input_quiet and not thinking and turns.pending_turn_id is None
                        and not speech.current_actions and motion.running
                        and motion.state == MotionDirectorState.IDLE
                        and motion.output_session_id is not None and motion.last_error is None)
            self._idle_policy.tick(now=time.monotonic() if now is None else now, eligible=eligible,
                submit=lambda key: self._motion.submit_idle_cue(PresetMotionAction(key), turn_id='autonomy'))

    def note_input_activity(self):
        # An utterance may start and finish between heartbeat ticks. Reset from
        # input events too, so "quiet for N seconds" means continuous quiet.
        with self._activity_lock:
            self._idle_policy.interrupt()

    def autonomy_snapshot(self):
        with self._activity_lock:
            return self._idle_policy.snapshot()

    def configure_observation(self, provider, **limits) -> None:
        """Wire the capture service at startup, before accepting utterances."""
        self._observation = ObservationLoop(provider, **limits)

    @property
    def observation_service(self):
        return self._observation

    def handle_utterance(
        self,
        *,
        transcript: str,
        screenshot: bytes | None,
    ) -> TurnOutcome:
        token = self.begin_utterance()
        return self.handle_reserved_utterance(
            token,
            transcript=transcript,
            screenshot=screenshot,
        )

    def begin_utterance(self) -> TurnToken:
        """Reserve freshness as soon as VAD closes an utterance."""
        with self._activity_lock:
            self._idle_policy.interrupt()
            token = self._turns.begin_turn()
            self._event("turn.started", token=token)
            return token

    def handle_reserved_utterance(
        self, token: TurnToken, *, transcript: str, screenshot: bytes | None = None
    ) -> TurnOutcome:
        if not isinstance(token, TurnToken):
            raise TypeError("token must be a TurnToken")
        with self._lock:
            duplicate = token in self._active_turns
            if not duplicate:
                self._active_turns.add(token)
        if duplicate:
            return self._superseded(token)
        trace = self._trace_factory(token) if self._trace_factory else None
        try:
            result = self._handle_reserved_utterance(
                token, transcript=transcript, screenshot=screenshot, trace=trace
            )
            if trace: trace.finish(result.status.value, result.error)
            return result
        finally:
            with self._lock:
                self._active_turns.discard(token)

    def _handle_reserved_utterance(
        self,
        token: TurnToken,
        *,
        transcript: str,
        screenshot: bytes | None,
        trace=None,
    ) -> TurnOutcome:
        if not isinstance(token, TurnToken):
            raise TypeError("token must be a TurnToken")
        if not isinstance(transcript, str):
            raise TypeError("transcript must be a string")
        transcript = transcript.strip()
        self._event("turn.transcribed", token=token, detail=transcript)

        if not self._turns.is_current(token):
            return self._superseded(token)

        if not transcript:
            self._turns.claim_current(token)
            return self._failed(token, "transcript is empty")

        if is_fast_stop(transcript):
            halt_holder: list[HaltResult] = []
            with self._activity_lock:
                accepted = self._turns.apply_if_current(
                    token,
                    lambda: halt_holder.append(self._halt_components(reason="fast_stop")),
                )
            if not accepted:
                return self._superseded(token)
            self._event("halt.requested", token=token, detail="fast_stop")
            self._clear_error()
            return TurnOutcome(
                turn_id=token.turn_id,
                turn_version=token.version,
                status=TurnOutcomeStatus.FAST_STOP_REQUESTED,
            )

        self._set_thinking(token, True)
        self._event("brain.requested", token=token)
        try:
            response = self._observation.run(
                request_factory=lambda: self._build_brain_request(
                    token, transcript, None
                ),
                brain=self._brain,
                current=lambda: self._turns.is_current(token),
                trace=trace,
                event=lambda kind, detail: self._event(
                    kind, token=token, detail=detail
                ),
            )
        except SupersededObservation:
            return self._superseded(token)
        except Exception as exc:
            if not self._turns.claim_current(token):
                return self._superseded(token)
            return self._failed(
                token, f"brain request failed: {type(exc).__name__}: {exc}"
            )
        finally:
            self._set_thinking(token, False)

        if not self._turns.is_current(token):
            return self._superseded(token)

        if trace: trace.stage("validation")
        validation_started_ns = time.perf_counter_ns()
        try:
            bundle = validate_action_bundle(response, behaviors=self.behaviors)
        except PlanValidationError as exc:
            if not self._turns.claim_current(token):
                return self._superseded(token)
            self._event("plan.rejected", token=token, detail=str(exc))
            return self._failed(token, str(exc))

        if trace: trace.validated(response, validation_started_ns)
        return self._dispatch_bundle(token, bundle)

    def fail_reserved_utterance(
        self,
        token: TurnToken,
        *,
        error: str,
    ) -> TurnOutcome:
        if not isinstance(token, TurnToken):
            raise TypeError("token must be a TurnToken")
        error = error.strip()
        if not error:
            raise ValueError("error must not be empty")
        if not self._turns.claim_current(token):
            return self._superseded(token)
        return self._failed(token, error)

    def apply_action_payload(
        self,
        payload: object,
        *,
        source: str,
    ) -> TurnOutcome:
        source = source.strip()
        if not source:
            raise ValueError("source must not be empty")
        token = self.begin_utterance()
        self._event("turn.source", token=token, detail=source)
        try:
            bundle = validate_action_bundle(payload, behaviors=self.behaviors)
        except PlanValidationError as exc:
            self._turns.claim_current(token)
            self._event("plan.rejected", token=token, detail=str(exc))
            return self._failed(token, str(exc))
        return self._dispatch_bundle(token, bundle)

    def _dispatch_bundle(
        self,
        token: TurnToken,
        bundle: ActionBundle,
    ) -> TurnOutcome:
        action_ids: list[str] = []

        def dispatch() -> None:
            # Evaluation preserves validation and turn ownership, but never gives
            # device executors a command or claims physical action completion.
            if self._plan_recorder is not None:
                self._plan_recorder(bundle)
                return
            for action in bundle.actions:
                if action.action_type == ActionType.SAY:
                    application = self._supervisor.apply_bundle(
                        ActionBundle(actions=(action,)),
                        turn_id=token.turn_id,
                    )
                    action_ids.extend(application.action_ids)
                    continue
                if not isinstance(action, (ArdyMotionAction, PresetMotionAction)):
                    raise TypeError("motion must be a generated or registered action")
                action_ids.append(
                    self._motion.submit_cue(action, turn_id=token.turn_id)
                )

        try:
            with self._activity_lock:
                accepted = self._turns.apply_if_current(token, dispatch)
        except Exception as exc:
            return self._failed(
                token,
                f"action dispatch failed: {type(exc).__name__}: {exc}",
            )
        if not accepted:
            return self._superseded(token)

        self._event(
            "plan.recorded" if self._plan_recorder is not None else "plan.applied",
            token=token,
            detail=",".join(action_ids),
        )
        self._clear_error()
        return TurnOutcome(
            turn_id=token.turn_id,
            turn_version=token.version,
            status=TurnOutcomeStatus.PLANNED if self._plan_recorder is not None else TurnOutcomeStatus.APPLIED,
            action_ids=tuple(action_ids),
        )

    def halt_all(self, *, reason: str) -> HaltResult:
        with self._activity_lock:
            self._turns.invalidate_pending(reason=reason)
            result = self._halt_components(reason=reason)
            self._event("runtime.halted", detail=reason)
            return result

    def snapshot(self) -> InteractionSnapshot:
        with self._lock:
            return InteractionSnapshot(
                turns=self._turns.snapshot(),
                supervisor=self._supervisor.snapshot(),
                motion=self._motion.snapshot(),
                thinking_turn_ids=tuple(sorted(self._thinking_turn_ids)),
                recent_events=tuple(self._events),
                heartbeat_monotonic=self._heartbeat_monotonic,
                last_error=self._last_error,
            )

    def _build_brain_request(
        self,
        token: TurnToken,
        transcript: str,
        screenshot: bytes | None,
    ) -> BrainRequest:
        supervisor = self._supervisor.snapshot()
        motion = self._motion.snapshot()
        speech = supervisor.current(ControlResource.VOICE_OUTPUT)
        current_ids = [action.action_id for action in supervisor.current_actions]
        if motion.active_cue_id is not None:
            current_ids.append(motion.active_cue_id)
        return BrainRequest(
            turn_id=token.turn_id,
            turn_version=token.version,
            transcript=transcript,
            screenshot=screenshot,
            body=(
                BodyActivity.ARDY_MOTION
                if motion.active_cue_id is not None
                else BodyActivity.IDLE
            ),
            speech=(
                SpeechActivity.SPEAKING if speech is not None else SpeechActivity.IDLE
            ),
            current_action_ids=tuple(current_ids),
            last_error=self.snapshot().last_error,
            available_motions=tuple(spec.describe() for spec in self.behaviors.values()),
            execution_feedback={
                "motion": asdict(motion),
                "recent_actions": [
                    asdict(action) for action in supervisor.recent_actions
                ],
                "execution_errors": list(supervisor.last_errors),
                "tracking": {
                    "status": "unknown",
                    "reason": "no SteamVR tracking telemetry in this path",
                },
            },
        )

    def _halt_components(self, *, reason: str) -> HaltResult:
        speech_result = self._supervisor.halt_all(reason=reason)
        motion_result = self._motion.halt(reason=reason)
        tokens = dict(speech_result.resource_tokens)
        tokens[ControlResource.FULL_BODY_POSE] = motion_result.lease_token
        return HaltResult(
            reason=reason,
            resource_tokens=tuple(
                sorted(tokens.items(), key=lambda item: item[0].value)
            ),
        )

    def _failed(self, token: TurnToken, error: str) -> TurnOutcome:
        with self._lock:
            self._last_error = error
            self._touch_locked()
        self._event("turn.failed", token=token, detail=error)
        return TurnOutcome(
            turn_id=token.turn_id,
            turn_version=token.version,
            status=TurnOutcomeStatus.FAILED,
            error=error,
        )

    def _superseded(self, token: TurnToken) -> TurnOutcome:
        self._event("brain.response_dropped", token=token, detail="superseded")
        return TurnOutcome(
            turn_id=token.turn_id,
            turn_version=token.version,
            status=TurnOutcomeStatus.SUPERSEDED,
        )

    def _set_thinking(self, token: TurnToken, value: bool) -> None:
        with self._lock:
            if value:
                self._thinking_turn_ids.add(token.turn_id)
            else:
                self._thinking_turn_ids.discard(token.turn_id)
            self._touch_locked()

    def _clear_error(self) -> None:
        with self._lock:
            self._last_error = None
            self._touch_locked()

    def _event(
        self,
        event_type: str,
        *,
        token: TurnToken | None = None,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            self._event_sequence += 1
            now = time.monotonic()
            self._events.append(
                RuntimeEvent(
                    sequence=self._event_sequence,
                    event_type=event_type,
                    turn_id=None if token is None else token.turn_id,
                    detail=detail,
                    monotonic=now,
                )
            )
            self._heartbeat_monotonic = now

    def _touch_locked(self) -> None:
        self._heartbeat_monotonic = time.monotonic()
