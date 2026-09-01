from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Callable

from .follow_control import FollowController, FollowDecision, FollowState
from .follow_receiver import LatestObservationStore


class LatestDecisionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._decision = FollowDecision.neutral(FollowState.LOST)

    def publish(self, decision: FollowDecision) -> None:
        with self._lock:
            self._decision = decision

    def snapshot(self) -> FollowDecision:
        with self._lock:
            return self._decision


class FollowPromptRouter:
    """Translate fast control states into infrequent ARDY motion-mode changes."""

    def __init__(
        self,
        *,
        set_prompt: Callable[[str], None],
        initial_prompt: str,
        walking_prompt: str,
        idle_prompt: str,
    ) -> None:
        if not initial_prompt.strip() or not walking_prompt.strip() or not idle_prompt.strip():
            raise ValueError("follow prompts must not be empty")
        self.set_prompt = set_prompt
        self.walking_prompt = walking_prompt
        self.idle_prompt = idle_prompt
        self._active_prompt = initial_prompt
        self._lock = threading.Lock()

    def __call__(self, _old_state: FollowState, new_state: FollowState) -> None:
        desired = (
            self.walking_prompt
            if new_state in (FollowState.RELOCATE, FollowState.ALIGN, FollowState.FOLLOW)
            else self.idle_prompt
        )
        with self._lock:
            if desired == self._active_prompt:
                return
            self.set_prompt(desired)
            self._active_prompt = desired


@dataclass(frozen=True)
class FollowLoopStatus:
    running: bool = False
    ticks: int = 0
    state: FollowState = FollowState.LOST
    last_error: str | None = None
    heartbeat_monotonic: float = 0.0


class FollowDecisionLoop:
    """Evaluate the newest observation at a fixed rate, independently of ARDY generation."""

    def __init__(
        self,
        *,
        observation_store: LatestObservationStore,
        controller: FollowController,
        decision_store: LatestDecisionStore,
        on_decision: Callable[[FollowDecision], None] | None = None,
        on_state_change: Callable[[FollowState, FollowState], None] | None = None,
        tick_hz: float = 20.0,
    ) -> None:
        if tick_hz <= 0:
            raise ValueError("tick_hz must be positive")
        self.observation_store = observation_store
        self.controller = controller
        self.decision_store = decision_store
        self.on_decision = on_decision
        self.on_state_change = on_state_change
        self.tick_hz = float(tick_hz)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = FollowLoopStatus()
        self._stopped_neutralized = False

    @property
    def status(self) -> FollowLoopStatus:
        with self._lock:
            return replace(self._status)

    def tick(self, *, now_monotonic: float | None = None) -> FollowDecision:
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        old_state = self.decision_store.snapshot().state
        decision = self.controller.step(
            self.observation_store.snapshot(),
            now_monotonic=float(now_monotonic),
        )
        self.decision_store.publish(decision)
        if self.on_decision is not None:
            self.on_decision(decision)
        with self._lock:
            self._stopped_neutralized = False
            self._status = replace(
                self._status,
                ticks=self._status.ticks + 1,
                state=decision.state,
                heartbeat_monotonic=float(now_monotonic),
            )
        if self.on_state_change is not None and old_state != decision.state:
            self.on_state_change(old_state, decision.state)
        return decision

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("follow decision loop is already started")
            self._stop_event.clear()
            self._stopped_neutralized = False
            self._status = replace(
                self._status,
                running=True,
                last_error=None,
                heartbeat_monotonic=time.monotonic(),
            )
            self._thread = threading.Thread(
                target=self._run,
                name="follow-decision-loop",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._lock:
            self._thread = None
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=time.monotonic(),
            )
        self._emit_stop_neutral()

    def _run(self) -> None:
        interval = 1.0 / self.tick_hz
        next_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                self.tick(now_monotonic=time.monotonic())
                next_tick += interval
                self._stop_event.wait(max(0.0, next_tick - time.monotonic()))
        except BaseException as exc:
            with self._lock:
                self._status = replace(self._status, last_error=str(exc))
            self._stop_event.set()
        finally:
            self._emit_stop_neutral()
            with self._lock:
                self._status = replace(
                    self._status,
                    running=False,
                    heartbeat_monotonic=time.monotonic(),
                )

    def _emit_stop_neutral(self) -> None:
        with self._lock:
            if self._stopped_neutralized:
                return
            self._stopped_neutralized = True
        neutral = FollowDecision.neutral(FollowState.LOST)
        self.decision_store.publish(neutral)
        if self.on_decision is None:
            return
        try:
            self.on_decision(neutral)
        except BaseException as exc:
            with self._lock:
                self._status = replace(self._status, last_error=str(exc))
