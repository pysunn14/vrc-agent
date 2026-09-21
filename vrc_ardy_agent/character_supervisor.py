from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading
import time
from typing import Protocol
import uuid

from .action_contracts import (
    Action,
    ActionBundle,
    ActionState,
    ActionType,
    ControlResource,
    ExecutionCommand,
    ResourceLease,
)
from .control_resources import ControlResourceManager


_ACTION_RESOURCES = {
    ActionType.SAY: ControlResource.VOICE_OUTPUT,
    ActionType.ARDY_MOTION: ControlResource.FULL_BODY_POSE,
    ActionType.MOTION: ControlResource.FULL_BODY_POSE,
}


class ActionExecutor(Protocol):
    def run(
        self,
        command: ExecutionCommand,
        cancel_event: threading.Event,
        mark_running: Callable[[], None],
    ) -> None: ...


class OutputAuthority(Protocol):
    def authorize(self, lease: ResourceLease) -> None: ...

    def neutralize(self, resource: ControlResource, lease_token: int) -> None: ...


class _NoopOutputAuthority:
    def authorize(self, lease: ResourceLease) -> None:
        return

    def neutralize(self, resource: ControlResource, lease_token: int) -> None:
        return


@dataclass(frozen=True, slots=True)
class ActionSnapshot:
    action_id: str
    turn_id: str
    action_type: ActionType
    resource: ControlResource
    state: ActionState
    lease_token: int | None
    error: str | None
    started_monotonic: float | None
    ended_monotonic: float | None


@dataclass(frozen=True, slots=True)
class SupervisorSnapshot:
    current_actions: tuple[ActionSnapshot, ...]
    pending_actions: tuple[ActionSnapshot, ...]
    recent_actions: tuple[ActionSnapshot, ...]
    heartbeat_monotonic: float
    last_errors: tuple[str, ...]

    def current(self, resource: ControlResource) -> ActionSnapshot | None:
        return next(
            (
                action
                for action in self.current_actions
                if action.resource == resource
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class BundleApplication:
    turn_id: str
    action_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HaltResult:
    reason: str
    resource_tokens: tuple[tuple[ControlResource, int], ...]


@dataclass(slots=True)
class _PendingAction:
    action_id: str
    turn_id: str
    action: Action
    requested_monotonic: float


@dataclass(slots=True)
class _ActionHandle:
    command: ExecutionCommand
    executor: ActionExecutor
    cancel_event: threading.Event
    done_event: threading.Event
    state: ActionState
    started_monotonic: float | None = None
    error: str | None = None
    thread: threading.Thread | None = None


class CharacterSupervisor:
    """Owns action lifecycle, preemption, and output-resource fencing."""

    def __init__(
        self,
        *,
        executors: Mapping[ActionType, ActionExecutor],
        output_authority: OutputAuthority | None = None,
        resource_manager: ControlResourceManager | None = None,
        action_id_factory: Callable[[], str] | None = None,
        history_limit: int = 50,
    ) -> None:
        if not executors:
            raise ValueError("at least one action executor is required")
        unsupported = set(executors) - set(ActionType)
        if unsupported:
            raise ValueError("executors contain an unsupported action type")
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self._executors = dict(executors)
        self._managed_resources = tuple(
            sorted(
                {_ACTION_RESOURCES[action_type] for action_type in executors},
                key=lambda resource: resource.value,
            )
        )
        self._output = output_authority or _NoopOutputAuthority()
        self._resources = resource_manager or ControlResourceManager()
        self._action_id_factory = action_id_factory or (
            lambda: uuid.uuid4().hex
        )
        self._condition = threading.Condition(threading.RLock())
        self._current: dict[ControlResource, _ActionHandle] = {}
        self._pending: dict[ControlResource, _PendingAction] = {}
        self._recent: deque[ActionSnapshot] = deque(maxlen=history_limit)
        self._last_errors: deque[str] = deque(maxlen=history_limit)
        self._heartbeat_monotonic = time.monotonic()

    def apply_bundle(
        self,
        bundle: ActionBundle,
        *,
        turn_id: str,
    ) -> BundleApplication:
        if not turn_id:
            raise ValueError("turn_id must not be empty")
        if not 1 <= len(bundle.actions) <= 2:
            raise ValueError("bundle must contain one or two actions")
        unmanaged = [
            action.action_type
            for action in bundle.actions
            if action.action_type not in self._executors
        ]
        if unmanaged:
            names = ", ".join(action_type.value for action_type in unmanaged)
            raise ValueError(f"action type is not managed by this supervisor: {names}")
        resources = [action.resource for action in bundle.actions]
        if len(resources) != len(set(resources)):
            raise ValueError("bundle cannot contain duplicate resources")

        pending_actions: list[_PendingAction] = []
        for action in bundle.actions:
            action_id = self._action_id_factory()
            if not isinstance(action_id, str) or not action_id:
                raise ValueError("action_id_factory must return a non-empty string")
            pending_actions.append(
                _PendingAction(
                    action_id=action_id,
                    turn_id=turn_id,
                    action=action,
                    requested_monotonic=time.monotonic(),
                )
            )

        with self._condition:
            for pending in pending_actions:
                self._schedule_locked(pending)
            self._touch_locked()
            self._condition.notify_all()

        return BundleApplication(
            turn_id=turn_id,
            action_ids=tuple(action.action_id for action in pending_actions),
        )

    def halt_all(self, *, reason: str) -> HaltResult:
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must not be empty")
        with self._condition:
            for resource, pending in tuple(self._pending.items()):
                self._recent.append(
                    self._pending_snapshot(
                        pending,
                        state=ActionState.CANCELLED,
                        error=f"cancelled before start: {reason}",
                    )
                )
                del self._pending[resource]

            for resource in self._managed_resources:
                handle = self._current.get(resource)
                if handle is not None and handle.state != ActionState.STOPPING:
                    self._cancel_current_locked(handle, reason=reason)
                elif handle is None:
                    authority = self._resources.current(resource)
                    self._safe_neutralize_locked(resource, authority.token)

            self._touch_locked()
            tokens = tuple(
                (resource, self._resources.current(resource).token)
                for resource in self._managed_resources
            )
            self._condition.notify_all()
            return HaltResult(reason=reason, resource_tokens=tokens)

    def wait_until_idle(self, *, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must not be negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._current or self._pending:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def wait_for_action_state(
        self,
        action_id: str,
        state: ActionState,
        *,
        timeout: float | None = None,
    ) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._find_action_state_locked(action_id) != state:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def snapshot(self) -> SupervisorSnapshot:
        with self._condition:
            return SupervisorSnapshot(
                current_actions=tuple(
                    self._handle_snapshot(handle)
                    for _, handle in sorted(
                        self._current.items(), key=lambda item: item[0].value
                    )
                ),
                pending_actions=tuple(
                    self._pending_snapshot(pending, state=ActionState.PREPARING)
                    for _, pending in sorted(
                        self._pending.items(), key=lambda item: item[0].value
                    )
                ),
                recent_actions=tuple(self._recent),
                heartbeat_monotonic=self._heartbeat_monotonic,
                last_errors=tuple(self._last_errors),
            )

    def _schedule_locked(self, pending: _PendingAction) -> None:
        resource = pending.action.resource
        current = self._current.get(resource)
        if current is None:
            previous_pending = self._pending.pop(resource, None)
            if previous_pending is not None:
                self._record_superseded_pending_locked(previous_pending)
            self._start_locked(pending)
            return

        previous_pending = self._pending.get(resource)
        if previous_pending is not None:
            self._record_superseded_pending_locked(previous_pending)
        self._pending[resource] = pending
        if current.state != ActionState.STOPPING:
            self._cancel_current_locked(current, reason="replaced")

    def _start_locked(self, pending: _PendingAction) -> None:
        resource = pending.action.resource
        lease = self._resources.acquire(resource, pending.action_id)
        command = ExecutionCommand(
            action_id=pending.action_id,
            turn_id=pending.turn_id,
            action=pending.action,
            resource=resource,
            lease_token=lease.token,
        )
        try:
            self._output.authorize(lease)
        except Exception as exc:
            self._resources.invalidate(
                resource,
                expected_action_id=pending.action_id,
            )
            error = f"output authorization failed: {exc}"
            self._last_errors.append(error)
            self._recent.append(
                ActionSnapshot(
                    action_id=pending.action_id,
                    turn_id=pending.turn_id,
                    action_type=pending.action.action_type,
                    resource=resource,
                    state=ActionState.FAILED,
                    lease_token=lease.token,
                    error=error,
                    started_monotonic=None,
                    ended_monotonic=time.monotonic(),
                )
            )
            self._safe_neutralize_locked(
                resource,
                self._resources.current(resource).token,
            )
            return

        handle = _ActionHandle(
            command=command,
            executor=self._executors[pending.action.action_type],
            cancel_event=threading.Event(),
            done_event=threading.Event(),
            state=ActionState.PREPARING,
        )
        thread = threading.Thread(
            target=self._run_handle,
            args=(handle,),
            name=f"action-{pending.action.action_type.value}-{pending.action_id}",
            daemon=True,
        )
        handle.thread = thread
        self._current[resource] = handle
        thread.start()

    def _run_handle(self, handle: _ActionHandle) -> None:
        error: BaseException | None = None
        try:
            handle.executor.run(
                handle.command,
                handle.cancel_event,
                lambda: self._mark_running(handle),
            )
        except BaseException as exc:
            error = exc
        self._finish_handle(handle, error=error)

    def _mark_running(self, handle: _ActionHandle) -> None:
        with self._condition:
            if self._current.get(handle.command.resource) is not handle:
                return
            if handle.state != ActionState.PREPARING or handle.cancel_event.is_set():
                return
            handle.state = ActionState.RUNNING
            handle.started_monotonic = time.monotonic()
            self._touch_locked()
            self._condition.notify_all()

    def _finish_handle(
        self,
        handle: _ActionHandle,
        *,
        error: BaseException | None,
    ) -> None:
        with self._condition:
            resource = handle.command.resource
            if self._current.get(resource) is not handle:
                handle.done_event.set()
                self._condition.notify_all()
                return

            if handle.state != ActionState.STOPPING:
                invalidated = self._resources.invalidate(
                    resource,
                    expected_action_id=handle.command.action_id,
                )
                self._safe_neutralize_locked(resource, invalidated.token)

            if error is not None:
                handle.state = ActionState.FAILED
                handle.error = f"{type(error).__name__}: {error}"
                self._last_errors.append(
                    f"{handle.command.action_id}: {handle.error}"
                )
            elif handle.cancel_event.is_set():
                handle.state = ActionState.CANCELLED
            else:
                handle.state = ActionState.COMPLETED

            ended = time.monotonic()
            self._recent.append(self._handle_snapshot(handle, ended_monotonic=ended))
            del self._current[resource]
            handle.done_event.set()

            pending = self._pending.pop(resource, None)
            if pending is not None:
                self._start_locked(pending)
            self._touch_locked()
            self._condition.notify_all()

    def _cancel_current_locked(
        self,
        handle: _ActionHandle,
        *,
        reason: str,
    ) -> None:
        # Fencing and neutral output must happen before the executor sees the
        # cancel signal. Otherwise a late frame can overwrite the stop output.
        invalidated = self._resources.invalidate(
            handle.command.resource,
            expected_action_id=handle.command.action_id,
        )
        self._safe_neutralize_locked(handle.command.resource, invalidated.token)
        handle.state = ActionState.STOPPING
        handle.error = None if reason == "replaced" else f"cancel requested: {reason}"
        handle.cancel_event.set()

    def _safe_neutralize_locked(
        self,
        resource: ControlResource,
        lease_token: int,
    ) -> None:
        try:
            self._output.neutralize(resource, lease_token)
        except Exception as exc:
            self._last_errors.append(
                f"{resource.value} neutralize failed at token {lease_token}: {exc}"
            )

    def _record_superseded_pending_locked(self, pending: _PendingAction) -> None:
        self._recent.append(
            self._pending_snapshot(
                pending,
                state=ActionState.CANCELLED,
                error="superseded before start",
            )
        )

    def _find_action_state_locked(self, action_id: str) -> ActionState | None:
        for handle in self._current.values():
            if handle.command.action_id == action_id:
                return handle.state
        for pending in self._pending.values():
            if pending.action_id == action_id:
                return ActionState.PREPARING
        for action in reversed(self._recent):
            if action.action_id == action_id:
                return action.state
        return None

    def _handle_snapshot(
        self,
        handle: _ActionHandle,
        *,
        ended_monotonic: float | None = None,
    ) -> ActionSnapshot:
        return ActionSnapshot(
            action_id=handle.command.action_id,
            turn_id=handle.command.turn_id,
            action_type=handle.command.action.action_type,
            resource=handle.command.resource,
            state=handle.state,
            lease_token=handle.command.lease_token,
            error=handle.error,
            started_monotonic=handle.started_monotonic,
            ended_monotonic=ended_monotonic,
        )

    @staticmethod
    def _pending_snapshot(
        pending: _PendingAction,
        *,
        state: ActionState,
        error: str | None = None,
    ) -> ActionSnapshot:
        return ActionSnapshot(
            action_id=pending.action_id,
            turn_id=pending.turn_id,
            action_type=pending.action.action_type,
            resource=pending.action.resource,
            state=state,
            lease_token=None,
            error=error,
            started_monotonic=None,
            ended_monotonic=(
                time.monotonic()
                if state in {ActionState.COMPLETED, ActionState.CANCELLED, ActionState.FAILED}
                else None
            ),
        )

    def _touch_locked(self) -> None:
        self._heartbeat_monotonic = time.monotonic()
