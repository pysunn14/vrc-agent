from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import threading
import time
from typing import Any

from .windows_vrchat_identity import (
    AmbiguousVrchatIdentityError,
    VrchatTarget,
)


class WindowsCompanionState(str, Enum):
    STOPPED = "STOPPED"
    WAITING = "WAITING"
    AMBIGUOUS = "AMBIGUOUS"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DETACHED = "DETACHED"
    RETRYING = "RETRYING"


@dataclass(frozen=True, slots=True)
class WindowsCompanionSupervisorSnapshot:
    running: bool
    state: WindowsCompanionState
    epoch: int
    target: Mapping[str, object] | None
    components: Mapping[str, object]
    attach_attempts: int
    successful_attaches: int
    detachments: int
    resolution_failures: int
    heartbeat_monotonic: float
    last_detach_reason: str | None
    last_error: str | None


class WindowsCompanionSupervisor:
    """Own replaceable companion epochs while keeping control state observable."""

    def __init__(
        self,
        *,
        resolve_target: Callable[[], VrchatTarget | None],
        target_is_current: Callable[[VrchatTarget], bool],
        runtime_factory: Callable[[VrchatTarget, int], Any],
        poll_interval_seconds: float = 0.5,
        retry_delay_seconds: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(resolve_target):
            raise TypeError("resolve_target must be callable")
        if not callable(target_is_current):
            raise TypeError("target_is_current must be callable")
        if not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if retry_delay_seconds <= 0:
            raise ValueError("retry_delay_seconds must be positive")

        self._resolve_target = resolve_target
        self._target_is_current = target_is_current
        self._runtime_factory = runtime_factory
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._retry_delay_seconds = float(retry_delay_seconds)
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._runtime: Any | None = None
        self._status = WindowsCompanionSupervisorSnapshot(
            running=False,
            state=WindowsCompanionState.STOPPED,
            epoch=0,
            target=None,
            components={},
            attach_attempts=0,
            successful_attaches=0,
            detachments=0,
            resolution_failures=0,
            heartbeat_monotonic=self._monotonic(),
            last_detach_reason=None,
            last_error=None,
        )

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Windows companion supervisor is already started")
            self._stop_event.clear()
            self._status = replace(
                self._status,
                running=True,
                state=WindowsCompanionState.WAITING,
                heartbeat_monotonic=self._monotonic(),
                last_error=None,
            )
            thread = threading.Thread(
                target=self._run,
                name="windows-companion-supervisor",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                self._status = replace(
                    self._status,
                    running=False,
                    state=WindowsCompanionState.STOPPED,
                    heartbeat_monotonic=self._monotonic(),
                )
                return
            self._stop_event.set()
        if thread is not threading.current_thread():
            thread.join(timeout=15.0)
            if thread.is_alive():
                raise TimeoutError("Windows companion supervisor did not stop")
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def snapshot(self) -> WindowsCompanionSupervisorSnapshot:
        with self._lock:
            runtime = self._runtime
            status = self._status
        components = status.components
        if runtime is not None:
            try:
                runtime_snapshot = runtime.snapshot()
                components = dict(getattr(runtime_snapshot, "components", {}))
            except Exception as exc:
                components = {
                    "supervisor_snapshot_error": (
                        f"{type(exc).__name__}: {exc}"
                    )
                }
        with self._lock:
            self._status = replace(
                self._status,
                components=components,
            )
            return replace(self._status, components=dict(components))

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                target = self._resolve_once()
                if target is None:
                    self._stop_event.wait(self._poll_interval_seconds)
                    continue
                self._run_epoch(target)
        except Exception as exc:
            with self._lock:
                self._status = replace(
                    self._status,
                    state=WindowsCompanionState.STOPPED,
                    heartbeat_monotonic=self._monotonic(),
                    last_error=(
                        f"supervisor crashed: {type(exc).__name__}: {exc}"
                    ),
                )
        finally:
            self._stop_current_runtime()
            with self._lock:
                self._status = replace(
                    self._status,
                    running=False,
                    state=WindowsCompanionState.STOPPED,
                    target=None,
                    components={},
                    heartbeat_monotonic=self._monotonic(),
                )

    def _resolve_once(self) -> VrchatTarget | None:
        self._transition(WindowsCompanionState.WAITING, target=None)
        try:
            target = self._resolve_target()
        except AmbiguousVrchatIdentityError as exc:
            self._record_resolution_failure(
                WindowsCompanionState.AMBIGUOUS,
                exc,
            )
            self._stop_event.wait(self._retry_delay_seconds)
            return None
        except Exception as exc:
            self._record_resolution_failure(
                WindowsCompanionState.RETRYING,
                exc,
            )
            self._stop_event.wait(self._retry_delay_seconds)
            return None
        return target

    def _run_epoch(self, target: VrchatTarget) -> None:
        with self._lock:
            proposed_epoch = self._status.epoch + 1
            self._status = replace(
                self._status,
                state=WindowsCompanionState.STARTING,
                target=_target_status(target),
                attach_attempts=self._status.attach_attempts + 1,
                heartbeat_monotonic=self._monotonic(),
                last_error=None,
            )
        runtime: Any | None = None
        try:
            runtime = self._runtime_factory(target, proposed_epoch)
            runtime.start()
        except Exception as exc:
            cleanup_error = _stop_runtime_instance(runtime)
            error = f"attach failed: {type(exc).__name__}: {exc}"
            if cleanup_error is not None:
                error = f"{error}; {cleanup_error}"
            with self._lock:
                self._status = replace(
                    self._status,
                    state=WindowsCompanionState.RETRYING,
                    target=None,
                    heartbeat_monotonic=self._monotonic(),
                    last_error=error,
                )
            self._stop_event.wait(self._retry_delay_seconds)
            return

        with self._lock:
            self._runtime = runtime
            self._status = replace(
                self._status,
                state=WindowsCompanionState.RUNNING,
                epoch=proposed_epoch,
                target=_target_status(target),
                successful_attaches=self._status.successful_attaches + 1,
                heartbeat_monotonic=self._monotonic(),
                last_error=None,
            )

        detach_reason: str | None = None
        while not self._stop_event.wait(self._poll_interval_seconds):
            try:
                target_is_current = self._target_is_current(target)
            except Exception as exc:
                detach_reason = (
                    f"target inspection failed: {type(exc).__name__}: {exc}"
                )
                break
            if not target_is_current:
                detach_reason = "target process changed or exited"
                break
            try:
                healthy, reason, components = inspect_runtime_health(runtime)
            except Exception as exc:
                detach_reason = (
                    f"runtime health check failed: {type(exc).__name__}: {exc}"
                )
                break
            with self._lock:
                self._status = replace(
                    self._status,
                    components=components,
                    heartbeat_monotonic=self._monotonic(),
                )
            if not healthy:
                detach_reason = reason
                break

        stop_error = self._stop_current_runtime()
        if detach_reason is None:
            return
        with self._lock:
            self._status = replace(
                self._status,
                state=WindowsCompanionState.DETACHED,
                target=None,
                components={},
                detachments=self._status.detachments + 1,
                heartbeat_monotonic=self._monotonic(),
                last_detach_reason=detach_reason,
                last_error=stop_error,
            )
        self._stop_event.wait(self._retry_delay_seconds)

    def _stop_current_runtime(self) -> str | None:
        with self._lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is None:
            return None
        error = _stop_runtime_instance(runtime)
        if error is not None:
            with self._lock:
                self._status = replace(
                    self._status,
                    heartbeat_monotonic=self._monotonic(),
                    last_error=error,
                )
        return error

    def _transition(
        self,
        state: WindowsCompanionState,
        *,
        target: Mapping[str, object] | None,
    ) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                state=state,
                target=target,
                heartbeat_monotonic=self._monotonic(),
            )

    def _record_resolution_failure(
        self,
        state: WindowsCompanionState,
        exc: BaseException,
    ) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                state=state,
                resolution_failures=self._status.resolution_failures + 1,
                heartbeat_monotonic=self._monotonic(),
                last_error=f"{type(exc).__name__}: {exc}",
            )


def inspect_runtime_health(runtime: Any) -> tuple[bool, str, dict[str, object]]:
    snapshot = runtime.snapshot()
    components = dict(getattr(snapshot, "components", {}))
    if not bool(getattr(snapshot, "running", False)):
        return False, "companion runtime stopped", components
    audio = components.get("audio")
    if audio is not None and not _component_running(audio):
        return False, "process audio capture stopped", components
    bridge = components.get("bridge")
    if bridge is not None and not _component_running(bridge):
        return False, "Mac bridge stopped", components
    return True, "", components


def _stop_runtime_instance(runtime: Any | None) -> str | None:
    if runtime is None:
        return None
    try:
        runtime.stop()
    except Exception as exc:
        return f"epoch stop failed: {type(exc).__name__}: {exc}"
    return None


def _component_running(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("running"))
    return bool(getattr(value, "running", False))


def _target_status(target: VrchatTarget) -> dict[str, object]:
    return {
        "pid": target.window.process_id,
        "hwnd": target.window.hwnd,
        "process_started_at": target.process_started_at,
        "display_name": target.identity.display_name,
        "user_id": target.identity.user_id,
        "log_file": Path(target.log_path).name,
    }
