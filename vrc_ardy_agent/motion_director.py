from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import threading
import time
from typing import Any
import uuid

from .action_contracts import ControlResource, ResourceLease
from .live_session import LiveSessionStatus
from .motion_cue import MotionCue, CuePhase
from .motion_director_actions import MotionDirectorActions
from .motion_director_lifecycle import MotionDirectorLifecycle
from .motion_session import MotionSessionPlayback
from .motion_stream import (
    CALIBRATED_IDLE,
    DEFAULT_POSE_TTL_MS,
    MotionDirectorSnapshot,
    MotionDirectorState,
    MotionHaltResult,
)


class MotionDirector(MotionDirectorActions, MotionDirectorLifecycle):
    """Own one output stream across idle, generated cues and registered motions.

    The canonical anchored session keeps calibrated idle without generation.
    Completion returns through that same stream, preserving tracker ownership.
    Explicit halt, output loss or failure fences the entire output resource.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        output: Any,
        mapper_factory: Callable[[], Any],
        session_factory: Callable[..., Any],
        idle_prompt: str = CALIBRATED_IDLE,
        motion_assets=None,
        replan_threshold_frames: int | None = None,
        pose_ttl_ms: int = DEFAULT_POSE_TTL_MS,
        cue_poll_seconds: float = 0.05,
        cue_prepare_timeout_seconds: float = 15.0,
        realtime: bool = True,
        action_id_factory: Callable[[], str] | None = None,
    ) -> None:
        idle_prompt = idle_prompt.strip()
        if not idle_prompt:
            raise ValueError("idle_prompt must not be empty")
        if isinstance(pose_ttl_ms, bool) or not isinstance(pose_ttl_ms, int):
            raise ValueError("pose_ttl_ms must be an integer")
        if pose_ttl_ms <= 0:
            raise ValueError("pose_ttl_ms must be positive")
        if cue_poll_seconds <= 0:
            raise ValueError("cue_poll_seconds must be positive")
        if cue_prepare_timeout_seconds <= 0:
            raise ValueError("cue_prepare_timeout_seconds must be positive")

        self.motion_assets = motion_assets
        self.runtime = runtime
        self.output = output
        self.mapper_factory = mapper_factory
        self.session_factory = session_factory
        self.idle_prompt = idle_prompt
        self.replan_threshold_frames = replan_threshold_frames
        self.pose_ttl_ms = pose_ttl_ms
        self.cue_poll_seconds = float(cue_poll_seconds)
        self.cue_prepare_timeout_seconds = float(cue_prepare_timeout_seconds)
        self.realtime = bool(realtime)
        self._action_id_factory = action_id_factory or (lambda: uuid.uuid4().hex)

        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._manager_thread: threading.Thread | None = None
        self._output_session_id: str | None = None
        self._enabled = False
        self._lease_token = 0
        self._stream_lease: ResourceLease | None = None
        self._stream_fence_token: int | None = None
        self._playback: MotionSessionPlayback | None = None
        self._cue: MotionCue | None = None
        self._status = MotionDirectorSnapshot(
            heartbeat_monotonic=time.monotonic()
        )

    def start(self) -> None:
        with self._condition:
            if self._manager_thread is not None:
                raise RuntimeError("motion director is already started")
            if self._stop_event.is_set():
                raise RuntimeError("motion director cannot be restarted after stop")
            self._enabled = True
            self._status = replace(
                self._status,
                state=MotionDirectorState.WAITING_OUTPUT,
                running=True,
                active_prompt=self.idle_prompt,
                heartbeat_monotonic=time.monotonic(),
                last_error=None,
            )
            thread = threading.Thread(
                target=self._manager_loop,
                name="ardy-motion-director",
                daemon=True,
            )
            self._manager_thread = thread
            thread.start()

    def stop(self) -> None:
        with self._condition:
            thread = self._manager_thread
            if thread is None:
                return
        self.halt(reason="motion_director_shutdown")
        with self._condition:
            self._stop_event.set()
            self._request_stream_stop_locked("shutdown")
            self._condition.notify_all()
        if thread is not threading.current_thread():
            thread.join()
        with self._condition:
            self._manager_thread = None

    def attach_output_session(self, session_id: str) -> None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        with self._condition:
            if self._manager_thread is None:
                raise RuntimeError("start the motion director before attaching output")
            if self._output_session_id == session_id:
                self._touch_locked()
                return
            if self._output_session_id is not None:
                raise RuntimeError("another output session is already attached")
            self._output_session_id = session_id
            self._status = replace(
                self._status,
                output_session_id=session_id,
                state=(
                    MotionDirectorState.STARTING
                    if self._enabled
                    else MotionDirectorState.HALTED
                ),
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()

    def detach_output_session(self, session_id: str, *, reason: str) -> bool:
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must not be empty")
        with self._condition:
            if self._output_session_id != session_id:
                return False
            self._output_session_id = None
            self._cue = None
            self._request_stream_stop_locked("output_detached")
            self._status = replace(
                self._status,
                state=(
                    MotionDirectorState.STOPPING
                    if self._stream_lease is not None
                    else (
                        MotionDirectorState.WAITING_OUTPUT
                        if self._enabled
                        else MotionDirectorState.HALTED
                    )
                ),
                output_session_id=None,
                active_prompt=(self.idle_prompt if self._enabled else None),
                active_cue_id=None,
                active_turn_id=None,
                cue_prompt_revision=None,
                cue_prepare_deadline_monotonic=None,
                cue_started_monotonic=None,
                cue_deadline_monotonic=None,
                idle_prompt_revision=None,
                playing_prompt=None,
                playing_prompt_revision=0,
                heartbeat_monotonic=time.monotonic(),
                last_error=reason,
            )
            self._condition.notify_all()
            return True


    def halt(self, *, reason: str) -> MotionHaltResult:
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must not be empty")
        neutralize_token: int | None = None
        with self._condition:
            if self._manager_thread is None:
                return MotionHaltResult(reason=reason, lease_token=self._lease_token)
            self._enabled = False
            self._cue = None
            if (
                self._stream_lease is not None
                and self._stream_fence_token is None
            ):
                self._lease_token = max(
                    self._lease_token,
                    self._stream_lease.token,
                ) + 1
                neutralize_token = self._lease_token
                self._stream_fence_token = neutralize_token
            self._request_stream_stop_locked("halted")
            self._status = replace(
                self._status,
                state=(
                    MotionDirectorState.STOPPING
                    if self._stream_lease is not None
                    else MotionDirectorState.HALTED
                ),
                lease_token=self._lease_token,
                active_prompt=None,
                active_cue_id=None,
                active_turn_id=None,
                cue_prompt_revision=None,
                cue_prepare_deadline_monotonic=None,
                cue_started_monotonic=None,
                cue_deadline_monotonic=None,
                idle_prompt_revision=None,
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()

        if neutralize_token is not None:
            try:
                self.output.neutralize(
                    ControlResource.FULL_BODY_POSE,
                    neutralize_token,
                )
            except BaseException as exc:
                self._record_error("motion neutralization failed", exc)
                raise
        return MotionHaltResult(reason=reason, lease_token=self._lease_token)

    def wait_until_state(
        self,
        state: MotionDirectorState,
        *,
        timeout: float | None = None,
    ) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._status.state == state,
                timeout=timeout,
            )

    def snapshot(self) -> MotionDirectorSnapshot:
        with self._condition:
            return replace(self._status)

    def _manager_loop(self) -> None:
        try:
            while True:
                start_spec: tuple[str, ResourceLease, str, str | None] | None = None
                transition_error: tuple[str, BaseException] | None = None
                with self._condition:
                    self._finish_stream_if_done_locked()
                    if self._stop_event.is_set() and self._playback is None:
                        return

                    now = time.monotonic()
                    cue = self._cue
                    timeout_kind = cue.timeout_kind(now) if cue is not None else None
                    if timeout_kind is not None:
                        transition_error = (
                            f"{timeout_kind} did not start",
                            TimeoutError(
                                f"{timeout_kind} did not reach output within "
                                f"{self.cue_prepare_timeout_seconds:g}s"
                            ),
                        )
                    elif cue is not None and (cue.playback_expired(now) or (
                        cue.completion_driven and cue.phase is CuePhase.PLAYING
                        and self._playback is not None
                        and self._playback.session.completed_revision == cue.prompt_revision
                    )):
                        playback = self._playback
                        if playback is None or self._stream_fence_token is not None:
                            transition_error = (
                                "idle prompt update failed",
                                RuntimeError("motion stream disappeared during cue playback"),
                            )
                        else:
                            try:
                                idle_revision = playback.session.set_prompt(
                                    self.idle_prompt
                                )
                                cue.begin_return_to_idle(
                                    idle_revision,
                                    now=now,
                                    timeout_seconds=self.cue_prepare_timeout_seconds,
                                )
                            except BaseException as exc:
                                transition_error = ("idle prompt update failed", exc)
                            else:
                                self._status = replace(
                                    self._status,
                                    state=MotionDirectorState.RETURNING_IDLE,
                                    active_prompt=self.idle_prompt,
                                    idle_prompt_revision=cue.idle_revision,
                                    heartbeat_monotonic=now,
                                )

                    if (
                        transition_error is None
                        and self._playback is None
                        and self._enabled
                        and self._output_session_id is not None
                    ):
                        self._lease_token += 1
                        lease = ResourceLease(
                            resource=ControlResource.FULL_BODY_POSE,
                            token=self._lease_token,
                            action_id=f"motion-stream-{uuid.uuid4().hex}",
                        )
                        self._stream_lease = lease
                        self._stream_fence_token = None
                        initial_prompt = cue.prompt if cue is not None else self.idle_prompt
                        start_spec = (
                            self._output_session_id,
                            lease,
                            initial_prompt,
                            cue.action_id if cue is not None else None,
                        )
                        self._status = replace(
                            self._status,
                            state=MotionDirectorState.STARTING,
                            stream_action_id=lease.action_id,
                            lease_token=lease.token,
                            active_prompt=initial_prompt,
                            playing_prompt=None,
                            playing_prompt_revision=0,
                            heartbeat_monotonic=now,
                            last_error=None,
                        )

                    if transition_error is None and start_spec is None:
                        timeout = self.cue_poll_seconds
                        deadline = cue.next_deadline() if cue is not None else None
                        if deadline is not None:
                            timeout = min(
                                timeout,
                                max(0.0, deadline - now),
                            )
                        self._condition.wait(timeout=timeout)
                        continue

                if transition_error is not None:
                    context, error = transition_error
                    self._fail_stream(context, error)
                    continue

                if start_spec is not None:
                    self._start_stream(*start_spec)
        finally:
            with self._condition:
                self._status = replace(
                    self._status,
                    state=MotionDirectorState.STOPPED,
                    running=False,
                    stream_action_id=None,
                    heartbeat_monotonic=time.monotonic(),
                )
                self._condition.notify_all()

    def _finish_stream_if_done_locked(self) -> None:
        playback = self._playback
        if playback is None or playback.is_alive:
            return
        error, stop_reason = playback.result()
        intentional = stop_reason is not None
        self._playback = None
        self._stream_lease = None
        self._stream_fence_token = None
        self.runtime.clear_history()

        if self._stop_event.is_set():
            return
        if error is not None and stop_reason == "failed":
            self._enabled = False
            self._status = replace(
                self._status,
                state=MotionDirectorState.FAILED,
                stream_action_id=None,
                last_error=self._status.last_error
                or f"motion stream failed: {type(error).__name__}: {error}",
                heartbeat_monotonic=time.monotonic(),
            )
            return
        if not intentional and error is not None:
            self._enabled = False
            self._status = replace(
                self._status,
                state=MotionDirectorState.FAILED,
                stream_action_id=None,
                last_error=f"motion stream failed: {type(error).__name__}: {error}",
                heartbeat_monotonic=time.monotonic(),
            )
            return
        if not intentional:
            self._enabled = False
            self._status = replace(
                self._status,
                state=MotionDirectorState.FAILED,
                stream_action_id=None,
                last_error="motion stream ended unexpectedly",
                heartbeat_monotonic=time.monotonic(),
            )
            return
        self._status = replace(
            self._status,
            state=(
                MotionDirectorState.HALTED
                if not self._enabled
                else MotionDirectorState.WAITING_OUTPUT
            ),
            stream_action_id=None,
            active_prompt=(
                None
                if not self._enabled
                else (self._cue.prompt if self._cue is not None else self.idle_prompt)
            ),
            playing_prompt=None,
            playing_prompt_revision=0,
            heartbeat_monotonic=time.monotonic(),
        )

    def _request_stream_stop_locked(self, reason: str) -> None:
        if self._playback is None:
            return
        self._playback.request_stop(reason)

    def _fail_stream(self, context: str, error: BaseException) -> None:
        neutralize_token: int | None = None
        with self._condition:
            self._enabled = False
            self._cue = None
            if (
                self._stream_lease is not None
                and self._stream_fence_token is None
            ):
                self._lease_token = max(
                    self._lease_token,
                    self._stream_lease.token,
                ) + 1
                neutralize_token = self._lease_token
                self._stream_fence_token = neutralize_token
            if self._playback is not None:
                self._playback.fail(error)
            self._status = replace(
                self._status,
                state=MotionDirectorState.STOPPING,
                lease_token=self._lease_token,
                active_prompt=None,
                active_cue_id=None,
                active_turn_id=None,
                cue_prompt_revision=None,
                cue_prepare_deadline_monotonic=None,
                cue_started_monotonic=None,
                cue_deadline_monotonic=None,
                idle_prompt_revision=None,
                last_error=f"{context}: {type(error).__name__}: {error}",
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()
        if neutralize_token is not None:
            try:
                self.output.neutralize(
                    ControlResource.FULL_BODY_POSE,
                    neutralize_token,
                )
            except BaseException as neutralize_error:
                with self._condition:
                    self._status = replace(
                        self._status,
                        last_error=(
                            f"{self._status.last_error}; motion neutralization failed: "
                            f"{type(neutralize_error).__name__}: {neutralize_error}"
                        ),
                        heartbeat_monotonic=time.monotonic(),
                    )
                    self._condition.notify_all()

    def _notify_manager(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _on_session_heartbeat(self, status: LiveSessionStatus) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                played_frames=int(status.played_frames),
                generated_chunks=int(status.generated_chunks),
                underruns=int(status.underruns),
                last_generation_seconds=status.last_generation_seconds,
                playing_prompt=status.playing_prompt,
                playing_prompt_revision=status.playing_prompt_revision,
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()

    def _touch_locked(self) -> None:
        self._status = replace(
            self._status,
            heartbeat_monotonic=time.monotonic(),
        )
