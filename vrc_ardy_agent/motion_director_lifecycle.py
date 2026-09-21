from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

from .action_contracts import ResourceLease
from .live_session import PromptPlaybackStarted
from .motion_cue import CuePlaybackObservation
from .motion_session import MotionSessionPlayback
from .motion_stream import DirectorPoseSink, MotionDirectorState


class MotionDirectorLifecycle:
    """Internal stream lifecycle used by MotionDirector under its condition lock."""

    def _start_stream(
        self,
        output_session_id: str,
        lease: ResourceLease,
        initial_prompt: str,
        initial_cue_id: str | None,
    ) -> None:
        session: Any | None = None
        history_touched = False
        try:
            self.output.authorize(lease)
            with self._condition:
                still_current = (
                    self._enabled
                    and not self._stop_event.is_set()
                    and self._output_session_id == output_session_id
                    and self._stream_lease == lease
                    and self._stream_fence_token is None
                )
                if not still_current:
                    if self._stream_lease == lease:
                        self._stream_lease = None
                        self._stream_fence_token = None
                    self._status = replace(
                        self._status,
                        state=(
                            MotionDirectorState.HALTED
                            if not self._enabled
                            else MotionDirectorState.WAITING_OUTPUT
                        ),
                        stream_action_id=None,
                        active_prompt=(None if not self._enabled else self.idle_prompt),
                        heartbeat_monotonic=time.monotonic(),
                    )
                    self._condition.notify_all()
            if not still_current:
                return
            session = self.session_factory(
                runtime=self.runtime,
                mapper=self.mapper_factory(),
                sink=DirectorPoseSink(
                    output=self.output,
                    lease=lease,
                    ttl_ms=self.pose_ttl_ms,
                ),
                replan_threshold_frames=self.replan_threshold_frames,
            )
            history_touched = True
            applied_revision = session.start(initial_prompt)
            applied_prompt = initial_prompt
            applied_cue_id = initial_cue_id

            while True:
                with self._condition:
                    still_current = (
                        self._enabled
                        and not self._stop_event.is_set()
                        and self._output_session_id == output_session_id
                        and self._stream_lease == lease
                        and self._stream_fence_token is None
                    )
                    cue = self._cue
                    desired_prompt = cue.prompt if cue is not None else self.idle_prompt
                    desired_cue_id = cue.action_id if cue is not None else None
                    reconciled = (
                        desired_prompt == applied_prompt
                        and desired_cue_id == applied_cue_id
                    )
                    if still_current and reconciled:
                        if cue is not None:
                            cue.bind_prompt_revision(
                                applied_revision,
                                now=time.monotonic(),
                                timeout_seconds=self.cue_prepare_timeout_seconds,
                            )
                        playback = MotionSessionPlayback(
                            session=session,
                            realtime=self.realtime,
                            heartbeat=self._on_session_heartbeat,
                            prompt_started=self._on_prompt_playback_started,
                            finished=self._notify_manager,
                        )
                        self._playback = playback
                        self._status = replace(
                            self._status,
                            state=(
                                MotionDirectorState.PREPARING_CUE
                                if cue is not None
                                else MotionDirectorState.IDLE
                            ),
                            active_prompt=desired_prompt,
                            active_cue_id=(cue.action_id if cue is not None else None),
                            active_turn_id=(cue.turn_id if cue is not None else None),
                            cue_prompt_revision=(
                                cue.prompt_revision if cue is not None else None
                            ),
                            cue_prepare_deadline_monotonic=(
                                cue.prepare_deadline_monotonic
                                if cue is not None
                                else None
                            ),
                            cue_started_monotonic=None,
                            cue_deadline_monotonic=None,
                            idle_prompt_revision=None,
                            heartbeat_monotonic=time.monotonic(),
                        )
                        playback.start()
                        self._condition.notify_all()
                        return
                if not still_current:
                    break
                applied_revision = session.set_prompt(desired_prompt)
                applied_prompt = desired_prompt
                applied_cue_id = desired_cue_id
        except BaseException as exc:
            if session is not None:
                try:
                    session.stop()
                except BaseException:
                    pass
            if history_touched:
                self.runtime.clear_history()
            with self._condition:
                if self._stream_lease == lease:
                    self._stream_lease = None
                    self._stream_fence_token = None
                if (
                    self._enabled
                    and self._output_session_id == output_session_id
                    and not self._stop_event.is_set()
                ):
                    self._enabled = False
                    self._cue = None
                    self._status = replace(
                        self._status,
                        state=MotionDirectorState.FAILED,
                        stream_action_id=None,
                        active_cue_id=None,
                        active_turn_id=None,
                        cue_prompt_revision=None,
                        cue_prepare_deadline_monotonic=None,
                        cue_started_monotonic=None,
                        cue_deadline_monotonic=None,
                        idle_prompt_revision=None,
                        last_error=f"stream start failed: {type(exc).__name__}: {exc}",
                        heartbeat_monotonic=time.monotonic(),
                    )
                elif not self._enabled:
                    self._status = replace(
                        self._status,
                        state=MotionDirectorState.HALTED,
                        stream_action_id=None,
                        active_prompt=None,
                        heartbeat_monotonic=time.monotonic(),
                    )
                elif self._output_session_id is None:
                    self._status = replace(
                        self._status,
                        state=MotionDirectorState.WAITING_OUTPUT,
                        stream_action_id=None,
                        active_prompt=self.idle_prompt,
                        heartbeat_monotonic=time.monotonic(),
                    )
                self._condition.notify_all()
            return
        try:
            session.request_stop()
            session.stop()
        finally:
            self.runtime.clear_history()
            with self._condition:
                if self._stream_lease == lease:
                    self._stream_lease = None
                    self._stream_fence_token = None
                self._status = replace(
                    self._status,
                    state=(
                        MotionDirectorState.HALTED
                        if not self._enabled
                        else MotionDirectorState.WAITING_OUTPUT
                    ),
                    stream_action_id=None,
                    heartbeat_monotonic=time.monotonic(),
                )
                self._condition.notify_all()

    def _on_prompt_playback_started(self, event: PromptPlaybackStarted) -> None:
        with self._condition:
            now = time.monotonic()
            cue = self._cue
            observation = (
                cue.observe_playback(event)
                if cue is not None
                else CuePlaybackObservation.IGNORED
            )
            if observation is CuePlaybackObservation.CUE_STARTED:
                self._status = replace(
                    self._status,
                    state=MotionDirectorState.CUE,
                    cue_prepare_deadline_monotonic=None,
                    cue_started_monotonic=cue.started_monotonic,
                    cue_deadline_monotonic=cue.playback_deadline_monotonic,
                    playing_prompt=event.prompt,
                    playing_prompt_revision=event.revision,
                    heartbeat_monotonic=now,
                )
            elif observation is CuePlaybackObservation.IDLE_STARTED:
                self._cue = None
                self._status = replace(
                    self._status,
                    state=MotionDirectorState.IDLE,
                    active_prompt=self.idle_prompt,
                    active_cue_id=None,
                    active_turn_id=None,
                    cue_prompt_revision=None,
                    cue_prepare_deadline_monotonic=None,
                    cue_started_monotonic=None,
                    cue_deadline_monotonic=None,
                    idle_prompt_revision=None,
                    playing_prompt=event.prompt,
                    playing_prompt_revision=event.revision,
                    heartbeat_monotonic=now,
                )
            else:
                self._status = replace(
                    self._status,
                    playing_prompt=event.prompt,
                    playing_prompt_revision=event.revision,
                    heartbeat_monotonic=now,
                )
            self._condition.notify_all()
