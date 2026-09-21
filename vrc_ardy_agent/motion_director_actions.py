from dataclasses import replace
import time
from .action_contracts import ArdyMotionAction, PresetMotionAction, validate_action_bundle
from .motion_cue import MotionCue
from .motion_stream import MotionDirectorState


class MotionDirectorActions:
    """Submit generated or registered actions under the director's single lease."""
    def submit_idle_cue(self, action, *, turn_id):
        # Recheck output ownership and body state atomically with submission.
        with self._condition:
            if (not self._enabled or self._output_session_id is None
                    or self._status.state != MotionDirectorState.IDLE or self._cue is not None):
                return None
            return self.submit_cue(action, turn_id=turn_id)

    def submit_cue(self, action: ArdyMotionAction | PresetMotionAction, *, turn_id: str) -> str:
        if isinstance(action, PresetMotionAction):
            if self.motion_assets is None:
                raise ValueError("registered motions require calibrated assets")
            payload = {"type": "motion", "name": action.name}
            if action.duration_seconds is not None:
                payload["duration_seconds"] = action.duration_seconds
            validate_action_bundle({"actions": [payload]}, behaviors=self.motion_assets.behaviors)
            spec = self.motion_assets.behaviors[action.name]
            prompt = spec.prompt if spec.source == "ardy" else "preset:" + action.name
            finite = spec.finite
            duration = self.motion_assets.duration(action.name) if action.duration_seconds is None else action.duration_seconds
        elif isinstance(action, ArdyMotionAction):
            if action.prompt.startswith(("preset:", "idle:")):
                raise ValueError("use a registered motion action, not a reserved ARDY prompt")
            prompt, duration, finite = action.prompt, action.duration_seconds, False
        else:
            raise TypeError("motion director requires a body action")
        turn_id = turn_id.strip()
        if not turn_id:
            raise ValueError("turn_id must not be empty")
        cue_id = self._action_id_factory()
        if not isinstance(cue_id, str) or not cue_id:
            raise ValueError("action_id_factory must return a non-empty string")

        prompt_error: BaseException | None = None
        with self._condition:
            if self._manager_thread is None:
                raise RuntimeError("motion director is not running")
            self._enabled = True
            cue = MotionCue(
                action_id=cue_id,
                turn_id=turn_id,
                prompt=prompt,
                duration_seconds=float(duration),
                completion_driven=finite,
            )
            self._cue = cue
            playback = self._playback
            session = (
                playback.session
                if playback is not None
                and self._stream_fence_token is None
                and self._output_session_id is not None
                and self._status.state
                in {
                    MotionDirectorState.IDLE,
                    MotionDirectorState.PREPARING_CUE,
                    MotionDirectorState.CUE,
                    MotionDirectorState.RETURNING_IDLE,
                }
                else None
            )
            if session is not None:
                try:
                    revision = session.set_prompt(prompt)
                except BaseException as exc:
                    prompt_error = exc
                else:
                    cue.bind_prompt_revision(
                        revision,
                        now=time.monotonic(),
                        timeout_seconds=self.cue_prepare_timeout_seconds,
                    )
            self._status = replace(
                self._status,
                state=(
                    MotionDirectorState.PREPARING_CUE
                    if session is not None
                    else (
                        MotionDirectorState.STOPPING
                        if playback is not None
                        else (
                            MotionDirectorState.STARTING
                            if self._output_session_id is not None
                            else MotionDirectorState.WAITING_OUTPUT
                        )
                    )
                ),
                active_prompt=prompt,
                active_cue_id=cue_id,
                active_turn_id=turn_id,
                cue_prompt_revision=cue.prompt_revision,
                cue_prepare_deadline_monotonic=cue.prepare_deadline_monotonic,
                cue_started_monotonic=None,
                cue_deadline_monotonic=None,
                idle_prompt_revision=None,
                heartbeat_monotonic=time.monotonic(),
                last_error=None,
            )
            self._condition.notify_all()

        if prompt_error is not None:
            self._fail_stream("prompt update failed", prompt_error)
            raise prompt_error
        return cue_id

