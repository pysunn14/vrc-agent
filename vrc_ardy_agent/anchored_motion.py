"""A resident output stream with fixed idle, finite presets, and on-demand ARDY."""
from dataclasses import replace
import math
import threading
import time

from .live_session import LiveArdySession, PromptPlaybackStarted, _BufferedMotionFrame
from .motion_assets import MotionAssets
from .motion_stream import CALIBRATED_IDLE
from .pose_transition import PoseTransition, validate_pose


class AnchoredMotionSession(LiveArdySession):
    def __init__(self, *, assets: MotionAssets, transition_seconds=1., **kwargs):
        super().__init__(**kwargs)
        if not math.isfinite(transition_seconds) or transition_seconds <= 0:
            raise ValueError('transition duration must be finite and positive')
        if float(self.runtime.fps) != assets.idle.fps:
            raise ValueError('runtime and calibrated pose FPS differ')
        self.assets = assets
        self._return_frames = max(1, math.ceil(transition_seconds * assets.idle.fps))
        self._last_sent = assets.idle
        self._last_request = None
        self._selected_revision = 0
        self._preset_index = 0
        self._return = None
        self._return_index = 0
        self._completed_revision = 0
        self._entry = None
        self._entry_index = 0
        self._needs_entry = False

    @property
    def completed_revision(self):
        with self._condition:
            return self._completed_revision

    def start(self, prompt):
        with self._condition:
            if self._started:
                raise RuntimeError('session already started')
            self._started = True
            revision = self.set_prompt(prompt)
            self._producer_thread = threading.Thread(target=self._producer_loop,
                                                     name='ardy-on-demand', daemon=True)
            self._producer_thread.start()
            return revision

    def set_prompt(self, prompt):
        prompt = prompt.strip()
        if prompt.startswith('preset:') and (prompt[7:] not in self.assets.behaviors
                or self.assets.behaviors[prompt[7:]].source == 'ardy'):
            raise ValueError('unknown registered motion')
        with self._condition:
            revision = super().set_prompt(prompt)
            # A revision change invalidates queued and in-flight generation alike.
            self._buffer.clear()
            self._producer_error = None
            self._condition.notify_all()
            return revision

    def _producer_loop(self):
        generating_revision = None
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._stop_event.is_set() or (
                    self._requested_prompt is not None
                    and self._requested_prompt.prompt != CALIBRATED_IDLE
                    and not self._requested_prompt.prompt.startswith('preset:')
                    and self._producer_error is None
                    and len(self._buffer) <= self.replan_threshold_frames))
                if self._stop_event.is_set():
                    return
                request = self._requested_prompt
                self._status = replace(self._status, generation_in_progress=True)
            try:
                if generating_revision != request.revision:
                    # Idle isn't synthesized, so a new generated action must not
                    # inherit history claiming the body still occupies its old pose.
                    self.runtime.clear_history()
                    self.mapper.reset()
                    self.runtime.set_prompt(request.prompt)
                    generating_revision = request.revision
                chunk = self.runtime.generate_next()
                frames = self.mapper.map_chunk(chunk)
                if not frames:
                    raise RuntimeError('ARDY generated an empty horizon')
                frames = [replace(f, locomotion_x=0., locomotion_y=0., locomotion_turn=0., face=self.assets.idle.face)
                          for f in frames]
                for frame in frames:
                    validate_pose(frame)
                    if (frame.fps != self.assets.idle.fps or frame.scale != self.assets.idle.scale
                            or frame.tracker_activation != self.assets.idle.tracker_activation):
                        raise ValueError('generated pose does not match the idle tracking configuration')
                with self._condition:
                    if request == self._requested_prompt:
                        self._buffer.extend(_BufferedMotionFrame(f, request) for f in frames)
                    self._status = replace(self._status,
                        generated_chunks=self._status.generated_chunks + 1,
                        generated_prompt=request.prompt, generated_prompt_revision=request.revision,
                        last_generation_seconds=float(chunk.generation_seconds))
            except BaseException as exc:
                with self._condition:
                    if request == self._requested_prompt:
                        self._producer_error = exc
            finally:
                with self._condition:
                    self._status = replace(self._status, generation_in_progress=False)
                    self._condition.notify_all()

    def _select_locked(self):
        request = self._requested_prompt
        if request.revision != self._selected_revision:
            self._selected_revision = request.revision
            self._preset_index = self._return_index = 0
            self._return = None
            self._entry = None
            self._entry_index = 0
            self._needs_entry = True
        if self._producer_error is not None:
            raise RuntimeError('ARDY generation failed') from self._producer_error
        idle = self.assets.idle
        if request.prompt == CALIBRATED_IDLE:
            if self._return is None:
                # Stop world movement and facial cues immediately. Only pose
                # targets interpolate; easing movement here would overshoot walks.
                source = replace(self._last_sent, locomotion_x=0., locomotion_y=0.,
                                 locomotion_turn=0.)
                self._return = PoseTransition(source, idle)
            self._return_index = min(self._return_index + 1, self._return_frames)
            frame = self._return.at(self._return_index / self._return_frames)
            observed = request if self._return_index == self._return_frames else self._last_request
            return frame, observed, False
        if request.prompt.startswith('preset:'):
            key = request.prompt[7:]
            spec = self.assets.behaviors[key]
            if spec.finite:
                frames = self.assets.clips[key].frames
                index = min(self._preset_index, len(frames) - 1)
                frame, observed, accepted = self._enter_locked(frames[index], request)
                return frame, observed, accepted and self._preset_index < len(frames)
            frame, observed, accepted = self._enter_locked(idle, request)
            x, y, turn = spec.velocity if accepted else (0., 0., 0.)
            return replace(frame, locomotion_x=x, locomotion_y=y, locomotion_turn=turn), observed, False
        if self._buffer:
            buffered = self._buffer[0]
            frame, observed, accepted = self._enter_locked(buffered.frame, buffered.prompt_request)
            if accepted:
                self._buffer.popleft()
                self._condition.notify_all()
            return frame, observed, False
        # Keep tracking alive during generation latency. This is an observable
        # hold (not successful cue playback), never a fabricated generated frame.
        self._status = replace(self._status, underruns=self._status.underruns + 1)
        return replace(self._last_sent, face=self.assets.idle.face, locomotion_x=0., locomotion_y=0., locomotion_turn=0.), self._last_request, False

    def _enter_locked(self, frame, request):
        if not self._needs_entry:
            return frame, request, True
        if self._entry is None:
            source = replace(self._last_sent, locomotion_x=0., locomotion_y=0., locomotion_turn=0.)
            if source == frame:
                self._needs_entry = False
                return frame, request, True
            self._entry = PoseTransition(source, frame)
        self._entry_index += 1
        if self._entry_index >= self._return_frames:
            self._needs_entry = False
            return frame, request, True
        # Entry blending isn't clip playback. The cue clock/revision starts only
        # after its first target has reached the output stream.
        return self._entry.at(self._entry_index / self._return_frames), self._last_request, False

    def send_next(self):
        """One authoritative output step, also used by headless regression tests."""
        with self._condition:
            if not self._started or self._stop_event.is_set():
                raise RuntimeError('session is not running')
            frame, request, advance_clip = self._select_locked()
            # Serialize requests with accepted sends: an old revision cannot
            # slip out after a newer request has returned to its caller.
            self.sink.send(frame)
            self._last_sent = frame
            event = None
            count = self._status.played_frames + 1
            if request is not None:
                self._last_request = request
                if self._status.playing_prompt_revision != request.revision:
                    event = PromptPlaybackStarted(request.revision, request.prompt, count, time.monotonic())
                self._status = replace(self._status, playing_prompt=request.prompt,
                                       playing_prompt_revision=request.revision)
            if advance_clip:
                self._preset_index += 1
                if self._preset_index == len(self.assets.clips[request.prompt[7:]].frames):
                    self._completed_revision = request.revision
            self._status = replace(self._status, running=True, played_frames=count,
                                   heartbeat_monotonic=time.monotonic())
            return event

    def run_started(self, *, realtime=True, cancel_event=None, heartbeat=None,
                    prompt_started=None):
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set():
                if cancel_event is not None and cancel_event.is_set():
                    break
                if realtime and self._stop_event.wait(max(0., deadline-time.monotonic())):
                    break
                event = self.send_next()
                if event is not None and prompt_started is not None:
                    prompt_started(event)
                if heartbeat is not None:
                    heartbeat(self.status)
                deadline = max(deadline + 1/self.assets.idle.fps, time.monotonic())
        finally:
            self.stop()
        return self.status
