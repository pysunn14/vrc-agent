"""One output owner for finite motion, return, and indefinite idle keepalive."""
from dataclasses import dataclass, replace
import math
import threading
import time

from .pose_transition import PoseTransition, validate_pose


@dataclass(frozen=True)
class ReplayStatus:
    state: str = 'ready'
    motion_frames_sent: int = 0
    motion_frames_total: int = 0
    transition_frames_sent: int = 0
    transition_frames_total: int = 0
    idle_frames_sent: int = 0
    heartbeat_monotonic: float = 0.
    last_error: str | None = None


class MotionReplayRunner:
    def __init__(self, *, frames, idle, sink, transition_seconds=1., clock=time.monotonic):
        if not math.isfinite(transition_seconds) or transition_seconds <= 0:
            raise ValueError('transition_seconds must be finite and positive')
        self.frames = tuple(frames)
        if not self.frames:
            raise ValueError('motion clip must not be empty')
        validate_pose(idle)
        if any(getattr(idle, name) != 0 for name in ('locomotion_x', 'locomotion_y', 'locomotion_turn')):
            raise ValueError('idle locomotion must be zero')
        for frame in self.frames:
            validate_pose(frame)
            if frame.fps != idle.fps or frame.tracker_activation != idle.tracker_activation:
                raise ValueError('clip and idle must share fps and tracker activation')
        self.idle, self.sink, self.clock = idle, sink, clock
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._status = ReplayStatus(
            motion_frames_total=len(self.frames),
            transition_frames_total=max(1, math.ceil(transition_seconds * idle.fps)),
        )

    @property
    def status(self):
        with self._lock:
            return self._status

    def _update(self, **values):
        with self._lock:
            self._status = replace(self._status, heartbeat_monotonic=self.clock(), **values)

    def request_stop(self):
        """Explicit shutdown, distinct from normal clip completion."""
        self._stop.set()

    def run(self, *, realtime=True, heartbeat=None):
        with self._lock:
            if self._started:
                raise RuntimeError('motion replay runner is one-shot')
            self._started = True
        deadline = self.clock()
        last_sent = None
        transition = None
        try:
            while not self._stop.is_set():
                status = self.status
                if status.motion_frames_sent < len(self.frames):
                    state, counter = 'playing', 'motion_frames_sent'
                    frame = self.frames[status.motion_frames_sent]
                elif status.transition_frames_sent < status.transition_frames_total:
                    state, counter = 'returning_idle', 'transition_frames_sent'
                    if transition is None:
                        # Only a successfully sent frame can anchor the return.
                        transition = PoseTransition(last_sent, self.idle)
                    frame = transition.at((status.transition_frames_sent + 1) / status.transition_frames_total)
                else:
                    state, counter, frame = 'idle', 'idle_frames_sent', self.idle
                if realtime and self._stop.wait(max(0., deadline - self.clock())):
                    break
                self.sink.send(frame)
                last_sent = frame
                self._update(state=state, **{counter: getattr(status, counter) + 1})
                if heartbeat:
                    heartbeat(self.status)
                # Never burst stale frames after a stalled send/callback. Keep
                # the clip intact and stretch playback instead of skipping it.
                deadline = max(deadline + 1 / self.idle.fps, self.clock())
        except Exception as exc:
            self._update(state='failed', last_error=str(exc))
            raise
        finally:
            try:
                self.sink.close()
            except Exception as exc:
                previous = self.status.last_error
                self._update(state='failed', last_error=f'{previous}; close: {exc}' if previous else str(exc))
                raise
            finally:
                if self.status.state != 'failed':
                    self._update(state='stopped')
        return self.status
