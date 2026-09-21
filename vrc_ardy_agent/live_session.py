from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class LiveSessionStatus:
    running: bool = False
    paused: bool = False
    requested_prompt: str | None = None
    requested_prompt_revision: int = 0
    generated_prompt: str | None = None
    generated_prompt_revision: int = 0
    playing_prompt: str | None = None
    playing_prompt_revision: int = 0
    buffered_frames: int = 0
    played_frames: int = 0
    generated_chunks: int = 0
    generation_in_progress: bool = False
    last_generation_seconds: float | None = None
    underruns: int = 0
    heartbeat_monotonic: float = 0.0


@dataclass(frozen=True, slots=True)
class PromptPlaybackStarted:
    revision: int
    prompt: str
    played_frames: int
    started_monotonic: float


@dataclass(frozen=True, slots=True)
class _PromptRequest:
    revision: int
    prompt: str


@dataclass(frozen=True, slots=True)
class _BufferedMotionFrame:
    frame: Any
    prompt_request: _PromptRequest


class LiveArdySession:
    """Headless ARDY auto-replan loop feeding a frame sink.

    This mirrors the official interactive demo's behavior rather than running
    independent clip generations: one resident runtime owns the rolling native
    history, one generation worker appends a horizon when playback approaches
    the buffered end, and playback consumes frames at the model's native FPS.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        mapper: Any,
        sink: Any,
        replan_threshold_frames: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.mapper = mapper
        self.sink = sink
        horizon = int(runtime.horizon_frames)
        if horizon <= 0:
            raise ValueError("runtime horizon must be positive")
        if replan_threshold_frames is None:
            # The first real MPS integration run prepared a bridge-ready
            # 40-frame horizon in about 0.4 s. Trigger at 40% of the horizon
            # (16 frames / 0.8 s for Core-20FPS) to keep useful scheduling
            # headroom instead of racing generation against the buffer edge.
            replan_threshold_frames = min(
                horizon - 1,
                max(1, int(round(horizon * 0.4))),
            )
        if not 0 <= replan_threshold_frames < horizon:
            raise ValueError("replan_threshold_frames must be in [0, horizon)")
        self.replan_threshold_frames = int(replan_threshold_frames)

        self._buffer: deque[_BufferedMotionFrame] = deque()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._producer_thread: threading.Thread | None = None
        self._next_prompt_revision = 0
        self._requested_prompt: _PromptRequest | None = None
        self._generating_prompt: _PromptRequest | None = None
        self._producer_error: BaseException | None = None
        self._started = False
        self._paused = False
        self._status = LiveSessionStatus()

    @property
    def status(self) -> LiveSessionStatus:
        with self._condition:
            return replace(self._status, buffered_frames=len(self._buffer))

    def start(self, prompt: str) -> int:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        with self._condition:
            if self._started:
                raise RuntimeError("live session is already started")
            self._started = True
            request = self._new_prompt_request_locked(prompt)
            self._requested_prompt = request
            self._status = replace(
                self._status,
                running=True,
                requested_prompt=prompt,
                requested_prompt_revision=request.revision,
                heartbeat_monotonic=time.monotonic(),
            )

        try:
            # Prefill one horizon synchronously so playback never starts from
            # an empty buffer. Later horizons are produced in the background.
            self.runtime.set_prompt(prompt)
            self._generating_prompt = request
            chunk = self.runtime.generate_next()
            frames = self.mapper.map_chunk(chunk)
            if not frames:
                raise RuntimeError("ARDY generated an empty first horizon")
            with self._condition:
                self._buffer.extend(
                    _BufferedMotionFrame(frame=frame, prompt_request=request)
                    for frame in frames
                )
                self._status = replace(
                    self._status,
                    generated_chunks=1,
                    generated_prompt=request.prompt,
                    generated_prompt_revision=request.revision,
                    last_generation_seconds=float(chunk.generation_seconds),
                    buffered_frames=len(self._buffer),
                )
        except BaseException:
            with self._condition:
                self._started = False
                self._status = replace(self._status, running=False)
            raise

        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="ardy-generation-worker",
            daemon=True,
        )
        self._producer_thread.start()
        return request.revision

    def set_prompt(self, prompt: str) -> int:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        with self._condition:
            if not self._started:
                raise RuntimeError("start the live session before changing its prompt")
            request = self._new_prompt_request_locked(prompt)
            self._requested_prompt = request
            self._paused = False
            self._status = replace(
                self._status,
                requested_prompt=prompt,
                requested_prompt_revision=request.revision,
                paused=False,
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()
            return request.revision

    def pause(self) -> None:
        with self._condition:
            if not self._started:
                raise RuntimeError("start the live session before pausing it")
            if self._paused:
                return
            self._paused = True
            self._status = replace(
                self._status,
                paused=True,
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()
        self.sink.neutralize_inputs()

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

    def run(
        self,
        *,
        prompt: str,
        duration_seconds: float | None = None,
        max_frames: int | None = None,
        realtime: bool = True,
        cancel_event: threading.Event | None = None,
        heartbeat_interval_seconds: float = 2.0,
        heartbeat: Callable[[LiveSessionStatus], None] | None = None,
        prompt_started: Callable[[PromptPlaybackStarted], None] | None = None,
    ) -> LiveSessionStatus:
        self.start(prompt)
        return self.run_started(
            duration_seconds=duration_seconds,
            max_frames=max_frames,
            realtime=realtime,
            cancel_event=cancel_event,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            heartbeat=heartbeat,
            prompt_started=prompt_started,
        )

    def run_started(
        self,
        *,
        duration_seconds: float | None = None,
        max_frames: int | None = None,
        realtime: bool = True,
        cancel_event: threading.Event | None = None,
        heartbeat_interval_seconds: float = 2.0,
        heartbeat: Callable[[LiveSessionStatus], None] | None = None,
        prompt_started: Callable[[PromptPlaybackStarted], None] | None = None,
    ) -> LiveSessionStatus:
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        with self._condition:
            if not self._started:
                raise RuntimeError("live session has not been started")

        fps = float(self.runtime.fps)
        if fps <= 0:
            raise ValueError("runtime FPS must be positive")

        started_at = time.perf_counter()
        next_frame_deadline = started_at
        next_heartbeat = time.monotonic() + heartbeat_interval_seconds
        playback_clock_needs_reset = False

        try:
            while not self._stop_event.is_set():
                if cancel_event is not None and cancel_event.is_set():
                    self.request_stop()
                    break
                current = self.status
                if max_frames is not None and current.played_frames >= max_frames:
                    break
                if duration_seconds is not None and time.perf_counter() - started_at >= duration_seconds:
                    break

                with self._condition:
                    if self._paused:
                        playback_clock_needs_reset = True
                        self._condition.wait(timeout=0.1)
                        now = time.monotonic()
                        if heartbeat is not None and now >= next_heartbeat:
                            heartbeat(self.status)
                            next_heartbeat = now + heartbeat_interval_seconds
                        continue

                if realtime and playback_clock_needs_reset:
                    next_frame_deadline = time.perf_counter()
                    playback_clock_needs_reset = False

                buffered = self._pop_next_frame()
                self.sink.send(buffered.frame)
                playback_event: PromptPlaybackStarted | None = None
                with self._condition:
                    played_frames = self._status.played_frames + 1
                    request = buffered.prompt_request
                    if request.revision != self._status.playing_prompt_revision:
                        playback_event = PromptPlaybackStarted(
                            revision=request.revision,
                            prompt=request.prompt,
                            played_frames=played_frames,
                            started_monotonic=time.monotonic(),
                        )
                    self._status = replace(
                        self._status,
                        played_frames=played_frames,
                        playing_prompt=request.prompt,
                        playing_prompt_revision=request.revision,
                        buffered_frames=len(self._buffer),
                        heartbeat_monotonic=time.monotonic(),
                    )
                if playback_event is not None and prompt_started is not None:
                    prompt_started(playback_event)

                if realtime:
                    next_frame_deadline += 1.0 / fps
                    delay = next_frame_deadline - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)

                now = time.monotonic()
                if heartbeat is not None and now >= next_heartbeat:
                    heartbeat(self.status)
                    next_heartbeat = now + heartbeat_interval_seconds
        finally:
            self.stop()

        return self.status

    def stop(self) -> None:
        self.request_stop()
        thread = self._producer_thread
        if thread is not None and thread is not threading.current_thread():
            # ARDY generation itself is not cancellable. Abandoning this worker
            # would let CharacterSupervisor start the next action against the
            # same resident model while the old generation still mutates its
            # history, so shutdown must wait for the authoritative worker end.
            thread.join()
        self.sink.close()
        with self._condition:
            self._started = False
            self._paused = False
            self._status = replace(
                self._status,
                running=False,
                paused=False,
                generation_in_progress=False,
                buffered_frames=len(self._buffer),
                heartbeat_monotonic=time.monotonic(),
            )

    def _producer_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_event.is_set()
                        or (not self._paused and len(self._buffer) <= self.replan_threshold_frames)
                    )
                    if self._stop_event.is_set():
                        return
                    request = self._requested_prompt
                    self._status = replace(self._status, generation_in_progress=True)

                if request is None:
                    raise RuntimeError("live session has no prompt")
                if (
                    self._generating_prompt is None
                    or request.revision != self._generating_prompt.revision
                ):
                    self.runtime.set_prompt(request.prompt)
                    self._generating_prompt = request

                chunk = self.runtime.generate_next()
                frames = self.mapper.map_chunk(chunk)
                if not frames:
                    raise RuntimeError("ARDY generated an empty horizon")

                with self._condition:
                    self._buffer.extend(
                        _BufferedMotionFrame(frame=frame, prompt_request=request)
                        for frame in frames
                    )
                    self._status = replace(
                        self._status,
                        generated_chunks=self._status.generated_chunks + 1,
                        generated_prompt=request.prompt,
                        generated_prompt_revision=request.revision,
                        generation_in_progress=False,
                        last_generation_seconds=float(chunk.generation_seconds),
                        buffered_frames=len(self._buffer),
                        heartbeat_monotonic=time.monotonic(),
                    )
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._producer_error = exc
                self._status = replace(self._status, generation_in_progress=False)
                self._condition.notify_all()

    def _pop_next_frame(self) -> _BufferedMotionFrame:
        counted_underrun = False
        with self._condition:
            while not self._buffer:
                if self._producer_error is not None:
                    raise RuntimeError("ARDY generation worker failed") from self._producer_error
                if self._stop_event.is_set():
                    raise RuntimeError("live session stopped while waiting for a frame")
                if not counted_underrun:
                    self._status = replace(self._status, underruns=self._status.underruns + 1)
                    counted_underrun = True
                self._condition.notify_all()
                self._condition.wait(timeout=0.1)

            frame = self._buffer.popleft()
            self._status = replace(self._status, buffered_frames=len(self._buffer))
            self._condition.notify_all()
            return frame

    def _new_prompt_request_locked(self, prompt: str) -> _PromptRequest:
        self._next_prompt_revision += 1
        return _PromptRequest(revision=self._next_prompt_revision, prompt=prompt)
