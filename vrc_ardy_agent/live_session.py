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
    prompt: str | None = None
    buffered_frames: int = 0
    played_frames: int = 0
    generated_chunks: int = 0
    generation_in_progress: bool = False
    last_generation_seconds: float | None = None
    underruns: int = 0
    heartbeat_monotonic: float = 0.0


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

        self._buffer: deque[Any] = deque()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._producer_thread: threading.Thread | None = None
        self._desired_prompt: str | None = None
        self._active_prompt: str | None = None
        self._producer_error: BaseException | None = None
        self._started = False
        self._paused = False
        self._status = LiveSessionStatus()

    @property
    def status(self) -> LiveSessionStatus:
        with self._condition:
            return replace(self._status, buffered_frames=len(self._buffer))

    def start(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        with self._condition:
            if self._started:
                raise RuntimeError("live session is already started")
            self._started = True
            self._desired_prompt = prompt
            self._status = replace(
                self._status,
                running=True,
                prompt=prompt,
                heartbeat_monotonic=time.monotonic(),
            )

        try:
            # Prefill one horizon synchronously so playback never starts from
            # an empty buffer. Later horizons are produced in the background.
            self.runtime.set_prompt(prompt)
            self._active_prompt = prompt
            chunk = self.runtime.generate_next()
            frames = self.mapper.map_chunk(chunk)
            if not frames:
                raise RuntimeError("ARDY generated an empty first horizon")
            with self._condition:
                self._buffer.extend(frames)
                self._status = replace(
                    self._status,
                    generated_chunks=1,
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

    def set_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        with self._condition:
            if not self._started:
                raise RuntimeError("start the live session before changing its prompt")
            self._desired_prompt = prompt
            self._paused = False
            self._status = replace(self._status, prompt=prompt, paused=False)
            self._condition.notify_all()

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
        heartbeat_interval_seconds: float = 2.0,
        heartbeat: Callable[[LiveSessionStatus], None] | None = None,
    ) -> LiveSessionStatus:
        self.start(prompt)
        return self.run_started(
            duration_seconds=duration_seconds,
            max_frames=max_frames,
            realtime=realtime,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            heartbeat=heartbeat,
        )

    def run_started(
        self,
        *,
        duration_seconds: float | None = None,
        max_frames: int | None = None,
        realtime: bool = True,
        heartbeat_interval_seconds: float = 2.0,
        heartbeat: Callable[[LiveSessionStatus], None] | None = None,
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

                frame = self._pop_next_frame()
                self.sink.send(frame)
                with self._condition:
                    self._status = replace(
                        self._status,
                        played_frames=self._status.played_frames + 1,
                        buffered_frames=len(self._buffer),
                        heartbeat_monotonic=time.monotonic(),
                    )

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
            thread.join(timeout=5.0)
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
                    prompt = self._desired_prompt
                    self._status = replace(self._status, generation_in_progress=True)

                if prompt is None:
                    raise RuntimeError("live session has no prompt")
                if prompt != self._active_prompt:
                    self.runtime.set_prompt(prompt)
                    self._active_prompt = prompt

                chunk = self.runtime.generate_next()
                frames = self.mapper.map_chunk(chunk)
                if not frames:
                    raise RuntimeError("ARDY generated an empty horizon")

                with self._condition:
                    self._buffer.extend(frames)
                    self._status = replace(
                        self._status,
                        generated_chunks=self._status.generated_chunks + 1,
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

    def _pop_next_frame(self) -> Any:
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