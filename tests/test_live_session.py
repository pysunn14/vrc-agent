from __future__ import annotations

import threading
import time
import unittest

from vrc_ardy_agent.live_session import LiveArdySession


class _FakeChunk:
    def __init__(self, start: int, count: int = 4) -> None:
        self.values = list(range(start, start + count))
        self.generation_seconds = 0.01


class _FakeRuntime:
    fps = 20.0
    horizon_frames = 4

    def __init__(self) -> None:
        self.prompt = None
        self.next_value = 0
        self.generate_calls = 0
        self.prompt_calls: list[str] = []

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt
        self.prompt_calls.append(prompt)

    def generate_next(self) -> _FakeChunk:
        self.generate_calls += 1
        chunk = _FakeChunk(self.next_value, self.horizon_frames)
        self.next_value += self.horizon_frames
        return chunk


class _FakeMapper:
    def map_chunk(self, chunk: _FakeChunk) -> list[int]:
        return chunk.values


class _FakeSink:
    def __init__(self) -> None:
        self.frames: list[int] = []
        self.closed = False
        self.neutralize_calls = 0

    def send(self, frame: int) -> None:
        self.frames.append(frame)

    def neutralize_inputs(self) -> None:
        self.neutralize_calls += 1

    def close(self) -> None:
        self.closed = True


class LiveArdySessionTests(unittest.TestCase):
    def test_replans_in_background_and_plays_consecutive_frames(self):
        runtime = _FakeRuntime()
        sink = _FakeSink()
        session = LiveArdySession(
            runtime=runtime,
            mapper=_FakeMapper(),
            sink=sink,
            replan_threshold_frames=1,
        )

        session.run(prompt="walk", max_frames=10, realtime=False)

        self.assertEqual(sink.frames, list(range(10)))
        self.assertGreaterEqual(runtime.generate_calls, 3)
        self.assertEqual(runtime.prompt_calls[0], "walk")
        self.assertTrue(sink.closed)
        self.assertEqual(session.status.played_frames, 10)
        self.assertGreaterEqual(session.status.generated_chunks, 3)

    def test_prompt_update_is_applied_by_the_generation_worker(self):
        runtime = _FakeRuntime()
        session = LiveArdySession(
            runtime=runtime,
            mapper=_FakeMapper(),
            sink=_FakeSink(),
            replan_threshold_frames=1,
        )

        session.start("walk")
        session.set_prompt("wave while walking")
        session.run_started(max_frames=6, realtime=False)

        self.assertIn("wave while walking", runtime.prompt_calls)
        self.assertEqual(session.status.prompt, "wave while walking")

    def test_pause_neutralizes_inputs_and_next_prompt_resumes(self):
        runtime = _FakeRuntime()
        sink = _FakeSink()
        session = LiveArdySession(
            runtime=runtime,
            mapper=_FakeMapper(),
            sink=sink,
            replan_threshold_frames=1,
        )

        session.start("walk")
        session.pause()

        self.assertTrue(session.status.paused)
        self.assertEqual(sink.neutralize_calls, 1)

        session.set_prompt("wave")

        self.assertFalse(session.status.paused)
        self.assertEqual(session.status.prompt, "wave")
        session.request_stop()
        session.stop()

    def test_resume_resets_realtime_deadline_instead_of_catching_up(self):
        runtime = _FakeRuntime()

        class PausingSink(_FakeSink):
            def __init__(self) -> None:
                super().__init__()
                self.session: LiveArdySession | None = None
                self.send_times: list[float] = []

            def send(self, frame: int) -> None:
                super().send(frame)
                self.send_times.append(time.perf_counter())
                if len(self.send_times) == 1:
                    assert self.session is not None
                    self.session.pause()
                    threading.Timer(0.2, lambda: self.session.set_prompt("wave")).start()

        sink = PausingSink()
        session = LiveArdySession(
            runtime=runtime,
            mapper=_FakeMapper(),
            sink=sink,
            replan_threshold_frames=1,
        )
        sink.session = session

        session.run(prompt="walk", max_frames=3, realtime=True)

        self.assertEqual(len(sink.send_times), 3)
        self.assertGreaterEqual(sink.send_times[2] - sink.send_times[1], 0.03)


if __name__ == "__main__":
    unittest.main()