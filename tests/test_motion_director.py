from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import unittest

from vrc_ardy_agent.action_contracts import (
    ArdyMotionAction,
    ControlResource,
    ResourceLease,
)
from vrc_ardy_agent.motion_director import MotionDirector
from vrc_ardy_agent.live_session import PromptPlaybackStarted
from vrc_ardy_agent.motion_stream import MotionDirectorState


class _FakeRuntime:
    horizon_frames = 40
    fps = 20.0

    def __init__(self) -> None:
        self.clear_history_calls = 0

    def clear_history(self) -> None:
        self.clear_history_calls += 1


class _FakeMapper:
    pass


@dataclass(frozen=True)
class _FakeSessionStatus:
    played_frames: int = 3
    generated_chunks: int = 2
    underruns: int = 0
    last_generation_seconds: float = 0.1
    playing_prompt: str | None = None
    playing_prompt_revision: int = 0


class _FakeSession:
    instances: list["_FakeSession"] = []

    def __init__(self, *, runtime, mapper, sink, replan_threshold_frames=None) -> None:
        self.runtime = runtime
        self.mapper = mapper
        self.sink = sink
        self.replan_threshold_frames = replan_threshold_frames
        self.prompts: list[str] = []
        self.started = threading.Event()
        self.stop_requests = 0
        self.stop_calls = 0
        self.next_revision = 0
        self.prompt_revisions: list[int] = []
        self.prompt_started = None
        self.__class__.instances.append(self)

    def start(self, prompt: str) -> int:
        self.prompts.append(prompt)
        self.next_revision += 1
        self.prompt_revisions.append(self.next_revision)
        self.started.set()
        return self.next_revision

    def run_started(
        self,
        *,
        realtime: bool,
        cancel_event: threading.Event,
        heartbeat,
        prompt_started,
    ) -> _FakeSessionStatus:
        del realtime
        self.prompt_started = prompt_started
        self.emit_prompt_started()
        heartbeat(_FakeSessionStatus())
        cancel_event.wait(timeout=2.0)
        self.stop()
        return _FakeSessionStatus()

    def set_prompt(self, prompt: str) -> int:
        self.prompts.append(prompt)
        self.next_revision += 1
        self.prompt_revisions.append(self.next_revision)
        return self.next_revision

    def emit_prompt_started(self, revision: int | None = None) -> None:
        if self.prompt_started is None:
            return
        selected = self.prompt_revisions[-1] if revision is None else revision
        prompt = self.prompts[self.prompt_revisions.index(selected)]
        self.prompt_started(
            PromptPlaybackStarted(
                revision=selected,
                prompt=prompt,
                played_frames=1,
                started_monotonic=time.monotonic(),
            )
        )

    def request_stop(self) -> None:
        self.stop_requests += 1

    def stop(self) -> None:
        self.stop_calls += 1


class _DelayedStopSession(_FakeSession):
    release_stop = threading.Event()

    def run_started(
        self,
        *,
        realtime: bool,
        cancel_event: threading.Event,
        heartbeat,
        prompt_started,
    ) -> _FakeSessionStatus:
        del realtime
        self.prompt_started = prompt_started
        self.emit_prompt_started()
        heartbeat(_FakeSessionStatus())
        cancel_event.wait(timeout=2.0)
        self.release_stop.wait(timeout=2.0)
        self.stop()
        return _FakeSessionStatus()


class _RejectingPromptSession(_FakeSession):
    def set_prompt(self, prompt: str) -> int:
        raise RuntimeError(f"cannot apply prompt: {prompt}")


class _RecordingOutput:
    def __init__(self) -> None:
        self.authorized: list[ResourceLease] = []
        self.neutralized: list[tuple[ControlResource, int]] = []
        self.pose_calls: list[tuple[ResourceLease, object, int, int]] = []

    def authorize(self, lease: ResourceLease) -> None:
        self.authorized.append(lease)

    def neutralize(self, resource: ControlResource, lease_token: int) -> None:
        self.neutralized.append((resource, lease_token))

    def send_pose(
        self,
        lease: ResourceLease,
        frame: object,
        *,
        sequence: int,
        ttl_ms: int,
    ) -> None:
        self.pose_calls.append((lease, frame, sequence, ttl_ms))


class _BlockingOutput(_RecordingOutput):
    def __init__(self) -> None:
        super().__init__()
        self.authorize_started = threading.Event()
        self.release_authorize = threading.Event()

    def authorize(self, lease: ResourceLease) -> None:
        self.authorize_started.set()
        self.release_authorize.wait(timeout=2.0)
        super().authorize(lease)


class MotionDirectorTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSession.instances.clear()
        _DelayedStopSession.release_stop.clear()
        self.runtime = _FakeRuntime()
        self.output = _RecordingOutput()
        self.director = MotionDirector(
            runtime=self.runtime,
            output=self.output,
            mapper_factory=_FakeMapper,
            session_factory=_FakeSession,
            idle_prompt="A girl stands naturally with subtle idle motion.",
            cue_poll_seconds=0.01,
            realtime=False,
        )
        self.addCleanup(self.director.stop)

    def _connect(self) -> _FakeSession:
        self.director.start()
        self.director.attach_output_session("session-1")
        self.assertTrue(
            self.director.wait_until_state(MotionDirectorState.IDLE, timeout=1.0)
        )
        self.assertEqual(len(_FakeSession.instances), 1)
        return _FakeSession.instances[0]

    def test_cue_returns_to_idle_without_recreating_session_or_clearing_history(self) -> None:
        session = self._connect()

        cue_id = self.director.submit_cue(
            ArdyMotionAction(
                prompt="A girl waves with her right hand.",
                duration_seconds=0.05,
            ),
            turn_id="turn-1",
        )

        self.assertTrue(cue_id)
        self.assertEqual(
            self.director.snapshot().state,
            MotionDirectorState.PREPARING_CUE,
        )
        self.assertIsNone(self.director.snapshot().cue_deadline_monotonic)
        self.assertEqual(session.prompts[-1], "A girl waves with her right hand.")
        session.emit_prompt_started()
        self.assertEqual(self.director.snapshot().state, MotionDirectorState.CUE)
        self.assertIsNotNone(self.director.snapshot().cue_started_monotonic)
        self.assertTrue(
            self.director.wait_until_state(
                MotionDirectorState.RETURNING_IDLE,
                timeout=1.0,
            )
        )
        session.emit_prompt_started()
        self.assertTrue(
            self.director.wait_until_state(MotionDirectorState.IDLE, timeout=1.0)
        )
        self.assertEqual(session.prompts[-1], self.director.idle_prompt)
        self.assertEqual(len(_FakeSession.instances), 1)
        self.assertEqual(self.runtime.clear_history_calls, 0)
        self.assertEqual(len(self.output.authorized), 1)

    def test_new_cue_replaces_old_cue_in_the_same_live_session(self) -> None:
        session = self._connect()

        first_id = self.director.submit_cue(
            ArdyMotionAction(prompt="A girl dances.", duration_seconds=1.0),
            turn_id="turn-1",
        )
        second_id = self.director.submit_cue(
            ArdyMotionAction(prompt="A girl waves.", duration_seconds=1.0),
            turn_id="turn-2",
        )

        self.assertNotEqual(first_id, second_id)
        self.assertEqual(
            session.prompts[-2:],
            ["A girl dances.", "A girl waves."],
        )
        self.assertEqual(len(_FakeSession.instances), 1)
        self.assertEqual(len(self.output.authorized), 1)
        self.assertEqual(self.director.snapshot().active_cue_id, second_id)

    def test_cue_duration_does_not_run_before_matching_frames_are_played(self) -> None:
        session = self._connect()
        self.director.submit_cue(
            ArdyMotionAction(prompt="A girl dances.", duration_seconds=0.05),
            turn_id="turn-1",
        )

        time.sleep(0.08)
        pending = self.director.snapshot()
        self.assertEqual(pending.state, MotionDirectorState.PREPARING_CUE)
        self.assertIsNone(pending.cue_started_monotonic)
        self.assertIsNone(pending.cue_deadline_monotonic)

        session.emit_prompt_started()
        playing = self.director.snapshot()
        self.assertEqual(playing.state, MotionDirectorState.CUE)
        self.assertAlmostEqual(
            playing.cue_deadline_monotonic - playing.cue_started_monotonic,
            0.05,
            places=3,
        )

    def test_late_frames_from_replaced_cue_do_not_start_newer_cue_clock(self) -> None:
        session = self._connect()
        self.director.submit_cue(
            ArdyMotionAction(prompt="A girl dances.", duration_seconds=1.0),
            turn_id="turn-1",
        )
        old_revision = session.prompt_revisions[-1]
        second_id = self.director.submit_cue(
            ArdyMotionAction(prompt="A girl waves.", duration_seconds=1.0),
            turn_id="turn-2",
        )
        new_revision = session.prompt_revisions[-1]

        session.emit_prompt_started(old_revision)
        pending = self.director.snapshot()
        self.assertEqual(pending.state, MotionDirectorState.PREPARING_CUE)
        self.assertEqual(pending.active_cue_id, second_id)
        self.assertIsNone(pending.cue_deadline_monotonic)

        session.emit_prompt_started(new_revision)
        self.assertEqual(self.director.snapshot().state, MotionDirectorState.CUE)

    def test_cue_that_never_reaches_output_fails_observably(self) -> None:
        self.director.stop()
        _FakeSession.instances.clear()
        director = MotionDirector(
            runtime=self.runtime,
            output=self.output,
            mapper_factory=_FakeMapper,
            session_factory=_FakeSession,
            cue_poll_seconds=0.005,
            cue_prepare_timeout_seconds=0.05,
            realtime=False,
        )
        self.director = director
        self.addCleanup(director.stop)
        director.start()
        director.attach_output_session("session-timeout")
        self.assertTrue(
            director.wait_until_state(MotionDirectorState.IDLE, timeout=1.0)
        )

        director.submit_cue(
            ArdyMotionAction(prompt="A girl dances.", duration_seconds=1.0),
            turn_id="turn-timeout",
        )

        self.assertTrue(
            director.wait_until_state(MotionDirectorState.FAILED, timeout=1.0)
        )
        self.assertIn("cue playback did not start", director.snapshot().last_error or "")

    def test_halt_fences_output_and_clears_history_after_stream_stops(self) -> None:
        session = self._connect()
        active_lease = self.output.authorized[-1]

        result = self.director.halt(reason="fast_stop")

        self.assertGreater(result.lease_token, active_lease.token)
        self.assertEqual(
            self.output.neutralized[-1],
            (ControlResource.FULL_BODY_POSE, result.lease_token),
        )
        self.assertTrue(
            self.director.wait_until_state(MotionDirectorState.HALTED, timeout=1.0)
        )
        self.assertGreaterEqual(session.stop_requests, 1)
        self.assertEqual(self.runtime.clear_history_calls, 1)

    def test_repeated_halt_is_idempotent_while_stream_is_stopping(self) -> None:
        self._connect()

        first = self.director.halt(reason="fast_stop")
        second = self.director.halt(reason="fast_stop")

        self.assertEqual(second.lease_token, first.lease_token)
        self.assertEqual(
            self.output.neutralized,
            [(ControlResource.FULL_BODY_POSE, first.lease_token)],
        )

    def test_new_cue_restarts_stream_after_explicit_halt(self) -> None:
        self._connect()
        self.director.halt(reason="fast_stop")
        self.assertTrue(
            self.director.wait_until_state(MotionDirectorState.HALTED, timeout=1.0)
        )

        cue_id = self.director.submit_cue(
            ArdyMotionAction(prompt="A girl nods.", duration_seconds=1.0),
            turn_id="turn-2",
        )

        self.assertTrue(cue_id)
        self.assertTrue(
            self.director.wait_until_state(MotionDirectorState.CUE, timeout=1.0)
        )
        self.assertEqual(len(_FakeSession.instances), 2)
        self.assertEqual(len(self.output.authorized), 2)
        self.assertGreater(
            self.output.authorized[-1].token,
            self.output.authorized[0].token,
        )

    def test_cue_queued_during_stop_never_updates_the_fenced_session(self) -> None:
        self.director.stop()
        _FakeSession.instances.clear()
        director = MotionDirector(
            runtime=self.runtime,
            output=self.output,
            mapper_factory=_FakeMapper,
            session_factory=_DelayedStopSession,
            cue_poll_seconds=0.01,
            realtime=False,
        )
        self.director = director
        self.addCleanup(director.stop)
        director.start()
        director.attach_output_session("session-queued-cue")
        self.assertTrue(
            director.wait_until_state(MotionDirectorState.IDLE, timeout=1.0)
        )
        first_session = _FakeSession.instances[0]

        director.halt(reason="fast_stop")
        cue_id = director.submit_cue(
            ArdyMotionAction(
                prompt="A girl waves after stopping.",
                duration_seconds=1.0,
            ),
            turn_id="turn-after-stop",
        )

        self.assertEqual(director.snapshot().state, MotionDirectorState.STOPPING)
        self.assertEqual(first_session.prompts, [director.idle_prompt])
        _DelayedStopSession.release_stop.set()
        self.assertTrue(
            director.wait_until_state(MotionDirectorState.CUE, timeout=1.0)
        )
        self.assertEqual(len(_FakeSession.instances), 2)
        self.assertEqual(
            _FakeSession.instances[1].prompts,
            ["A girl waves after stopping."],
        )
        self.assertEqual(director.snapshot().active_cue_id, cue_id)

    def test_output_sink_sequences_frames_under_one_stream_lease(self) -> None:
        session = self._connect()
        lease = self.output.authorized[-1]

        session.sink.send("frame-a")
        session.sink.send("frame-b")

        self.assertEqual(
            self.output.pose_calls,
            [
                (lease, "frame-a", 0, 2000),
                (lease, "frame-b", 1, 2000),
            ],
        )

    def test_prompt_update_failure_fences_stream_and_becomes_failed(self) -> None:
        self.director.stop()
        _FakeSession.instances.clear()
        director = MotionDirector(
            runtime=self.runtime,
            output=self.output,
            mapper_factory=_FakeMapper,
            session_factory=_RejectingPromptSession,
            cue_poll_seconds=0.01,
            realtime=False,
        )
        self.director = director
        self.addCleanup(director.stop)
        director.start()
        director.attach_output_session("session-prompt-failure")
        self.assertTrue(
            director.wait_until_state(MotionDirectorState.IDLE, timeout=1.0)
        )
        active_lease = self.output.authorized[-1]

        with self.assertRaisesRegex(RuntimeError, "cannot apply prompt"):
            director.submit_cue(
                ArdyMotionAction(
                    prompt="A girl waves.",
                    duration_seconds=1.0,
                ),
                turn_id="turn-failure",
            )

        self.assertTrue(
            director.wait_until_state(MotionDirectorState.FAILED, timeout=1.0)
        )
        status = director.snapshot()
        self.assertIsNone(status.active_cue_id)
        self.assertIn("prompt update failed", status.last_error or "")
        self.assertGreater(status.lease_token, active_lease.token)
        self.assertEqual(
            self.output.neutralized[-1],
            (ControlResource.FULL_BODY_POSE, status.lease_token),
        )

    def test_detach_during_authorization_does_not_start_an_orphan_session(self) -> None:
        self.director.stop()
        _FakeSession.instances.clear()
        output = _BlockingOutput()
        director = MotionDirector(
            runtime=self.runtime,
            output=output,
            mapper_factory=_FakeMapper,
            session_factory=_FakeSession,
            cue_poll_seconds=0.01,
            realtime=False,
        )
        self.director = director
        self.addCleanup(director.stop)
        director.start()
        director.attach_output_session("session-starting")
        self.assertTrue(output.authorize_started.wait(timeout=1.0))

        self.assertTrue(
            director.detach_output_session(
                "session-starting",
                reason="socket closed",
            )
        )
        output.release_authorize.set()

        self.assertTrue(
            director.wait_until_state(
                MotionDirectorState.WAITING_OUTPUT,
                timeout=1.0,
            )
        )
        self.assertEqual(_FakeSession.instances, [])
        self.assertEqual(self.runtime.clear_history_calls, 0)

    def test_detach_without_a_stream_preserves_authoritative_halted_state(self) -> None:
        self.director.start()
        self.director.halt(reason="operator")
        self.director.attach_output_session("session-halted")

        self.assertTrue(
            self.director.detach_output_session(
                "session-halted",
                reason="socket closed",
            )
        )

        self.assertEqual(
            self.director.snapshot().state,
            MotionDirectorState.HALTED,
        )


if __name__ == "__main__":
    unittest.main()
