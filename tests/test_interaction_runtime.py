from __future__ import annotations

import threading
import unittest

from vrc_ardy_agent.action_contracts import (
    ActionBundle,
    ActionState,
    ActionType,
    ArdyMotionAction,
    ControlResource,
    ExecutionCommand,
    SayAction,
)
from vrc_ardy_agent.character_supervisor import CharacterSupervisor
from vrc_ardy_agent.interaction_runtime import (
    InteractionRuntime,
    TurnOutcomeStatus,
)
from vrc_ardy_agent.motion_stream import (
    MotionDirectorSnapshot,
    MotionDirectorState,
    MotionHaltResult,
)


class _BlockingExecutor:
    def __init__(self) -> None:
        self.commands: list[ExecutionCommand] = []
        self._condition = threading.Condition()

    def run(self, command, cancel_event, mark_running) -> None:
        with self._condition:
            self.commands.append(command)
            self._condition.notify_all()
        mark_running()
        cancel_event.wait()

    def wait_for_commands(self, count: int, timeout: float = 1.0) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self.commands) >= count,
                timeout=timeout,
            )


class _StaticBrain:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        return self.response


class _LatestWinsBrain:
    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        if request.transcript == "첫 번째 요청":
            self.first_started.set()
            self.release_first.wait(timeout=1.0)
            return {
                "actions": [
                    {
                        "type": "ardy_motion",
                        "prompt": "A girl performs the obsolete first action.",
                        "duration_seconds": 3,
                    }
                ]
            }
        return {
            "actions": [
                {
                    "type": "ardy_motion",
                    "prompt": "A girl waves with her right hand.",
                    "duration_seconds": 3,
                }
            ]
        }


class _FakeMotionDirector:
    def __init__(self) -> None:
        self.cues: list[tuple[str, ArdyMotionAction, str]] = []
        self.halt_reasons: list[str] = []
        self.active_cue_id: str | None = None
        self.lease_token = 10

    def submit_cue(self, action: ArdyMotionAction, *, turn_id: str) -> str:
        cue_id = f"cue-{len(self.cues) + 1}"
        self.cues.append((cue_id, action, turn_id))
        self.active_cue_id = cue_id
        return cue_id

    def halt(self, *, reason: str) -> MotionHaltResult:
        self.halt_reasons.append(reason)
        self.active_cue_id = None
        self.lease_token += 1
        return MotionHaltResult(reason=reason, lease_token=self.lease_token)

    def snapshot(self) -> MotionDirectorSnapshot:
        return MotionDirectorSnapshot(
            state=(
                MotionDirectorState.CUE
                if self.active_cue_id is not None
                else MotionDirectorState.IDLE
            ),
            running=True,
            lease_token=self.lease_token,
            active_cue_id=self.active_cue_id,
        )


class InteractionRuntimeTests(unittest.TestCase):
    def _build(self, brain):
        speech = _BlockingExecutor()
        motion = _FakeMotionDirector()
        supervisor = CharacterSupervisor(
            executors={ActionType.SAY: speech},
            action_id_factory=(f"action-{index}" for index in range(20)).__next__,
        )
        runtime = InteractionRuntime(
            brain=brain,
            supervisor=supervisor,
            motion_director=motion,
        )

        def cleanup() -> None:
            supervisor.halt_all(reason="test_cleanup")
            supervisor.wait_until_idle(timeout=1.0)

        self.addCleanup(cleanup)
        return runtime, supervisor, speech, motion

    def test_exact_fast_stop_bypasses_brain_and_cancels_both_resources(self) -> None:
        brain = _StaticBrain({"actions": [{"type": "say", "text": "unused"}]})
        runtime, supervisor, speech, motion = self._build(brain)
        supervisor.apply_bundle(
            ActionBundle(actions=(SayAction(text="말하는 중."),)),
            turn_id="setup",
        )
        motion.submit_cue(
            ArdyMotionAction(prompt="A girl dances.", duration_seconds=4.0),
            turn_id="setup",
        )
        self.assertTrue(speech.wait_for_commands(1))

        outcome = runtime.handle_utterance(transcript=" 멈춰!!! ", screenshot=None)

        self.assertEqual(outcome.status, TurnOutcomeStatus.FAST_STOP_REQUESTED)
        self.assertEqual(brain.requests, [])
        self.assertTrue(supervisor.wait_until_idle(timeout=1.0))
        self.assertIsNone(supervisor.snapshot().current(ControlResource.VOICE_OUTPUT))
        self.assertEqual(motion.halt_reasons, ["fast_stop"])
        self.assertIsNone(motion.active_cue_id)

    def test_slow_old_brain_response_is_dropped_after_a_newer_turn(self) -> None:
        brain = _LatestWinsBrain()
        runtime, _, _, motion = self._build(brain)
        outcomes = {}

        first_thread = threading.Thread(
            target=lambda: outcomes.setdefault(
                "first",
                runtime.handle_utterance(
                    transcript="첫 번째 요청",
                    screenshot=b"first-image",
                ),
            )
        )
        first_thread.start()
        self.assertTrue(brain.first_started.wait(timeout=1.0))

        outcomes["second"] = runtime.handle_utterance(
            transcript="두 번째 요청",
            screenshot=b"second-image",
        )
        brain.release_first.set()
        first_thread.join(timeout=1.0)

        self.assertFalse(first_thread.is_alive())
        self.assertEqual(outcomes["second"].status, TurnOutcomeStatus.APPLIED)
        self.assertEqual(outcomes["first"].status, TurnOutcomeStatus.SUPERSEDED)
        self.assertEqual(len(motion.cues), 1)
        self.assertEqual(
            motion.cues[0][1].prompt,
            "A girl waves with her right hand.",
        )

    def test_invalid_brain_response_starts_nothing_and_preserves_current_action(self) -> None:
        brain = _StaticBrain(
            {
                "actions": [{"type": "say", "text": "말할게."}],
                "intent": "not allowed",
            }
        )
        runtime, supervisor, _, motion = self._build(brain)
        current_id = motion.submit_cue(
            ArdyMotionAction(
                prompt="A girl keeps dancing.",
                duration_seconds=5.0,
            ),
            turn_id="setup",
        )

        outcome = runtime.handle_utterance(
            transcript="계속해",
            screenshot=b"image",
        )

        self.assertEqual(outcome.status, TurnOutcomeStatus.FAILED)
        self.assertIn("invalid fields", outcome.error)
        self.assertEqual(motion.active_cue_id, current_id)

    def test_english_stop_uses_the_brain_instead_of_the_fast_path(self) -> None:
        brain = _StaticBrain(
            {"actions": [{"type": "say", "text": "Okay, I will stop."}]}
        )
        runtime, _, speech, _ = self._build(brain)

        outcome = runtime.handle_utterance(transcript="stop", screenshot=b"image")

        self.assertEqual(outcome.status, TurnOutcomeStatus.APPLIED)
        self.assertEqual(len(brain.requests), 1)
        self.assertTrue(speech.wait_for_commands(1))
        self.assertEqual(brain.requests[0].body.value, "IDLE")
        self.assertEqual(brain.requests[0].speech.value, "IDLE")

    def test_turn_reserved_at_vad_end_fences_slow_older_transcription(self) -> None:
        brain = _StaticBrain(
            {"actions": [{"type": "say", "text": "최신 요청이야."}]}
        )
        runtime, _, speech, _ = self._build(brain)

        older = runtime.begin_utterance()
        newer = runtime.begin_utterance()
        old_outcome = runtime.handle_reserved_utterance(
            older,
            transcript="느리게 전사된 예전 요청",
            screenshot=b"old-image",
        )
        new_outcome = runtime.handle_reserved_utterance(
            newer,
            transcript="최신 요청",
            screenshot=b"new-image",
        )

        self.assertEqual(old_outcome.status, TurnOutcomeStatus.SUPERSEDED)
        self.assertEqual(new_outcome.status, TurnOutcomeStatus.APPLIED)
        self.assertEqual(len(brain.requests), 1)
        self.assertTrue(speech.wait_for_commands(1))


if __name__ == "__main__":
    unittest.main()
