from __future__ import annotations

import threading
import time
import unittest

from vrc_ardy_agent.action_contracts import (
    ActionBundle,
    ActionState,
    ActionType,
    ArdyMotionAction,
    ControlResource,
    ExecutionCommand,
    ResourceLease,
    SayAction,
)
from vrc_ardy_agent.character_supervisor import CharacterSupervisor
from vrc_ardy_agent.control_resources import (
    ControlResourceManager,
    ResourceBusyError,
)


class _BlockingExecutor:
    def __init__(self) -> None:
        self.commands: list[ExecutionCommand] = []
        self.cancelled_action_ids: list[str] = []
        self._condition = threading.Condition()

    def run(
        self,
        command: ExecutionCommand,
        cancel_event: threading.Event,
        mark_running,
    ) -> None:
        with self._condition:
            self.commands.append(command)
            self._condition.notify_all()
        mark_running()
        cancel_event.wait()
        with self._condition:
            self.cancelled_action_ids.append(command.action_id)
            self._condition.notify_all()

    def wait_for_commands(self, count: int, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self.commands) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


class _FailingExecutor:
    def run(
        self,
        command: ExecutionCommand,
        cancel_event: threading.Event,
        mark_running,
    ) -> None:
        mark_running()
        raise RuntimeError("synthesis failed")


class _RecordingOutputAuthority:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def authorize(self, lease: ResourceLease) -> None:
        self.events.append(("authorize", lease.resource, lease.token, lease.action_id))

    def neutralize(self, resource: ControlResource, lease_token: int) -> None:
        self.events.append(("neutralize", resource, lease_token))


class ControlResourceManagerTests(unittest.TestCase):
    def test_resource_has_one_owner_and_monotonic_fencing_tokens(self) -> None:
        manager = ControlResourceManager()

        first = manager.acquire(ControlResource.VOICE_OUTPUT, "speech-a")
        with self.assertRaises(ResourceBusyError):
            manager.acquire(ControlResource.VOICE_OUTPUT, "speech-b")

        invalidated = manager.invalidate(
            ControlResource.VOICE_OUTPUT,
            expected_action_id="speech-a",
        )
        second = manager.acquire(ControlResource.VOICE_OUTPUT, "speech-b")

        self.assertEqual(first.token, 1)
        self.assertEqual(invalidated.token, 2)
        self.assertIsNone(invalidated.action_id)
        self.assertEqual(second.token, 3)
        self.assertFalse(manager.is_current(first))
        self.assertTrue(manager.is_current(second))


class CharacterSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.speech = _BlockingExecutor()
        self.motion = _BlockingExecutor()
        self.output = _RecordingOutputAuthority()
        action_ids = iter(
            ("speech-a", "motion-a", "speech-b", "motion-b", "speech-c")
        )
        self.supervisor = CharacterSupervisor(
            executors={
                ActionType.SAY: self.speech,
                ActionType.ARDY_MOTION: self.motion,
            },
            output_authority=self.output,
            action_id_factory=action_ids.__next__,
        )

    def tearDown(self) -> None:
        self.supervisor.halt_all(reason="test_cleanup")
        self.supervisor.wait_until_idle(timeout=1.0)

    def test_speech_and_motion_start_as_independent_parallel_resources(self) -> None:
        result = self.supervisor.apply_bundle(
            ActionBundle(
                actions=(
                    SayAction(text="같이 시작할게."),
                    ArdyMotionAction(prompt="A girl dances.", duration_seconds=3.0),
                )
            ),
            turn_id="turn-1",
        )

        self.assertTrue(self.speech.wait_for_commands(1))
        self.assertTrue(self.motion.wait_for_commands(1))
        self.assertTrue(
            self.supervisor.wait_for_action_state(
                "speech-a", ActionState.RUNNING, timeout=1.0
            )
        )
        self.assertTrue(
            self.supervisor.wait_for_action_state(
                "motion-a", ActionState.RUNNING, timeout=1.0
            )
        )
        snapshot = self.supervisor.snapshot()

        self.assertEqual(set(result.action_ids), {"speech-a", "motion-a"})
        self.assertEqual(
            snapshot.current(ControlResource.VOICE_OUTPUT).state,
            ActionState.RUNNING,
        )
        self.assertEqual(
            snapshot.current(ControlResource.FULL_BODY_POSE).state,
            ActionState.RUNNING,
        )

    def test_new_speech_replaces_only_speech_and_keeps_motion_running(self) -> None:
        self.supervisor.apply_bundle(
            ActionBundle(
                actions=(
                    SayAction(text="첫 번째 말."),
                    ArdyMotionAction(prompt="A girl dances.", duration_seconds=3.0),
                )
            ),
            turn_id="turn-1",
        )
        self.assertTrue(self.speech.wait_for_commands(1))
        self.assertTrue(self.motion.wait_for_commands(1))
        original_motion_id = self.motion.commands[0].action_id

        self.supervisor.apply_bundle(
            ActionBundle(actions=(SayAction(text="두 번째 말."),)),
            turn_id="turn-2",
        )

        self.assertTrue(self.speech.wait_for_commands(2))
        self.assertTrue(
            self.supervisor.wait_for_action_state(
                "speech-b", ActionState.RUNNING, timeout=1.0
            )
        )
        snapshot = self.supervisor.snapshot()
        self.assertEqual(
            snapshot.current(ControlResource.VOICE_OUTPUT).action_id,
            "speech-b",
        )
        self.assertEqual(
            snapshot.current(ControlResource.FULL_BODY_POSE).action_id,
            original_motion_id,
        )
        pose_neutralizations = [
            event
            for event in self.output.events
            if event[:2] == ("neutralize", ControlResource.FULL_BODY_POSE)
        ]
        self.assertEqual(pose_neutralizations, [])

    def test_halt_all_invalidates_both_resources_and_is_idempotent(self) -> None:
        self.supervisor.apply_bundle(
            ActionBundle(
                actions=(
                    SayAction(text="말하는 중."),
                    ArdyMotionAction(prompt="A girl waves.", duration_seconds=3.0),
                )
            ),
            turn_id="turn-1",
        )
        self.assertTrue(self.speech.wait_for_commands(1))
        self.assertTrue(self.motion.wait_for_commands(1))

        first = self.supervisor.halt_all(reason="fast_stop")
        second = self.supervisor.halt_all(reason="fast_stop")

        self.assertEqual(first.resource_tokens, second.resource_tokens)
        self.assertTrue(self.supervisor.wait_until_idle(timeout=1.0))
        snapshot = self.supervisor.snapshot()
        self.assertIsNone(snapshot.current(ControlResource.VOICE_OUTPUT))
        self.assertIsNone(snapshot.current(ControlResource.FULL_BODY_POSE))
        self.assertEqual(
            {record.state for record in snapshot.recent_actions},
            {ActionState.CANCELLED},
        )

    def test_one_executor_failure_does_not_cancel_the_other_resource(self) -> None:
        supervisor = CharacterSupervisor(
            executors={
                ActionType.SAY: _FailingExecutor(),
                ActionType.ARDY_MOTION: self.motion,
            },
            output_authority=self.output,
            action_id_factory=iter(("speech-fail", "motion-ok")).__next__,
        )
        try:
            supervisor.apply_bundle(
                ActionBundle(
                    actions=(
                        SayAction(text="실패할 말."),
                        ArdyMotionAction(
                            prompt="A girl keeps dancing.",
                            duration_seconds=3.0,
                        ),
                    )
                ),
                turn_id="turn-1",
            )

            self.assertTrue(self.motion.wait_for_commands(1))
            self.assertTrue(
                supervisor.wait_for_action_state(
                    "speech-fail",
                    ActionState.FAILED,
                    timeout=1.0,
                )
            )
            self.assertEqual(
                supervisor.wait_for_action_state(
                    "motion-ok",
                    ActionState.RUNNING,
                    timeout=1.0,
                ),
                True,
            )
            self.assertEqual(
                supervisor.snapshot().current(ControlResource.FULL_BODY_POSE).state,
                ActionState.RUNNING,
            )
        finally:
            supervisor.halt_all(reason="test_cleanup")
            supervisor.wait_until_idle(timeout=1.0)

    def test_speech_only_supervisor_owns_only_voice_output(self) -> None:
        supervisor = CharacterSupervisor(
            executors={ActionType.SAY: self.speech},
            output_authority=self.output,
            action_id_factory=lambda: "speech-only",
        )
        try:
            result = supervisor.apply_bundle(
                ActionBundle(actions=(SayAction(text="안녕."),)),
                turn_id="turn-speech",
            )
            self.assertEqual(result.action_ids, ("speech-only",))
            self.assertTrue(self.speech.wait_for_commands(1))

            with self.assertRaisesRegex(ValueError, "not managed"):
                supervisor.apply_bundle(
                    ActionBundle(
                        actions=(
                            ArdyMotionAction(
                                prompt="A girl waves.",
                                duration_seconds=3.0,
                            ),
                        )
                    ),
                    turn_id="turn-motion",
                )

            halted = supervisor.halt_all(reason="test")
            self.assertEqual(
                halted.resource_tokens,
                ((ControlResource.VOICE_OUTPUT, 2),),
            )
            self.assertFalse(
                any(
                    event[1] == ControlResource.FULL_BODY_POSE
                    for event in self.output.events
                )
            )
        finally:
            supervisor.halt_all(reason="test_cleanup")
            supervisor.wait_until_idle(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
