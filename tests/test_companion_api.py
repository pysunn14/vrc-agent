from __future__ import annotations

from dataclasses import dataclass
import threading
import unittest

from vrc_ardy_agent.action_contracts import ActionType, ArdyMotionAction, ExecutionCommand
from vrc_ardy_agent.character_supervisor import CharacterSupervisor
from vrc_ardy_agent.companion_api import (
    CompanionApiClient,
    CompanionApiError,
    CompanionControlService,
    create_companion_server,
)
from vrc_ardy_agent.interaction_runtime import InteractionRuntime
from vrc_ardy_agent.motion_stream import (
    MotionDirectorSnapshot,
    MotionDirectorState,
    MotionHaltResult,
)


class _HoldExecutor:
    def __init__(self) -> None:
        self.commands: list[ExecutionCommand] = []
        self.started = threading.Event()

    def run(self, command, cancel_event, mark_running) -> None:
        self.commands.append(command)
        mark_running()
        self.started.set()
        cancel_event.wait()


class _UnusedBrain:
    def propose(self, request):
        raise AssertionError("brain should not be called by test-control endpoints")


class _MotionDirector:
    def __init__(self) -> None:
        self.cues: list[ArdyMotionAction] = []
        self.active_cue_id: str | None = None
        self.started = threading.Event()
        self.lease_token = 1

    def submit_cue(self, action: ArdyMotionAction, *, turn_id: str) -> str:
        del turn_id
        self.cues.append(action)
        self.active_cue_id = f"cue-{len(self.cues)}"
        self.started.set()
        return self.active_cue_id

    def halt(self, *, reason: str) -> MotionHaltResult:
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


@dataclass(frozen=True)
class _ComponentState:
    running: bool
    heartbeat_monotonic: float


class _StatusSource:
    def snapshot(self):
        return _ComponentState(running=True, heartbeat_monotonic=12.0)


class _CallableStatusSource:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bytes:
        self.calls += 1
        return b"device action"

    def snapshot(self):
        return {"running": True}


class CompanionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.speech = _HoldExecutor()
        self.motion_executor = _HoldExecutor()
        self.motion = _MotionDirector()
        self.supervisor = CharacterSupervisor(
            executors={
                ActionType.SAY: self.speech,
                ActionType.ARDY_MOTION: self.motion_executor,
            }
        )
        runtime = InteractionRuntime(
            brain=_UnusedBrain(),
            supervisor=self.supervisor,
            motion_director=self.motion,
        )
        service = CompanionControlService(runtime, enable_test_controls=True)
        self.server = create_companion_server(("127.0.0.1", 0), service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = CompanionApiClient(f"http://{host}:{port}")

    def tearDown(self) -> None:
        self.client.stop(reason="test_cleanup")
        self.supervisor.wait_until_idle(timeout=1.0)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.0)

    def test_status_test_action_and_stop_use_one_runtime_control_boundary(self) -> None:
        health = self.client.health()
        applied = self.client.test_bundle(
            {
                "actions": [
                    {"type": "say", "text": "안녕."},
                    {
                        "type": "ardy_motion",
                        "prompt": "A girl waves.",
                        "duration_seconds": 3,
                    },
                ]
            }
        )
        self.assertTrue(self.speech.started.wait(timeout=1.0))
        self.assertTrue(self.motion.started.wait(timeout=1.0))
        status = self.client.status()

        self.assertEqual(health["status"], "ok")
        self.assertEqual(applied["status"], "APPLIED")
        self.assertEqual(len(status["supervisor"]["current_actions"]), 1)
        self.assertEqual(status["motion"]["active_cue_id"], "cue-1")

        stopped = self.client.stop(reason="manual_test")
        self.assertEqual(stopped["reason"], "manual_test")
        self.assertTrue(self.supervisor.wait_until_idle(timeout=1.0))

    def test_test_screenshot_returns_current_windows_jpeg(self) -> None:
        jpeg = b"\xff\xd8current-window\xff\xd9"
        self.server.control_service.screenshot_provider = lambda: jpeg

        self.assertEqual(self.client.test_screenshot(), jpeg)

    def test_test_controls_can_be_disabled(self) -> None:
        service = CompanionControlService(
            InteractionRuntime(
                brain=_UnusedBrain(),
                supervisor=self.supervisor,
                motion_director=self.motion,
            ),
            enable_test_controls=False,
        )
        server = create_companion_server(("127.0.0.1", 0), service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        client = CompanionApiClient(f"http://{host}:{port}")
        try:
            with self.assertRaises(CompanionApiError) as raised:
                client.test_say("안녕.")
            self.assertEqual(raised.exception.status, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)

    def test_status_includes_registered_component_snapshots(self) -> None:
        runtime = InteractionRuntime(
            brain=_UnusedBrain(),
            supervisor=self.supervisor,
            motion_director=self.motion,
        )
        service = CompanionControlService(
            runtime,
            status_sources={
                "bridge": _StatusSource(),
                "custom": lambda: {"queue_depth": 2},
            },
        )

        status = service.status()

        self.assertEqual(
            status["components"],
            {
                "bridge": {"running": True, "heartbeat_monotonic": 12.0},
                "custom": {"queue_depth": 2},
            },
        )

    def test_status_prefers_snapshot_over_callable_device_action(self) -> None:
        source = _CallableStatusSource()
        service = CompanionControlService(
            InteractionRuntime(
                brain=_UnusedBrain(),
                supervisor=self.supervisor,
                motion_director=self.motion,
            ),
            status_sources={"device": source},
        )

        status = service.status()

        self.assertEqual(status["components"]["device"], {"running": True})
        self.assertEqual(source.calls, 0)


if __name__ == "__main__":
    unittest.main()
