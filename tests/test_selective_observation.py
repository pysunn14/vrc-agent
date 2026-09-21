from dataclasses import replace
import threading
import unittest
from tests.test_interaction_runtime import InteractionRuntimeTests, _StaticBrain
from tests.test_hermes_brain import _brain_request, _RecordingOpener, _FakeResponse
from vrc_ardy_agent.hermes_brain import HermesBrainAdapter, HermesBrainProtocolError


class ObservationTests(unittest.TestCase):
    _build = InteractionRuntimeTests._build

    def test_language_only_does_not_capture(self):
        runtime, _, _, _ = self._build(
            _StaticBrain({"actions": [{"type": "say", "text": "안녕"}]})
        )
        runtime.configure_observation(lambda: self.fail("unexpected capture"))
        outcome = runtime.handle_utterance(transcript="안녕", screenshot=None)
        self.assertEqual(outcome.status.value, "APPLIED")

    def test_observation_returns_to_same_turn_and_refreshes_state(self):
        class Brain:
            requests = []

            def propose(self, request):
                self.requests.append(request)
                if request.screenshot is None:
                    return {"type": "observe_scene", "query": "옷 색상을 확인"}
                return {
                    "actions": [{"type": "say", "text": "흰색이에요"}],
                    "requires_visual": True,
                }

        brain = Brain()
        runtime, _, _, _ = self._build(brain)
        captures = []
        runtime.configure_observation(
            lambda: captures.append(1) or b"\xff\xd8frame\xff\xd9"
        )
        outcome = runtime.handle_utterance(transcript="내 옷 어때?", screenshot=None)
        self.assertEqual(outcome.status.value, "APPLIED")
        self.assertEqual(len(captures), 1)
        self.assertEqual([r.round_index for r in brain.requests], [0, 1])
        self.assertEqual(brain.requests[0].turn_id, brain.requests[1].turn_id)
        self.assertEqual(brain.requests[1].observations[0].query, "옷 색상을 확인")

    def test_visual_plan_without_observe_call_is_gated(self):
        brain = _StaticBrain(
            {"actions": [{"type": "say", "text": "흰색"}], "requires_visual": True}
        )
        runtime, _, _, _ = self._build(brain)
        captures = []
        runtime.configure_observation(
            lambda: captures.append(1) or b"\xff\xd8frame\xff\xd9"
        )
        outcome = runtime.handle_utterance(
            transcript="지금 어떻게 보여?", screenshot=None
        )
        self.assertEqual(outcome.status.value, "APPLIED")
        self.assertEqual(len(captures), 1)
        self.assertEqual(len(brain.requests), 2)

    def test_capture_failure_starts_no_action(self):
        runtime, _, speech, motion = self._build(
            _StaticBrain({"type": "observe_scene", "query": "현재 화면"})
        )

        def fail():
            raise OSError("capture failed")

        runtime.configure_observation(fail)
        result = runtime.handle_utterance(transcript="이거 뭐야", screenshot=None)
        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("capture failed", result.error)
        self.assertEqual(speech.commands, [])
        self.assertEqual(motion.cues, [])

    def test_repeated_observe_is_bounded(self):
        runtime, _, _, _ = self._build(
            _StaticBrain({"type": "observe_scene", "query": "현재 화면"})
        )
        runtime.configure_observation(
            lambda: b"\xff\xd8frame\xff\xd9", max_observations=1
        )
        result = runtime.handle_utterance(transcript="화면", screenshot=None)
        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("limit", result.error)

    def test_stop_while_capture_waits_discards_late_result(self):
        runtime, _, speech, _ = self._build(
            _StaticBrain({"type": "observe_scene", "query": "화면"})
        )
        started = threading.Event()
        release = threading.Event()

        def capture():
            started.set()
            release.wait(2)
            return b"\xff\xd8late\xff\xd9"

        runtime.configure_observation(capture)
        results = []
        thread = threading.Thread(
            target=lambda: results.append(
                runtime.handle_utterance(transcript="봐줘", screenshot=None)
            )
        )
        thread.start()
        self.assertTrue(started.wait(1))
        runtime.halt_all(reason="test stop")
        thread.join(1)
        release.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].status.value, "SUPERSEDED")
        self.assertEqual(speech.commands, [])


class AdapterObservationTests(unittest.TestCase):
    def test_no_image_and_round_idempotency(self):
        import json

        opener = _RecordingOpener(
            _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"type":"observe_scene","query":"view"}'
                            }
                        }
                    ]
                }
            )
        )
        adapter = HermesBrainAdapter(
            endpoint="http://127.0.0.1:8642/v1/chat/completions",
            api_key="test",
            opener=opener,
        )
        adapter.propose(replace(_brain_request(), screenshot=None, round_index=2))
        body = json.loads(opener.requests[0].data)
        self.assertEqual(len(body["messages"][1]["content"]), 1)
        self.assertEqual(
            opener.requests[0].get_header("Idempotency-key"), "brain-turn-7-v7-r2"
        )

    def test_partial_json_plan_is_not_executable(self):
        opener = _RecordingOpener(
            _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": '{"actions":[{"type":"say","text":"hi"}]}'
                            },
                        }
                    ]
                }
            )
        )
        adapter = HermesBrainAdapter(
            endpoint="http://127.0.0.1:8642/v1/chat/completions",
            api_key="test",
            opener=opener,
        )
        with self.assertRaises(HermesBrainProtocolError):
            adapter.propose(_brain_request())


class ObservationBoundaryTests(unittest.TestCase):
    _build = InteractionRuntimeTests._build

    def test_capture_deadline_releases_turn_without_waiting_for_worker(self):
        release = threading.Event()
        runtime, _, _, _ = self._build(
            _StaticBrain({"type": "observe_scene", "query": "view"})
        )
        runtime.configure_observation(
            lambda: release.wait(2) or b"", timeout_seconds=0.03
        )
        try:
            outcome = runtime.handle_utterance(transcript="봐줘", screenshot=None)
            self.assertEqual(outcome.status.value, "FAILED")
            self.assertIn("deadline", outcome.error)
        finally:
            release.set()

    def test_duplicate_reserved_turn_does_not_capture_twice(self):
        runtime, _, _, _ = self._build(
            _StaticBrain({"type": "observe_scene", "query": "view"})
        )
        started = threading.Event()
        release = threading.Event()
        calls = []

        def capture():
            calls.append(1)
            started.set()
            release.wait(1)
            return b"\xff\xd8x\xff\xd9"

        runtime.configure_observation(capture, max_observations=1)
        token = runtime.begin_utterance()
        thread = threading.Thread(
            target=lambda: runtime.handle_reserved_utterance(
                token, transcript="봐줘", screenshot=None
            )
        )
        thread.start()
        self.assertTrue(started.wait(1))
        duplicate = runtime.handle_reserved_utterance(
            token, transcript="봐줘", screenshot=None
        )
        release.set()
        thread.join(1)
        self.assertEqual(duplicate.status.value, "SUPERSEDED")
        self.assertEqual(calls, [1])

    def test_execution_state_is_exposed_without_claiming_tracking_success(self):
        brain = _StaticBrain({"actions": [{"type": "say", "text": "안녕"}]})
        runtime, _, _, _ = self._build(brain)
        runtime.handle_utterance(transcript="안녕", screenshot=None)
        self.assertEqual(
            brain.requests[0].execution_feedback["tracking"]["status"], "unknown"
        )
        self.assertIn("recent_actions", brain.requests[0].execution_feedback)


class StaleFrameTests(unittest.TestCase):
    _build = InteractionRuntimeTests._build

    def test_slow_visual_answer_must_not_execute_with_expired_evidence(self):
        import time

        class Brain:
            def propose(self, request):
                if request.screenshot is None:
                    return {"type": "observe_scene", "query": "view"}
                time.sleep(0.02)
                return {
                    "requires_visual": True,
                    "actions": [{"type": "say", "text": "old view"}],
                }

        runtime, _, speech, _ = self._build(Brain())
        runtime.configure_observation(
            lambda: b"\xff\xd8x\xff\xd9", max_observations=1, max_frame_age=0.01
        )
        outcome = runtime.handle_utterance(transcript="현재 화면", screenshot=None)
        self.assertEqual(outcome.status.value, "FAILED")
        self.assertEqual(speech.commands, [])

    def test_worker_capacity_stays_bounded_after_cancellation(self):
        from vrc_ardy_agent.observation_loop import ObservationLoop
        import time

        loop = ObservationLoop(max_active_calls=1)
        release = threading.Event()
        try:
            with self.assertRaises(TimeoutError):
                loop._call(
                    lambda: release.wait(1), lambda: True, time.monotonic() + 0.02
                )
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                loop._call(lambda: None, lambda: True, time.monotonic() + 1)
        finally:
            release.set()


class ObservationCliTests(unittest.TestCase):
    _build = InteractionRuntimeTests._build

    def test_agentctl_observation_round_trip_and_status(self):
        from scripts.agentctl import build_parser
        from vrc_ardy_agent.agentctl_companion import run_companion
        from vrc_ardy_agent.companion_api import (
            CompanionControlService,
            create_companion_server,
        )

        class Brain:
            def propose(self, request):
                if request.screenshot is None:
                    return {"type": "observe_scene", "query": "view"}
                return {
                    "requires_visual": True,
                    "actions": [{"type": "say", "text": "확인했어요"}],
                }

        runtime, _, _, _ = self._build(Brain())
        runtime.configure_observation(lambda: b"\xff\xd8x\xff\xd9")
        service = CompanionControlService(
            runtime,
            enable_test_controls=True,
            status_sources={"observation": runtime.observation_service},
        )
        server = create_companion_server(("127.0.0.1", 0), service)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            args = build_parser().parse_args(
                ["companion", "test-utterance", "--url", url, "--text", "봐줘"]
            )
            result = run_companion(args)
            self.assertEqual(result["status"], "APPLIED")
            status = run_companion(
                build_parser().parse_args(["companion", "status", "--url", url])
            )
            self.assertEqual(status["components"]["observation"]["observations"], 1)
            self.assertEqual(status["components"]["observation"]["image_calls"], 1)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(1)
