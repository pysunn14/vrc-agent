import threading
import time
import unittest
from dataclasses import replace

from tests.test_interaction_runtime import InteractionRuntimeTests, _StaticBrain
from vrc_ardy_agent.observation_loop import ObservationLoop, ReplayFrame
from vrc_ardy_agent.planning_trace import PlanningTrace

JPEG = b'\xff\xd8fixture\xff\xd9'
PLAN = {'requires_visual': False, 'actions': [{'type': 'say', 'text': '안녕'}]}


class PolicyTests(unittest.TestCase):
    _build = InteractionRuntimeTests._build

    def test_always_first_call_has_replay_evidence_selective_has_none(self):
        for policy, expected in [('always', True), ('selective', False)]:
            brain = _StaticBrain(PLAN)
            runtime, _, _, _ = self._build(brain)
            runtime.configure_observation(lambda: ReplayFrame(JPEG), policy=policy)
            result = runtime.handle_utterance(transcript='안녕', screenshot=None)
            self.assertEqual(result.status.value, 'APPLIED')
            request = brain.requests[0]
            self.assertEqual(request.screenshot is not None, expected)
            if expected:
                self.assertEqual(request.observations[0].source, 'replay')
                self.assertEqual(request.observations[0].freshness_basis, 'fixed_fixture')
                self.assertTrue(request.observations[0].content_sha256)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            ObservationLoop(policy='wrong')

    def test_replay_age_does_not_masquerade_as_live_frame_age(self):
        class Brain:
            def propose(self, request):
                time.sleep(.02)
                return {**PLAN, 'requires_visual': True}
        runtime, _, _, _ = self._build(Brain())
        runtime.configure_observation(lambda: ReplayFrame(JPEG), policy='always', max_frame_age=.001)
        self.assertEqual(runtime.handle_utterance(transcript='색상', screenshot=None).status.value, 'APPLIED')


class TraceTests(unittest.TestCase):
    def test_timeout_and_late_completion_are_same_call(self):
        trace = PlanningTrace('turn', 1)
        loop = ObservationLoop()
        release = threading.Event()
        with self.assertRaises(TimeoutError):
            loop._call(lambda: release.wait(2), lambda: True, time.monotonic()+.02,
                       trace=trace, kind='model', round_index=0)
        pending = trace.snapshot()['calls'][0]
        self.assertEqual(pending['wait_status'], 'timeout')
        self.assertEqual(pending['status'], 'running')
        release.set()
        for _ in range(100):
            if trace.snapshot()['calls'][0]['status'] == 'completed': break
            time.sleep(.005)
        calls = trace.snapshot()['calls']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['call_id'], pending['call_id'])
        self.assertTrue(calls[0]['late'])
        self.assertIsNone(calls[0]['usage'])

    def test_worker_capacity_not_counted_as_started_call(self):
        trace = PlanningTrace('turn', 1)
        loop = ObservationLoop(max_active_calls=1)
        release = threading.Event()
        try:
            with self.assertRaises(TimeoutError):
                loop._call(lambda: release.wait(2), lambda: True, time.monotonic()+.01,
                           trace=trace, kind='model', round_index=0)
            with self.assertRaisesRegex(RuntimeError, 'capacity'):
                loop._call(lambda: None, lambda: True, time.monotonic()+1,
                           trace=trace, kind='model', round_index=1)
            calls = trace.snapshot()['calls']
            self.assertEqual([x['status'] for x in calls], ['running', 'not_started'])
            self.assertEqual(calls[1]['wait_status'], 'capacity')
        finally: release.set()

    def test_runtime_records_validated_plan_without_dispatch(self):
        from vrc_ardy_agent.interaction_runtime import InteractionRuntime
        base = InteractionRuntimeTests()
        brain = _StaticBrain(PLAN)
        runtime, supervisor, speech, motion = base._build(brain)
        trace = PlanningTrace('placeholder', 1)
        plans = []
        runtime = InteractionRuntime(brain=brain, supervisor=supervisor, motion_director=motion,
                                     trace_factory=lambda token: trace, plan_recorder=plans.append)
        result = runtime.handle_utterance(transcript='안녕', screenshot=None)
        self.assertEqual(result.status.value, 'PLANNED')
        self.assertEqual(len(plans), 1)
        self.assertEqual(speech.commands, [])
        self.assertEqual(motion.cues, [])
        record = trace.snapshot()
        self.assertGreater(record['plan_ms'], 0)
        self.assertGreaterEqual(record['plan_ms'], record['validation_ms'])
        self.assertEqual(record['status'], 'PLANNED')
        base.doCleanups()

    def test_invalid_plan_has_no_success_latency(self):
        from vrc_ardy_agent.interaction_runtime import InteractionRuntime
        base = InteractionRuntimeTests()
        brain = _StaticBrain({'actions': []})
        _, supervisor, _, motion = base._build(brain)
        trace = PlanningTrace('turn', 1)
        runtime = InteractionRuntime(brain=brain, supervisor=supervisor, motion_director=motion,
                                     trace_factory=lambda token: trace, plan_recorder=lambda b: self.fail('dispatch'))
        self.assertEqual(runtime.handle_utterance(transcript='안녕', screenshot=None).status.value, 'FAILED')
        self.assertIsNone(trace.snapshot()['plan_ms'])
        self.assertEqual(trace.snapshot()['failure_stage'], 'validation')
        base.doCleanups()

class AdditionalEvidenceTests(unittest.TestCase):
    def test_adapter_keeps_usage_even_when_plan_json_is_invalid(self):
        from tests.test_hermes_brain import _brain_request, _RecordingOpener, _FakeResponse
        from vrc_ardy_agent.hermes_brain import HermesBrainAdapter, HermesBrainProtocolError
        records=[]
        opener=_RecordingOpener(_FakeResponse({'usage':{'prompt_tokens':123,'completion_tokens':7},
            'model':'test-model','choices':[{'finish_reason':'stop','message':{'content':'invalid'}}]}))
        adapter=HermesBrainAdapter(endpoint='http://127.0.0.1:8642/v1/chat/completions',api_key='test',opener=opener)
        with self.assertRaises(HermesBrainProtocolError):
            adapter.propose(replace(_brain_request(),record_model_metadata=records.append))
        self.assertEqual(records[0]['usage']['prompt_tokens'],123)

    def test_stop_fences_record_only_plan_and_attributes_late_usage(self):
        from vrc_ardy_agent.interaction_runtime import InteractionRuntime
        base=InteractionRuntimeTests()
        started=threading.Event();release=threading.Event();trace_done=threading.Event()
        class Brain:
            def propose(self, request):
                started.set();release.wait(2)
                request.record_model_metadata({'usage':{'prompt_tokens':42}})
                return PLAN
        _, supervisor, _, motion=base._build(_StaticBrain(PLAN))
        records=[];traces=[]
        def factory(token):
            trace=PlanningTrace(token.turn_id,token.version);traces.append(trace);return trace
        runtime=InteractionRuntime(brain=Brain(),supervisor=supervisor,motion_director=motion,
            trace_factory=factory,plan_recorder=records.append)
        outcomes=[]
        worker=threading.Thread(target=lambda:outcomes.append(runtime.handle_utterance(transcript='안녕',screenshot=None)))
        worker.start();self.assertTrue(started.wait(1))
        begin=time.perf_counter();runtime.halt_all(reason='test');stop_ms=(time.perf_counter()-begin)*1000
        worker.join(1)
        self.assertEqual(outcomes[0].status.value,'SUPERSEDED')
        self.assertLess(stop_ms,200)
        release.set()
        for _ in range(100):
            if not runtime.observation_service.snapshot()['active_calls']:break
            time.sleep(.005)
        trace=traces[0].snapshot()
        self.assertEqual(records,[])
        self.assertIsNone(trace['plan_ms'])
        self.assertEqual(trace['calls'][0]['usage'],{'prompt_tokens':42})
        self.assertTrue(trace['calls'][0]['late'])
        base.doCleanups()
