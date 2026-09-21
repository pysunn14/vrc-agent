from vrc_ardy_agent.idle_behavior_policy import IdleBehaviorPolicy


def test_policy_waits_for_continuous_quiet_and_rotates_available_pool():
    policy = IdleBehaviorPolicy({'enabled': True, 'interval_seconds': 30, 'behaviors': ['missing', 'stretch', 'wave']}, {'stretch', 'wave'})
    dispatched = []
    def submit(name): dispatched.append(name); return f'action-{len(dispatched)}'
    policy.tick(now=0, eligible=True, submit=submit)
    policy.tick(now=29, eligible=True, submit=submit)
    policy.tick(now=30, eligible=False, submit=submit)
    policy.tick(now=31, eligible=True, submit=submit)
    policy.tick(now=60, eligible=True, submit=submit)
    assert not dispatched
    policy.tick(now=61, eligible=True, submit=submit)
    policy.tick(now=61, eligible=True, submit=submit)
    assert dispatched == ['stretch']
    policy.tick(now=70, eligible=False, submit=submit)
    policy.tick(now=80, eligible=True, submit=submit)
    policy.tick(now=110, eligible=True, submit=submit)
    assert dispatched == ['stretch', 'wave']
    assert policy.snapshot()['unavailable'] == ['missing']


def test_failed_autonomous_action_is_observable_and_does_not_retry_forever():
    policy = IdleBehaviorPolicy({'enabled': True, 'interval_seconds': 1, 'behaviors': ['wave']}, {'wave'})
    def fail(name): raise RuntimeError('output closed')
    policy.tick(now=0, eligible=True, submit=fail)
    policy.tick(now=1, eligible=True, submit=fail)
    assert policy.snapshot()['state'] == 'failed'
    assert 'output closed' in policy.snapshot()['last_error']
    policy.tick(now=100, eligible=True, submit=lambda _: (_ for _ in ()).throw(AssertionError('must not retry')))


def test_runtime_blocks_autonomy_for_speech_input_pending_turn_and_disconnected_output():
    from dataclasses import replace
    from tests.test_interaction_runtime import _StaticBrain, _FakeMotionDirector, _BlockingExecutor
    from vrc_ardy_agent.action_contracts import ActionType
    from vrc_ardy_agent.behavior_catalog import BehaviorSpec
    from vrc_ardy_agent.character_supervisor import CharacterSupervisor
    from vrc_ardy_agent.interaction_runtime import InteractionRuntime
    class Motion(_FakeMotionDirector):
        connected = True
        def snapshot(self):
            return replace(super().snapshot(), output_session_id='session' if self.connected else None)
        def submit_idle_cue(self, action, *, turn_id): return self.submit_cue(action, turn_id=turn_id)
    motion = Motion()
    runtime = InteractionRuntime(brain=_StaticBrain({}), motion_director=motion,
        supervisor=CharacterSupervisor(executors={ActionType.SAY: _BlockingExecutor()}),
        behaviors={'wave': BehaviorSpec.parse('wave', {'source': 'ardy', 'prompt': 'Wave.', 'duration_seconds': 3})},
        autonomy={'enabled': True, 'interval_seconds': 30, 'behaviors': ['wave']})
    runtime.tick_autonomy(now=0, input_quiet=True)
    runtime.tick_autonomy(now=30, input_quiet=False)
    assert not motion.cues
    runtime.tick_autonomy(now=31, input_quiet=True)
    token = runtime.begin_utterance()
    runtime.tick_autonomy(now=61, input_quiet=True)
    assert not motion.cues
    runtime.fail_reserved_utterance(token, error='no transcript')
    runtime.tick_autonomy(now=62, input_quiet=True)
    motion.connected = False
    runtime.tick_autonomy(now=92, input_quiet=True)
    assert not motion.cues
    motion.connected = True
    runtime.tick_autonomy(now=93, input_quiet=True)
    runtime.note_input_activity()
    runtime.tick_autonomy(now=123, input_quiet=True)
    assert not motion.cues
    runtime.tick_autonomy(now=153, input_quiet=True)
    assert len(motion.cues) == 1
    runtime.tick_autonomy(now=200, input_quiet=True)
    assert len(motion.cues) == 1
