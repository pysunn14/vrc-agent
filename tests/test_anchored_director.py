from dataclasses import replace
from functools import partial
import time
import unittest

from tests.test_motion_replay import pose
from tests.test_anchored_motion import Runtime, Mapper, specs
from tests.test_motion_director import _RecordingOutput
from vrc_ardy_agent.anchored_motion import AnchoredMotionSession, MotionAssets
from vrc_ardy_agent.behavior_catalog import BehaviorSpec
from vrc_ardy_agent.motion_assets import MotionClip
from vrc_ardy_agent.action_contracts import PresetMotionAction
from vrc_ardy_agent.motion_director import MotionDirector
from vrc_ardy_agent.motion_stream import MotionDirectorState, CALIBRATED_IDLE


class AnchoredDirectorTests(unittest.TestCase):
    def test_finite_clip_returns_idle_under_same_lease_without_timer_truncation(self):
        idle=pose();assets=MotionAssets(idle,specs(),{"yawn": MotionClip((idle,replace(pose(.2),face=(('Expression', 1.),)),idle))})
        output=_RecordingOutput();runtime=Runtime()
        director=MotionDirector(runtime=runtime,output=output,mapper_factory=Mapper,
            motion_assets=assets,idle_prompt=CALIBRATED_IDLE,session_factory=partial(AnchoredMotionSession,assets=assets,transition_seconds=.2),cue_poll_seconds=.005)
        director.start();director.attach_output_session('test')
        try:
            self.wait(lambda: len(output.pose_calls)>0)
            director.submit_cue(PresetMotionAction('yawn'),turn_id='yawn')
            self.wait(lambda: any(dict(call[1].face).get('Expression',0)==1 for call in output.pose_calls))
            self.wait(lambda: director.snapshot().state==MotionDirectorState.IDLE)
            self.assertIsNone(director.snapshot().last_error)
            self.assertEqual(runtime.calls,0)
            self.assertEqual(len(output.authorized),1)
            self.assertEqual(output.neutralized,[])
            self.assertEqual(sum(dict(call[1].face).get('Expression',0)==1 for call in output.pose_calls),1)
            self.assertEqual(output.pose_calls[-1][1],idle)
            sequences=[call[2] for call in output.pose_calls]
            self.assertEqual(sequences,list(range(len(sequences))))
        finally: director.stop()

    def wait(self, predicate):
        deadline=time.monotonic()+2
        while not predicate() and time.monotonic()<deadline: time.sleep(.01)
        self.assertTrue(predicate())
