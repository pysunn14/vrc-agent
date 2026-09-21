import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

from tests.test_motion_replay import pose, Sink
from vrc_ardy_agent.anchored_motion import AnchoredMotionSession, MotionAssets
from vrc_ardy_agent.behavior_catalog import BehaviorSpec
from vrc_ardy_agent.motion_assets import MotionClip
from vrc_ardy_agent.motion_stream import CALIBRATED_IDLE as DEFAULT_IDLE_PROMPT
from vrc_ardy_agent.action_contracts import validate_action_bundle, PlanValidationError
from vrc_ardy_agent.device_payloads import encode_pose_payload, decode_pose_payload


def specs():
    return {key: BehaviorSpec.parse(key, data) for key, data in {
        'yawn': {'source': 'clip', 'frames': 'motion.json'},
        'walk_forward': {'source': 'locomotion', 'velocity': [0, .18, 0], 'duration_seconds': 5},
        'walk_backward': {'source': 'locomotion', 'velocity': [0, -.18, 0], 'duration_seconds': 5},
    }.items()}


class Runtime:
    fps = 20
    horizon_frames = 40
    def __init__(self):
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
    def clear_history(self): pass
    def set_prompt(self, prompt): pass
    def generate_next(self):
        self.calls += 1
        self.started.set()
        self.release.wait(2)
        return SimpleNamespace(generation_seconds=0)


class Mapper:
    def reset(self): pass
    def map_chunk(self, chunk): return [pose(1)] * 4


class AnchoredTests(unittest.TestCase):
    def make(self):
        idle = pose()
        assets = MotionAssets(idle, specs(), {"yawn": MotionClip((idle, replace(pose(.1), face=(('Expression', 1.),)), idle))})
        runtime, sink = Runtime(), Sink()
        session = AnchoredMotionSession(runtime=runtime, mapper=Mapper(), sink=sink,
                                        assets=assets, transition_seconds=.1)
        session.start(DEFAULT_IDLE_PROMPT)
        return session, runtime, sink, idle

    def step(self, session):
        return session.send_next()

    def test_idle_never_generates_and_preserves_tracking(self):
        session, runtime, sink, idle = self.make()
        try:
            for _ in range(10): self.step(session)
            self.assertEqual(sink.frames, [idle] * 10)
            self.assertEqual(runtime.calls, 0)
            self.assertEqual(sink.closed, 0)
        finally: session.stop()

    def test_yawn_finishes_once_and_returns_exact_idle(self):
        session, runtime, sink, idle = self.make()
        try:
            revision = session.set_prompt('preset:yawn')
            for _ in range(3): self.step(session)
            self.assertEqual(session.completed_revision, revision)
            self.assertEqual([dict(f.face).get('Expression',0) for f in sink.frames], [0, 1, 0])
            idle_revision = session.set_prompt(DEFAULT_IDLE_PROMPT)
            for _ in range(8): self.step(session)
            self.assertEqual(sink.frames[-5:], [idle] * 5)
            self.assertEqual(session.status.playing_prompt_revision, idle_revision)
            self.assertEqual(runtime.calls, 0)
            self.assertEqual(sink.closed, 0)
        finally: session.stop()

    def test_walk_signs_and_immediate_stop_before_pose_blend(self):
        session, runtime, sink, idle = self.make()
        try:
            session.set_prompt('preset:walk_forward'); self.step(session)
            self.assertEqual(sink.frames[-1].locomotion_y, .18)
            session.set_prompt('preset:walk_backward'); self.step(session)
            self.assertEqual(sink.frames[-1].locomotion_y, -.18)
            session.set_prompt(DEFAULT_IDLE_PROMPT); self.step(session)
            self.assertEqual(sink.frames[-1].locomotion_y, 0)
            self.assertEqual(dict(sink.frames[-1].face).get('Expression',0), 0)
        finally: session.stop()

    def test_superseded_generation_cannot_overwrite_idle(self):
        session, runtime, sink, idle = self.make()
        try:
            session.set_prompt('Raise one hand.')
            self.assertTrue(runtime.started.wait(1))
            self.step(session)
            self.assertEqual(sink.frames[-1], idle)
            session.set_prompt(DEFAULT_IDLE_PROMPT)
            runtime.release.set()
            deadline=time.monotonic()+1
            while session.status.generation_in_progress and time.monotonic()<deadline:
                time.sleep(.001)
            self.assertFalse(session.status.generation_in_progress)
            for _ in range(10): self.step(session)
            self.assertEqual(sink.frames[-1], idle)
            self.assertNotIn(pose(1), sink.frames)
        finally:
            runtime.release.set(); session.stop()

    def test_generated_cue_returns_to_calibrated_idle_without_generating_idle(self):
        session, runtime, sink, idle = self.make()
        try:
            runtime.release.set()
            session.set_prompt('Raise one hand.')
            deadline=time.monotonic()+1
            while session.status.buffered_frames == 0 and time.monotonic()<deadline:
                time.sleep(.001)
            self.step(session); self.step(session)
            self.assertEqual(sink.frames[-1],pose(1))
            session.set_prompt(DEFAULT_IDLE_PROMPT)
            for _ in range(4): self.step(session)
            self.assertEqual(sink.frames[-1],idle)
        finally: session.stop()

    def test_generation_failure_is_reported_instead_of_silent_idle(self):
        session, runtime, sink, idle = self.make()
        def fail(): raise ValueError('generation broke')
        runtime.generate_next=fail
        try:
            session.set_prompt('Raise one hand.')
            deadline=time.monotonic()+1
            while session._producer_error is None and time.monotonic()<deadline:
                time.sleep(.001)
            with self.assertRaisesRegex(RuntimeError,'generation failed'):
                self.step(session)
        finally: session.stop()

    def test_cancel_yawn_clears_face_and_restores_pose(self):
        session, runtime, sink, idle = self.make()
        try:
            session.set_prompt('preset:yawn'); self.step(session); self.step(session)
            self.assertEqual(dict(sink.frames[-1].face).get('Expression',0), 1)
            session.set_prompt(DEFAULT_IDLE_PROMPT)
            for _ in range(4): self.step(session)
            self.assertEqual(sink.frames[-1], idle)
        finally: session.stop()

    def test_face_channels_survive_device_transport(self):
        p=replace(pose(), face=(('Expression', .75),))
        self.assertEqual(decode_pose_payload(encode_pose_payload(p)), p)

    def test_preset_contract_rejects_two_body_owners(self):
        bundle=validate_action_bundle({'actions':[{'type':'motion','name':'yawn'}]}, behaviors=specs())
        self.assertEqual(bundle.actions[0].name,'yawn')
        with self.assertRaises(PlanValidationError):
            validate_action_bundle({'actions':[{'type':'motion','name':'yawn'},
                {'type':'ardy_motion','prompt':'Wave.','duration_seconds':2}]})
        with self.assertRaises(PlanValidationError):
            validate_action_bundle({'actions':[{'type':'motion','name':'walk_forward','duration_seconds':float('nan')}]})

class FacialOutputTests(unittest.TestCase):
    def test_face_and_pose_share_output_and_neutralize_together(self):
        import struct
        from tests.test_live_sink import _FakeSocket
        from vrc_ardy_agent.live_sink import SixPointUdpSink
        sock = _FakeSocket()
        sink = SixPointUdpSink(host='127.0.0.1', send_face=True,
                               send_locomotion=False, socket_factory=lambda: sock)
        sink.send(replace(pose(), face=(('Expression', .75),)))
        self.assertTrue(sock.sent[0][0].startswith(b'/avatar/parameters/Expression'))
        self.assertEqual(struct.unpack('>f',sock.sent[0][0][-4:])[0],.75)
        sink.neutralize_pose()
        faces=[p for p,a in sock.sent if p.startswith(b'/avatar/parameters/Expression')]
        self.assertEqual(struct.unpack('>f',faces[-1][-4:])[0],0)
        sink.close()
