from tests.rig_fixture import RIG_PATH
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.test_motion_replay import pose
from vrc_ardy_agent.action_contracts import validate_action_bundle, PlanValidationError
from vrc_ardy_agent.device_payloads import encode_pose_payload, decode_pose_payload
from vrc_ardy_agent.behavior_catalog import BehaviorSpec
from vrc_ardy_agent.motion_assets import MotionAssets, MotionClip
from vrc_ardy_agent.avatar_config import attest_calibration, inspect_avatar


def test_base_pose_without_behaviors_is_ready_and_unknown_action_is_rejected():
    assets = MotionAssets(pose())
    assert assets.behaviors == {}
    with pytest.raises(PlanValidationError, match='registered'):
        validate_action_bundle({'actions': [{'type': 'motion', 'name': 'missing'}]}, behaviors=assets.behaviors)


def test_registered_clip_name_and_duration_come_from_data():
    spec = BehaviorSpec.parse('stretch', {'source': 'clip', 'frames': 'stretch.json'})
    assets = MotionAssets(pose(), {'stretch': spec}, {'stretch': MotionClip((pose(), pose(.2), pose()))})
    assert assets.duration('stretch') == .15
    result = validate_action_bundle({'actions': [{'type': 'motion', 'name': 'stretch'}]}, behaviors=assets.behaviors)
    assert result.actions[0].name == 'stretch'
    with pytest.raises(PlanValidationError):
        validate_action_bundle({'actions': [{'type': 'motion', 'name': 'stretch', 'duration_seconds': 3}]}, behaviors=assets.behaviors)


def test_generic_face_channels_roundtrip_and_blend():
    from vrc_ardy_agent.pose_transition import PoseTransition
    source = replace(pose(), face=(('EyeClosure', .8), ('MouthOpen', .4)))
    assert decode_pose_payload(encode_pose_payload(source)) == source
    middle = PoseTransition(source, pose()).at(.5)
    assert dict(middle.face) == {'EyeClosure': .4, 'MouthOpen': .2}
    assert PoseTransition(source, pose()).at(1) == pose()


def test_calibration_attestation_is_invalidated_by_changed_base_pose():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'base.json').write_text(json.dumps(encode_pose_payload(pose())))
        avatar = {'rig': RIG_PATH, 'base_pose': 'base.json', 'face_channels': {}, 'hmd_base': [0, 1, 0]}
        avatar['calibration'] = attest_calibration(avatar, root, tracking_active=True)
        assert inspect_avatar(avatar, root)['ready'] is True
        (root / 'base.json').write_text(json.dumps(encode_pose_payload(pose(.1))))
        assert inspect_avatar(avatar, root)['ready'] is False
        with pytest.raises(ValueError, match='active'):
            attest_calibration(avatar, root, tracking_active=False)


def test_missing_optional_clip_is_unavailable_without_blocking_base_pose():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'base.json').write_text(json.dumps(encode_pose_payload(pose())))
        assets = MotionAssets.load(root / 'base.json', {
            'stretch': {'source': 'clip', 'frames': 'missing.json'},
            'step': {'source': 'locomotion', 'velocity': [0, .18, 0], 'duration_seconds': 5},
        }, base_dir=root, face_channels={})
        assert set(assets.behaviors) == {'step'}
        assert 'stretch' in assets.unavailable
        assert assets.idle == pose()


def test_wrong_face_track_length_is_reported_and_clip_excluded():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'base.json').write_text(json.dumps(encode_pose_payload(pose())))
        (root / 'clip.json').write_text(json.dumps([encode_pose_payload(pose())] * 3))
        (root / 'face.json').write_text('[0, 1]')
        assets = MotionAssets.load(root / 'base.json', {'stretch': {
            'source': 'clip', 'frames': 'clip.json', 'face_tracks': {'expression': 'face.json'},
        }}, base_dir=root, face_channels={'expression': 'Expression'})
        assert not assets.behaviors
        assert 'length' in assets.unavailable['stretch']


def test_profile_and_doctor_distinguish_avatar_from_optional_behaviors():
    from tests.test_provider_profiles import profile_data
    from vrc_ardy_agent.runner.profile import Profile
    from vrc_ardy_agent.runner.diagnostics import doctor
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'base.json').write_text(json.dumps(encode_pose_payload(pose())))
        data = profile_data()
        data['avatar'] = {'rig': RIG_PATH, 'base_pose': 'base.json', 'face_channels': {}, 'hmd_base': [0, 1, 0]}
        data['avatar']['calibration'] = attest_calibration(data['avatar'], root, tracking_active=True)
        data['behaviors'] = {'stretch': {'source': 'clip', 'frames': 'absent.json'}}
        data['autonomy'] = {'enabled': False, 'interval_seconds': 30, 'behaviors': []}
        profile = Profile.parse(data, root / 'agent.json')
        rows = doctor(profile, offline=True)
        assert next(row for row in rows if row['service'] == 'avatar')['result'] == 'pass'
        assert next(row for row in rows if row['service'] == 'behavior:stretch')['result'] == 'warning'


def test_attested_clip_is_invalidated_when_motion_or_face_file_changes():
    from vrc_ardy_agent.runner.avatar_setup import setup_action
    from vrc_ardy_agent.avatar_config import signature
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'base.json').write_text(json.dumps(encode_pose_payload(pose())))
        (root / 'clip.json').write_text(json.dumps([encode_pose_payload(pose())] * 2))
        avatar = {'rig': RIG_PATH, 'base_pose': 'base.json', 'face_channels': {}, 'hmd_base': [0, 1, 0]}
        definitions = setup_action('behaviors.attest', {'avatar': avatar, 'behaviors': {
            'stretch': {'source': 'clip', 'frames': 'clip.json'}}, 'tracking_active': True}, base=root)
        assets = MotionAssets.load(root / 'base.json', definitions, base_dir=root,
                                   face_channels={}, avatar_signature=signature(avatar, root))
        assert 'stretch' in assets.behaviors
        (root / 'clip.json').write_text(json.dumps([encode_pose_payload(pose(.2))] * 2))
        assets = MotionAssets.load(root / 'base.json', definitions, base_dir=root,
                                   face_channels={}, avatar_signature=signature(avatar, root))
        assert 'stretch' in assets.unavailable
        assert 'changed' in assets.unavailable['stretch']


def test_api_and_brain_advertise_only_the_runtime_registry():
    from vrc_ardy_agent.companion_api import CompanionControlService
    from vrc_ardy_agent.character_supervisor import CharacterSupervisor
    from vrc_ardy_agent.interaction_runtime import InteractionRuntime
    from vrc_ardy_agent.providers.decision import messages
    from tests.test_interaction_runtime import _StaticBrain, _FakeMotionDirector, _BlockingExecutor
    from vrc_ardy_agent.action_contracts import ActionType
    spec = BehaviorSpec.parse('stretch', {'source': 'ardy', 'prompt': 'Stretch gently.', 'duration_seconds': 3})
    brain = _StaticBrain({'actions': [{'type': 'motion', 'name': 'stretch'}]})
    motion = _FakeMotionDirector()
    runtime = InteractionRuntime(brain=brain, supervisor=CharacterSupervisor(executors={ActionType.SAY: _BlockingExecutor()}),
                                 motion_director=motion, behaviors={'stretch': spec})
    service = CompanionControlService(runtime)
    assert service.actions()['actions']['motion']['names'] == ['stretch']
    result = runtime.handle_utterance(transcript='stretch please', screenshot=None)
    assert result.status.value == 'APPLIED'
    context = json.loads(messages(brain.requests[0])[1]['content'][0]['text'])
    assert [row['name'] for row in context['available_motions']] == ['stretch']
    failed = runtime.apply_action_payload({'actions': [{'type': 'motion', 'name': 'missing'}]}, source='test')
    assert failed.status.value == 'FAILED'
    assert len(motion.cues) == 1


def test_saved_clip_can_end_anywhere_and_preemption_blends_from_last_sent_pose():
    from tests.test_anchored_motion import Runtime, Mapper
    from tests.test_motion_replay import Sink
    from vrc_ardy_agent.anchored_motion import AnchoredMotionSession
    from vrc_ardy_agent.motion_stream import CALIBRATED_IDLE
    spec = BehaviorSpec.parse('stretch', {'source': 'clip', 'frames': 'clip.json'})
    step = BehaviorSpec.parse('step', {'source': 'locomotion', 'velocity': [0, .18, 0], 'duration_seconds': 1})
    assets = MotionAssets(pose(), {'stretch': spec, 'step': step}, {'stretch': MotionClip((pose(1), pose(2)))})
    sink = Sink()
    session = AnchoredMotionSession(runtime=Runtime(), mapper=Mapper(), sink=sink, assets=assets, transition_seconds=.1)
    session.start(CALIBRATED_IDLE)
    try:
        session.send_next()
        revision = session.set_prompt('preset:stretch')
        session.send_next()
        assert sink.frames[-1].left.position[0] == .5
        assert session.completed_revision != revision
        session.send_next(); session.send_next()
        assert session.completed_revision == revision
        assert sink.frames[-1] == pose(2)
        session.set_prompt(CALIBRATED_IDLE)
        session.send_next()
        assert sink.frames[-1].left.position[0] == 1
        session.send_next()
        assert sink.frames[-1] == pose()
        session.set_prompt('preset:stretch'); session.send_next()
        session.set_prompt(CALIBRATED_IDLE); session.send_next()
        assert sink.frames[-1].left.position[0] == .25
        session.send_next()
        assert sink.frames[-1] == pose()
        session.set_prompt('preset:stretch')
        session.send_next(); session.send_next(); session.send_next()
        session.set_prompt('preset:step'); session.send_next()
        assert sink.frames[-1].left.position[0] == 1
        assert sink.frames[-1].locomotion_y == 0
        session.send_next()
        assert sink.frames[-1].left.position[0] == 0
        assert sink.frames[-1].locomotion_y == .18
    finally:
        session.stop()


def test_first_base_frame_resets_every_configured_face_parameter():
    from tests.test_live_sink import _FakeSocket
    from vrc_ardy_agent.live_sink import SixPointUdpSink
    import struct
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'base.json').write_text(json.dumps(encode_pose_payload(pose())))
        assets = MotionAssets.load(root / 'base.json', {}, base_dir=root,
                                  face_channels={'eyes': 'EyeClosure', 'mouth': 'MouthOpen'})
        sock = _FakeSocket()
        sink = SixPointUdpSink(host='127.0.0.1', send_face=True, socket_factory=lambda: sock)
        try:
            sink.send(assets.idle)
            packets = [packet for packet, _ in sock.sent if packet.startswith(b'/avatar/parameters/')]
            assert len(packets) == 2
            assert all(struct.unpack('>f', packet[-4:])[0] == 0 for packet in packets)
        finally: sink.close()
