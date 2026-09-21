import json
from tests.test_avatar_rig_profile import _profile_document
from copy import deepcopy
import pytest
from vrc_ardy_agent.avatar_rig_profile import AvatarRigProfile
from vrc_ardy_agent.calibration_pose import candidate, adjust


def test_candidate_is_neutral_with_active_hands_and_scales_with_height():
    rig=AvatarRigProfile.from_mapping(_profile_document())
    a=candidate(rig,[0,1,0]); b=candidate(rig,[0,2,0])
    assert a['tracker_activation']['left'] and a['tracker_activation']['right']
    assert a['locomotion']==[0,0,0]
    assert a['trackers']['left']['position'][0]<0<a['trackers']['right']['position'][0]
    for x,y in zip(a['trackers']['left']['position'],b['trackers']['left']['position']):
        assert y==pytest.approx(x*2)


def test_adjustment_moves_only_selected_hand_and_keeps_original():
    rig=AvatarRigProfile.from_mapping(_profile_document())
    pose=candidate(rig,[0,1,0]); old=deepcopy(pose)
    changed=adjust(pose,'left','y',.01)
    assert pose==old
    assert changed['trackers']['right']==pose['trackers']['right']
    assert changed['trackers']['left']['position'][1]==pytest.approx(pose['trackers']['left']['position'][1]+.01)
    turned=adjust(changed,'both','rx',5)
    for side in ('left','right'):
        assert sum(v*v for v in turned['trackers'][side]['quaternion_xyzw'])==pytest.approx(1)
    with pytest.raises(ValueError): adjust(pose,'head','y',.01)
    with pytest.raises(ValueError): adjust(pose,'left','y',float('nan'))


def test_core_prepares_and_checkpoints_without_models_or_an_active_profile(tmp_path):
    from vrc_ardy_agent.runner.control import dispatch
    from pathlib import Path
    rig_path = tmp_path / 'synthetic-rig.json'
    rig_path.write_text(json.dumps(_profile_document()))
    avatar=dict(rig=str(rig_path),
                base_pose='',hmd_base=[0,1,0],face_channels={})
    prepared=dispatch('calibration.prepare',{'avatar':avatar},path=tmp_path/'agent.json')
    original=Path(prepared['base_pose']).read_bytes()
    changed=dispatch('calibration.adjust',{'avatar':prepared,'hand':'left','axis':'y','delta':.01},path=tmp_path/'agent.json')
    assert changed['base_pose']!=prepared['base_pose']
    assert Path(prepared['base_pose']).read_bytes()==original
    assert 'calibration' not in changed
