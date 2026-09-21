import pytest
from tests.test_tracker_device_controller import _PoseSink
from tests.test_device_payloads import _frame
from tests.test_device_gateway import _RecordingDeviceSink
from vrc_ardy_agent.device_gateway import DeviceGateway
from vrc_ardy_agent.calibration_session import CalibrationSession
from vrc_ardy_agent.device_payloads import encode_pose_payload
from vrc_ardy_agent.tracker_device_controller import TrackerDeviceController
from vrc_ardy_agent.windows_device_sink import WindowsDeviceSink
from types import SimpleNamespace


@pytest.fixture
def session():
    sink=_PoseSink(); tracker=TrackerDeviceController(sink=sink,safe_frame=_frame())
    tracker.start()
    gateway=DeviceGateway(sink=WindowsDeviceSink(pose_sink=tracker,wave_player=SimpleNamespace(stop=lambda:None)))
    clock=[0.]
    service=CalibrationSession(gateway,tracker,hmd_base=[0,1,0],clock=lambda:clock[0])
    yield service,gateway,tracker,clock,sink
    service.close();tracker.close()


def start(service):
    pose=encode_pose_payload(_frame()); pose['locomotion']=[0,0,0]
    return service.start({'request_id':'first','pose':pose,'hmd_base':[0,1,0],'rig_hash':'a'*64})


def test_session_excludes_agent_and_second_wizard_then_restores(session):
    s,g,t,clock,sink=session
    a=start(s)
    assert a['state']=='active' and a['revision']==0
    assert start(s)['token']==a['token']
    with pytest.raises(RuntimeError):g.open_session('agent')
    with pytest.raises(RuntimeError):s.start({'request_id':'other','pose':a['pose'],'hmd_base':[0,1,0],'rig_hash':'a'*64})
    end=s.finish({'token':a['token'],'revision':0,'verified':False})
    assert end['state']=='canceled' and end['restored']
    assert t.snapshot().mode=='SAFE'
    assert s.finish({'token':a['token'],'revision':0,'verified':False})==end
    g.open_session('agent')
    with pytest.raises(RuntimeError):start(s)


def test_stale_update_and_expired_session_cannot_reactivate_output(session):
    s,g,t,clock,sink=session
    a=start(s)
    updated=s.update({'token':a['token'],'revision':0,'pose':a['pose']})
    assert updated['revision']==1
    with pytest.raises(ValueError):s.update({'token':a['token'],'revision':0,'pose':a['pose']})
    clock[0]=10
    s.poll()
    assert s.status({'token':a['token']})['state']=='expired'
    assert t.snapshot().mode=='SAFE'
    with pytest.raises(RuntimeError):s.heartbeat({'token':a['token']})


def test_finish_verifies_exact_revision_and_reports_applied_pose(session):
    s,g,t,clock,sink=session
    a=start(s)
    with pytest.raises(ValueError):s.finish({'token':a['token'],'revision':3,'verified':True})
    receipt=s.finish({'token':a['token'],'revision':0,'verified':True})
    assert receipt['state']=='confirmed' and receipt['pose']==a['pose']
    assert receipt['restored']
    assert g.snapshot().session_id is None


def test_output_failure_is_visible_and_never_confirms(session):
    s,g,t,clock,sink=session
    a=start(s)
    def fail(_):raise OSError('device unavailable')
    sink.send=fail
    with pytest.raises(OSError):s.update({'token':a['token'],'revision':0,'pose':a['pose']})
    state=s.status({'token':a['token']})
    assert state['state']=='failed' and not state['restored']
    assert state['last_error']
