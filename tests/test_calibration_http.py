import json
from tests.test_avatar_rig_profile import _profile_document
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest

from tests.test_tracker_device_controller import _PoseSink
from tests.test_device_payloads import _frame
from vrc_ardy_agent.tracker_device_controller import TrackerDeviceController
from vrc_ardy_agent.windows_device_sink import WindowsDeviceSink
from vrc_ardy_agent.device_gateway import DeviceGateway
from vrc_ardy_agent.calibration_session import CalibrationSession
from vrc_ardy_agent.windows_companion_api import WindowsCompanionControlService
from vrc_ardy_agent.companion_api import create_companion_server
from vrc_ardy_agent.companion_http_runtime import CompanionHttpRuntime
from vrc_ardy_agent.runner.control import dispatch
from vrc_ardy_agent.avatar_config import inspect_avatar


def test_real_http_prepare_adjust_confirm_persist_and_restore(tmp_path):
    tracker=TrackerDeviceController(sink=_PoseSink(),safe_frame=_frame());tracker.start()
    gateway=DeviceGateway(sink=WindowsDeviceSink(pose_sink=tracker,wave_player=SimpleNamespace(stop=lambda:None)))
    calibration=CalibrationSession(gateway,tracker,hmd_base=[0,1,0]);calibration.start_watchdog()
    service=WindowsCompanionControlService(SimpleNamespace(snapshot=lambda:{'running':True}),shutdown_requested=threading.Event(),calibration=calibration)
    http=CompanionHttpRuntime(create_companion_server(('127.0.0.1',0),service));http.start()
    try:
        config=tmp_path/'agent.json'
        def call(action,payload):return dispatch('calibration.'+action,payload,path=config)
        target={'host':'127.0.0.1','port':http.snapshot().bound_port}
        rig_path = tmp_path / 'synthetic-rig.json'
        rig_path.write_text(json.dumps(_profile_document()))
        avatar=call('prepare',{'avatar':dict(rig=str(rig_path),base_pose='',hmd_base=[0,1,0],face_channels={})})
        assert call('status',target)['available']
        started=call('start',{**target,'avatar':avatar,'request_id':'http-test'})
        modified=call('adjust',dict(avatar=avatar,hand='left',axis='y',delta=.01))
        with pytest.raises(ValueError,match='differs'):
            call('complete',{**target,'avatar':modified,'token':started['token'],'revision':0,'verified':True})
        current=call('update',{**target,'avatar':modified,'token':started['token'],'revision':0})
        completed=call('complete',{**target,'avatar':modified,'token':started['token'],'revision':current['revision'],'verified':True})
        assert inspect_avatar(completed['avatar'],tmp_path)['ready']
        assert completed['receipt']['restored']
        assert tracker.snapshot().mode=='SAFE'
        assert call('status',target)['available']
    finally:
        http.stop();calibration.close();tracker.close()


def test_target_epoch_change_cancels_preview_before_confirmation():
    events=[]
    sensors={'running':True,'state':'RUNNING','epoch':1}
    record={'token':'test','revision':0,'state':'active'}
    calibration=SimpleNamespace(
        status=lambda *args:dict(record),
        start=lambda payload:dict(record),
        finish=lambda payload:events.append(payload) or {'state':'canceled'},
    )
    service=WindowsCompanionControlService(
        SimpleNamespace(snapshot=lambda:{'running':True,'sensor_supervisor':sensors}),
        shutdown_requested=threading.Event(),calibration=calibration)
    service.calibrate({'action':'start'})
    sensors['epoch']=2
    with pytest.raises(RuntimeError,match='target changed'):
        service.calibrate({'action':'finish','token':'test','revision':0,'verified':True})
    assert events == [{'token':'test','revision':0,'verified':False}]
