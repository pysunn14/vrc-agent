import json
from tests.test_provider_adapters import Recorder, Response
from vrc_ardy_agent.providers.probe import probe_connection
from vrc_ardy_agent.runner.provider_observations import ObservationStore, fingerprint


def sample(**kw):
    return dict(provider='openai-compatible',config={'base_url':'http://example.test/v1'},capability='llm',settings={'model':'m'},**kw)


def test_unsaved_http_probe_needs_no_profile_or_ardy():
    response = Recorder(Response({'choices':[{'message':{'content':'OK'}}]}))
    result = probe_connection(**sample(),opener=response)
    assert result['state'] == 'success' and result['checks']['text'] == 'success'
    assert json.loads(response.calls[0][0].data)['model'] == 'm'
    assert 'OK' not in json.dumps(result)


def test_image_probe_is_separate_and_uses_synthetic_image():
    response = Recorder(Response({'choices':[{'message':{'content':'OK'}}]}))
    args = sample(); args['settings']['vision'] = True
    result = probe_connection(**args,opener=response)
    assert result['checks'] == {'text':'success','image_input':'success'}
    assert len(response.calls) == 2
    payload = json.loads(response.calls[1][0].data)
    assert payload['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')


def test_failed_probe_has_structured_error_not_response_text():
    response = Recorder(Response({'secret':'do not emit'})); response.response.status=401
    result = probe_connection(**sample(),opener=response)
    assert result['state']=='failed' and result['reason']=='auth-error'
    assert 'do not emit' not in json.dumps(result)


def test_history_fingerprint_and_concurrent_completion(tmp_path):
    args = sample(); key = fingerprint(**args)
    store = ObservationStore(tmp_path)
    first = store.begin(key); second = store.begin(key)
    assert not store.finish(key,first,{'state':'success'})
    assert store.finish(key,second,{'state':'failed','reason':'network-error'})
    assert store.read(key)['result']['state']=='failed'
    changed=sample(); changed['settings']['model']='other'
    assert fingerprint(**changed)!=key
    assert store.read(fingerprint(**changed)) is None
    assert not list(tmp_path.glob('*.wav'))


def test_bundled_sample_is_finite_pcm_with_recorded_provenance():
    import hashlib
    import wave
    from pathlib import Path
    root=Path('vrc_ardy_agent/providers/assets')
    metadata=json.loads((root/'speech-sample.json').read_text())
    assert hashlib.sha256((root/'speech-sample.wav').read_bytes()).hexdigest()==metadata['sha256']
    with wave.open(str(root/'speech-sample.wav')) as wav:
        assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())==(16000,1,2)
        assert len(wav.readframes(wav.getnframes()))==wav.getnframes()*2


def test_stt_and_tts_probe_use_production_contracts():
    from io import BytesIO
    import wave
    stt=Recorder(Response({'text':'test speech'}))
    result=probe_connection('crisperwhisper',{},'stt',{'model':'m','mode':'intended'},opener=stt)
    assert result['state']=='success'
    assert b'name="mode"\r\n\r\nintended' in stt.calls[0][0].data
    buffer=BytesIO()
    with wave.open(buffer,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(b'\0\0'*100)
    tts=Recorder(Response(buffer.getvalue(),'audio/wav'))
    result=probe_connection('gpt-sovits',{},'tts',{'model':'m','voice':'v'},opener=tts)
    assert result['checks']['sample_rate']==24000
    assert json.loads(tts.calls[0][0].data)['voice']=='v'


def test_observation_is_available_before_profile_and_changes_with_config(tmp_path,monkeypatch):
    from vrc_ardy_agent.runner.control import dispatch
    import vrc_ardy_agent.runner.connection_tests as module
    monkeypatch.setattr(module,'probe_connection',lambda **kw:{'state':'success','checks':{'text':'success'}})
    args=sample();path=tmp_path/'no-profile.json'
    result=dispatch('connection.test',args,path=path,state_dir=tmp_path)
    observed=dispatch('connection.observation',args,path=path,state_dir=tmp_path)
    assert observed['historical'] is True
    assert observed['observation']['result']['state']=='success'
    assert result['recorded'] is True and not path.exists()
    args['config']['credential_ref']='changed'
    assert dispatch('connection.observation',args,path=path,state_dir=tmp_path)['observation'] is None


def test_non_english_setting_requires_matching_sample_instead_of_silent_override():
    recorder=Recorder(Response({'text':'unexpected'}))
    result=probe_connection('crisperwhisper',{},'stt',{'model':'m','language':'ko'},opener=recorder)
    assert result['state']=='failed' and result['reason']=='sample-language-mismatch'
    assert recorder.calls==[]


def test_dead_probe_owner_is_observed_as_interrupted(tmp_path):
    from vrc_ardy_agent.runner.settings import atomic_json
    store=ObservationStore(tmp_path);key=fingerprint(**sample());store.begin(key)
    raw=store.read(key);raw['created']=-1
    atomic_json(store.path(key),raw)
    assert store.read(key)['state']=='interrupted'


def test_local_probe_supervisor_exits_when_original_parent_is_gone():
    import os,subprocess,sys
    import pytest
    if os.name=='nt':pytest.skip('native Windows process handle requires Windows runner')
    code='from vrc_ardy_agent.providers.probe_lifecycle import watch_parent; watch_parent(0)'
    result=subprocess.run([sys.executable,'-c',code],timeout=5,capture_output=True)
    assert result.returncode==130
