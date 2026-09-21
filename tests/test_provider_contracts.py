import json
import pytest
from tests.test_provider_adapters import Recorder, Response
from vrc_ardy_agent.providers.catalog import definition, validate_binding, validate_connection
from vrc_ardy_agent.providers.factory import build_connection
from vrc_ardy_agent.providers.discovery import discover_items


def test_presets_share_protocols_but_keep_extensions():
    assert definition('hermes')['protocol'] == definition('crisperwhisper')['protocol'] == definition('gpt-sovits')['protocol'] == 'openai-compatible'
    assert definition('hermes')['extension'] == 'hermes'
    assert definition('crisperwhisper')['extension'] == 'crisper'
    assert definition('elevenlabs')['protocol'] == 'elevenlabs'
    assert definition('whisper-local')['setup']['category'] == 'local'


def test_generic_connection_can_explicitly_use_crisper_contract():
    config = {'base_url':'http://example.test/v1', 'request_extension':'crisper'}
    binding = validate_binding('openai-compatible','stt',{'model':'m','mode':'intended'}, config=config)
    recorder = Recorder(Response({'text':'hello'}))
    adapter = build_connection('openai-compatible',config,'stt',binding,opener=recorder)
    assert adapter.transcribe_pcm(b'\0\0'*160) == 'hello'
    assert b'name="mode"\r\n\r\nintended' in recorder.calls[0][0].data
    with pytest.raises(ValueError):
        validate_binding('openai-compatible','llm',{'model':'m'},config=config)


def test_voice_discovery_uses_declared_contract_and_source():
    recorder = Recorder(Response({'status':'ok','voices':['voice-a','voice-b']}))
    result = discover_items('gpt-sovits',{},kind='voices',opener=recorder)
    assert recorder.calls[0][0].full_url == 'http://127.0.0.1:8001/health'
    assert result['source'] == 'server'
    assert [x['id'] for x in result['items']] == ['voice-a','voice-b']
    untouched = Recorder(Response({}))
    result = discover_items('openai-compatible',{'base_url':'http://example.test/v1'},kind='voices',opener=untouched)
    assert result['state'] == 'unsupported' and untouched.calls == []


def test_custom_voice_path_preserves_prefix_and_rejects_other_origins():
    recorder = Recorder(Response({'voices':['one']}))
    result = discover_items('openai-compatible',{'base_url':'http://example.test/proxy/v1','voices_path':'audio/voices'},kind='voices',opener=recorder)
    assert result['state'] == 'ready'
    assert recorder.calls[0][0].full_url == 'http://example.test/proxy/v1/audio/voices'
    for path in ('https://elsewhere.test/voices','//elsewhere.test/voices','../voices','/../voices','%2e%2e/voices'):
        with pytest.raises(ValueError):validate_connection('openai-compatible',{'base_url':'http://example.test/v1','voices_path':path})


def test_documented_voices_do_not_require_credentials_or_inference():
    recorder = Recorder(Response({}))
    result = discover_items('openai',{},kind='voices',environment={},opener=recorder)
    assert result['state'] == 'ready' and result['source'] == 'preset'
    assert result['items'] and recorder.calls == []


def test_declared_streaming_wav_contract_normalizes_only_completed_sentinel_frames():
    from io import BytesIO
    import wave,struct
    buffer=BytesIO()
    with wave.open(buffer,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(32000);wav.writeframes(b'\0\0'*100)
    raw=bytearray(buffer.getvalue());raw[4:8]=struct.pack('<I',0xfffffff7);raw[40:44]=struct.pack('<I',0xffffffff)
    recorder=Recorder(Response(bytes(raw),'audio/wav'))
    a=build_connection('gpt-sovits',{},'tts',{'model':'m','voice':'v'},opener=recorder)
    with wave.open(BytesIO(a.synthesize('test'))) as wav:assert wav.getnframes()==100
    a=build_connection('openai-compatible',{'base_url':'http://example.test/v1'},'tts',{'model':'m','voice':'v'},opener=recorder)
    with pytest.raises(RuntimeError):a.synthesize('test')


def test_incomplete_http_body_is_not_accepted_as_streaming_audio():
    from http.client import IncompleteRead
    def fail(*a,**kw):raise IncompleteRead(b'partial audio')
    from vrc_ardy_agent.providers.probe import probe_connection
    result=probe_connection('gpt-sovits',{},'tts',{'model':'m','voice':'v'},opener=fail)
    assert result['state']=='failed' and result['reason']=='network-error'
