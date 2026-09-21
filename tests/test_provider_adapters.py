import io
import json
import time
import unittest
import wave
from dataclasses import replace

from tests.test_hermes_brain import _brain_request
from tests.test_provider_profiles import profile_data
from vrc_ardy_agent.providers.factory import build_provider
from vrc_ardy_agent.runner.profile import Profile


class Response:
    status = 200
    def __init__(self, body, content_type="application/json"):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.headers = {"Content-Type": content_type}
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def read(self, n=-1): return self.body[:n] if n >= 0 else self.body


class Recorder:
    def __init__(self, response): self.response, self.calls = response, []
    def __call__(self, request, **kwargs):
        self.calls.append((request, kwargs)); return self.response


class AdapterTests(unittest.TestCase):
    def test_hermes_and_crisper_extensions_stay_in_their_own_adapters(self):
        data = profile_data()
        data["providers"]["chat"].update(provider="hermes", config={"api_key_env": "KEY"})
        data["bindings"]["llm"].update(upstream_provider="selected-upstream", reasoning_effort="low")
        opener = Recorder(Response({"choices": [{"message": {"content": '{"requires_visual":false,"actions":[{"type":"say","text":"Hi"}]}'}}]}))
        adapter = build_provider(Profile.parse(data), "llm", environment={"KEY": "example"}, opener=opener)
        adapter.propose(replace(_brain_request(), screenshot=None))
        sent = json.loads(opener.calls[0][0].data)
        self.assertEqual(sent["provider"], "selected-upstream")
        self.assertEqual(sent["model_options"], {"reasoning_effort": "low"})
        data = profile_data()
        data["providers"]["input"] = {"provider": "crisperwhisper", "label": "Input", "config": {},
                                      "deployment": {"kind": "external", "host": "runner"}}
        data["bindings"]["stt"].update(instance="input", mode="verbatim")
        opener = Recorder(Response({"text": "hello"}))
        build_provider(Profile.parse(data), "stt", opener=opener).transcribe_pcm(b"\0\0" * 160)
        self.assertIn(b'name="mode"\r\n\r\nverbatim', opener.calls[0][0].data)

    def test_local_whisper_loads_once_and_cannot_download_implicitly(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        model = Mock()
        model.transcribe.return_value = ([SimpleNamespace(text=" hello ")], None)
        create = Mock(return_value=model)
        data = profile_data()
        data["providers"]["local"] = {"provider": "whisper-local", "label": "Local", "config": {},
                                      "deployment": {"kind": "embedded", "host": "runner"}}
        data["bindings"]["stt"].update(instance="local", model="/models/installed-whisper")
        with patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=create)}):
            adapter = build_provider(Profile.parse(data), "stt")
            self.assertEqual(adapter.transcribe_pcm(b"\0\0" * 160), "hello")
            self.assertEqual(adapter.transcribe_pcm(b"\0\0" * 160), "hello")
        create.assert_called_once_with("/models/installed-whisper", device="cpu", compute_type="int8", local_files_only=True)

    def test_compatible_chat_preserves_decision_contract_without_vendor_fields(self):
        plan = {"requires_visual": False, "actions": [{"type": "say", "text": "Hello"}]}
        opener = Recorder(Response({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(plan)}}]}))
        data = profile_data(); data["bindings"]["llm"]["vision"] = True
        adapter = build_provider(Profile.parse(data), "llm", opener=opener)
        self.assertEqual(adapter.propose(_brain_request()), plan)
        request = opener.calls[0][0]; payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://agent-host:8000/v1/chat/completions")
        self.assertNotIn("provider", payload)
        self.assertNotIn("model_options", payload)
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")

    def test_missing_key_unsupported_vision_and_expired_deadline_fail_before_call(self):
        data = profile_data(); profile = Profile.parse(data)
        opener = Recorder(Response({}))
        with self.assertRaisesRegex(ValueError, "TEST_AUDIO_KEY"):
            build_provider(profile, "tts", environment={})
        adapter = build_provider(profile, "llm", opener=opener)
        with self.assertRaisesRegex(ValueError, "vision"):
            adapter.propose(_brain_request())
        request = replace(_brain_request(), screenshot=None, deadline_monotonic=time.monotonic()-1)
        with self.assertRaises(TimeoutError): adapter.propose(request)
        self.assertEqual(opener.calls, [])

    def test_standard_transcription_does_not_send_crisper_mode(self):
        opener = Recorder(Response({"text": " hello "}))
        adapter = build_provider(Profile.parse(profile_data()), "stt", environment={"TEST_AUDIO_KEY": "key"}, opener=opener)
        self.assertEqual(adapter.transcribe_pcm(b"\0\0"*160), "hello")
        request = opener.calls[0][0]
        self.assertNotIn(b'name="mode"', request.data)
        self.assertIn(b'filename="utterance.wav"', request.data)
        self.assertEqual(request.get_header("Authorization"), "Bearer key")

    def test_elevenlabs_raw_pcm_becomes_valid_wav(self):
        data = profile_data()
        data["providers"]["voice"] = {"provider": "elevenlabs", "label": "Voice",
            "config": {"api_key_env": "VOICE_KEY"}, "deployment": {"kind": "api"}}
        data["bindings"]["tts"].update(instance="voice")
        opener = Recorder(Response(b"\0\0"*100, "audio/pcm"))
        adapter = build_provider(Profile.parse(data), "tts", environment={"VOICE_KEY": "key"}, opener=opener)
        audio = adapter.synthesize("Hello")
        with wave.open(io.BytesIO(audio)) as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (24000, 1, 2))
            self.assertEqual(wav.getnframes(), 100)
        request = opener.calls[0][0]
        self.assertIn("text-to-speech/voice-id?output_format=pcm_24000", request.full_url)
        self.assertEqual(json.loads(request.data), {"text": "Hello", "model_id": "speech-model"})
        self.assertEqual(request.get_header("Xi-api-key"), "key")

    def test_tts_rejects_json_and_truncated_wav(self):
        for response in (Response(b'{"error":"failure"}'), Response(b'RIFFbad', "audio/wav")):
            adapter = build_provider(Profile.parse(profile_data()), "tts", environment={"TEST_AUDIO_KEY": "key"}, opener=Recorder(response))
            with self.assertRaises(RuntimeError): adapter.synthesize("Hello")
