from __future__ import annotations

import json
import threading
import unittest

from vrc_ardy_agent.action_contracts import (
    ControlResource,
    ExecutionCommand,
    SayAction,
)
from vrc_ardy_agent.speech_executor import GptSovitsClient, SpeechExecutor
from vrc_ardy_agent.speech_executor import MAX_WAV_BYTES
from vrc_ardy_agent.device_payloads import MAX_WAV_PAYLOAD_BYTES


class _FakeResponse:
    status = 200

    def __init__(self, body: bytes, *, content_type: str = "audio/wav") -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        return self._body if amount < 0 else self._body[:amount]


class _RecordingOpener:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return self.response


class _StaticSynthesizer:
    def __init__(self, wav: bytes = b"RIFF-audio") -> None:
        self.wav = wav
        self.texts: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        return self.wav


class _BlockingSynthesizer(_StaticSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        self.started.set()
        self.release.wait(timeout=1.0)
        return self.wav


class _RecordingSpeechOutput:
    def __init__(self) -> None:
        self.calls = []

    def play_wav(self, command, wav_bytes, cancel_event) -> None:
        self.calls.append((command, wav_bytes, cancel_event.is_set()))


def _command(text: str = "안녕.") -> ExecutionCommand:
    return ExecutionCommand(
        action_id="speech-1",
        turn_id="turn-1",
        action=SayAction(text=text),
        resource=ControlResource.VOICE_OUTPUT,
        lease_token=3,
    )


class GptSovitsClientTests(unittest.TestCase):
    def test_synthesis_limit_matches_the_cross_machine_payload_limit(self) -> None:
        self.assertEqual(MAX_WAV_BYTES, MAX_WAV_PAYLOAD_BYTES)

    def test_synthesize_uses_the_charlotte_openai_compatible_endpoint(self) -> None:
        opener = _RecordingOpener(_FakeResponse(b"RIFF-generated-wav"))
        client = GptSovitsClient(opener=opener, timeout_seconds=21)

        result = client.synthesize("안녕하세요.")

        self.assertEqual(result, b"RIFF-generated-wav")
        self.assertEqual(len(opener.requests), 1)
        request = opener.requests[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8001/v1/audio/speech",
        )
        self.assertEqual(
            json.loads(request.data),
            {
                "model": "gpt-sovits-v2proplus",
                "input": "안녕하세요.",
                "voice": "charlotte",
                "response_format": "wav",
            },
        )
        self.assertEqual(opener.timeouts, [21.0])


class SpeechExecutorTests(unittest.TestCase):
    def test_marks_running_only_after_synthesis_then_plays_the_wav(self) -> None:
        synthesizer = _StaticSynthesizer()
        output = _RecordingSpeechOutput()
        executor = SpeechExecutor(synthesizer=synthesizer, output=output)
        marked = []

        executor.run(_command(), threading.Event(), lambda: marked.append(True))

        self.assertEqual(synthesizer.texts, ["안녕."])
        self.assertEqual(marked, [True])
        self.assertEqual(len(output.calls), 1)
        self.assertEqual(output.calls[0][0].action_id, "speech-1")
        self.assertEqual(output.calls[0][1], b"RIFF-audio")

    def test_cancelled_synthesis_is_discarded_before_playback(self) -> None:
        synthesizer = _BlockingSynthesizer()
        output = _RecordingSpeechOutput()
        executor = SpeechExecutor(synthesizer=synthesizer, output=output)
        cancel = threading.Event()
        marked = []
        thread = threading.Thread(
            target=executor.run,
            args=(_command(), cancel, lambda: marked.append(True)),
        )
        thread.start()
        self.assertTrue(synthesizer.started.wait(timeout=1.0))

        cancel.set()
        synthesizer.release.set()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(marked, [])
        self.assertEqual(output.calls, [])


if __name__ == "__main__":
    unittest.main()
