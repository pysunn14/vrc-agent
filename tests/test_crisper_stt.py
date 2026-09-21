from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError

from vrc_ardy_agent.crisper_stt import (
    CrisperWhisperClient,
    TranscriptionError,
)


class _Response:
    status = 200

    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        return self._body if amount < 0 else self._body[:amount]


class _Opener:
    def __init__(self, payload: object) -> None:
        self.response = _Response(payload)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return self.response


class CrisperWhisperClientTests(unittest.TestCase):
    def test_posts_16khz_mono_wav_with_explicit_language_and_intended_mode(self) -> None:
        opener = _Opener(
            {"text": "춤춰 줘", "language": "ko", "mode": "intended"}
        )
        client = CrisperWhisperClient(
            language="ko",
            mode="intended",
            timeout_seconds=12,
            boundary_factory=lambda: "test-boundary",
            opener=opener,
        )

        result = client.transcribe_pcm(b"\x01\x00" * 320)

        self.assertEqual(result, "춤춰 줘")
        request = opener.requests[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8002/v1/audio/transcriptions",
        )
        self.assertIn("multipart/form-data; boundary=test-boundary", request.get_header("Content-type"))
        self.assertIn(b'\r\n\r\nko\r\n', request.data)
        self.assertIn(b'\r\n\r\nintended\r\n', request.data)
        self.assertIn(b"RIFF", request.data)
        self.assertIn(b"WAVE", request.data)
        self.assertEqual(opener.timeouts, [12.0])

    def test_empty_transcript_is_a_failure_not_a_silent_result(self) -> None:
        client = CrisperWhisperClient(opener=_Opener({"text": "  "}))

        with self.assertRaisesRegex(TranscriptionError, "empty"):
            client.transcribe_pcm(b"\x00\x00" * 320)

    def test_http_failure_is_not_retried_or_replaced(self) -> None:
        calls = []

        def fail(request, *, timeout):
            calls.append(request)
            raise HTTPError(request.full_url, 400, "bad language", {}, None)

        client = CrisperWhisperClient(opener=fail)

        with self.assertRaisesRegex(TranscriptionError, "HTTP 400"):
            client.transcribe_pcm(b"\x00\x00" * 320)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
