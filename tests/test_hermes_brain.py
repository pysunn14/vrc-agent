from __future__ import annotations

import base64
import json
import os
from urllib.error import HTTPError
import unittest

from vrc_ardy_agent.hermes_brain import (
    HermesBrainAdapter,
    HermesBrainError,
    HermesBrainProtocolError,
)
from vrc_ardy_agent.interaction_runtime import (
    BodyActivity,
    BrainRequest,
    SpeechActivity,
)


class _FakeResponse:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

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


def _brain_request() -> BrainRequest:
    return BrainRequest(
        turn_id="turn-7",
        turn_version=7,
        transcript="손 흔들면서 인사해 줘",
        screenshot=b"jpeg-frame",
        body=BodyActivity.ARDY_MOTION,
        speech=SpeechActivity.IDLE,
        current_action_ids=("action-4",),
        last_error="previous speech failed",
    )


class HermesBrainAdapterTests(unittest.TestCase):
    def test_propose_posts_multimodal_request_and_returns_json_plan(self) -> None:
        plan = {
            "requires_visual": False,
            "actions": [
                {"type": "say", "text": "응, 안녕!"},
                {
                    "type": "ardy_motion",
                    "prompt": "A girl waves with her right hand.",
                    "duration_seconds": 3,
                },
            ]
        }
        opener = _RecordingOpener(
            _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(plan, ensure_ascii=False),
                            }
                        }
                    ]
                }
            )
        )
        adapter = HermesBrainAdapter(
            endpoint="http://127.0.0.1:8642/v1/chat/completions",
            api_key="secret-key",
            timeout_seconds=12.5,
            opener=opener,
        )

        result = adapter.propose(_brain_request())

        self.assertEqual(result, plan)
        self.assertEqual(len(opener.requests), 1)
        sent = opener.requests[0]
        self.assertEqual(
            sent.full_url,
            "http://127.0.0.1:8642/v1/chat/completions",
        )
        self.assertEqual(sent.get_header("Authorization"), "Bearer secret-key")
        self.assertEqual(sent.get_header("Idempotency-key"), "brain-turn-7-v7-r0")
        self.assertEqual(opener.timeouts, [12.5])

        payload = json.loads(sent.data)
        self.assertEqual(payload["model"], "gpt-5.6-luna")
        self.assertEqual(payload["provider"], "openai-codex")
        self.assertEqual(
            payload["model_options"],
            {"reasoning_effort": "low"},
        )
        self.assertFalse(payload["stream"])
        self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user"])
        user_parts = payload["messages"][1]["content"]
        self.assertEqual(user_parts[0]["type"], "text")
        turn_context = json.loads(user_parts[0]["text"])
        self.assertEqual(turn_context["transcript"], "손 흔들면서 인사해 줘")
        self.assertEqual(turn_context["runtime"]["body"], "ARDY_MOTION")
        self.assertEqual(turn_context["runtime"]["speech"], "IDLE")
        self.assertEqual(turn_context["runtime"]["current_action_ids"], ["action-4"])
        self.assertEqual(turn_context["runtime"]["last_error"], "previous speech failed")
        self.assertEqual(
            user_parts[1],
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(b"jpeg-frame").decode("ascii"),
                    "detail": "high",
                },
            },
        )

    def test_response_must_be_plain_json_without_markdown_fences(self) -> None:
        opener = _RecordingOpener(
            _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '```json\n{"actions": [{"type": "say", "text": "안녕."}]}\n```'
                            }
                        }
                    ]
                }
            )
        )
        adapter = HermesBrainAdapter(
            endpoint="http://127.0.0.1:8642/v1/chat/completions",
            api_key="secret-key",
            opener=opener,
        )

        with self.assertRaisesRegex(HermesBrainProtocolError, "plain JSON"):
            adapter.propose(_brain_request())

    def test_missing_assistant_content_is_a_protocol_error(self) -> None:
        opener = _RecordingOpener(_FakeResponse({"choices": []}))
        adapter = HermesBrainAdapter(
            endpoint="http://127.0.0.1:8642/v1/chat/completions",
            api_key="secret-key",
            opener=opener,
        )

        with self.assertRaisesRegex(HermesBrainProtocolError, "assistant content"):
            adapter.propose(_brain_request())

    def test_http_failure_does_not_expose_api_key(self) -> None:
        def fail(request, *, timeout):
            raise HTTPError(request.full_url, 503, "unavailable", {}, None)

        adapter = HermesBrainAdapter(
            endpoint="http://127.0.0.1:8642/v1/chat/completions",
            api_key="top-secret-key",
            opener=fail,
        )

        with self.assertRaises(HermesBrainError) as raised:
            adapter.propose(_brain_request())

        self.assertIn("503", str(raised.exception))
        self.assertNotIn("top-secret-key", str(raised.exception))

    def test_from_environment_uses_the_existing_hermes_gateway(self) -> None:
        adapter = HermesBrainAdapter.from_environment(
            {
                "API_SERVER_KEY": "configured-key",
                "VRC_ARDY_HERMES_TIMEOUT_SECONDS": "17",
            }
        )

        self.assertEqual(
            adapter.endpoint,
            "http://127.0.0.1:8642/v1/chat/completions",
        )
        self.assertEqual(adapter.model, "gpt-5.6-luna")
        self.assertEqual(adapter.provider, "openai-codex")
        self.assertEqual(adapter.reasoning_effort, "low")
        self.assertEqual(adapter.timeout_seconds, 17.0)
        self.assertEqual(
            adapter.snapshot(),
            {
                "endpoint": "http://127.0.0.1:8642/v1/chat/completions",
                "model": "gpt-5.6-luna",
                "provider": "openai-codex",
                "reasoning_effort": "low",
                "timeout_seconds": 17.0,
            },
        )

    def test_from_environment_accepts_an_explicit_request_route(self) -> None:
        adapter = HermesBrainAdapter.from_environment(
            {
                "API_SERVER_KEY": "configured-key",
                "VRC_ARDY_HERMES_MODEL": "gpt-5.6-sol",
                "VRC_ARDY_HERMES_PROVIDER": "openai-codex",
                "VRC_ARDY_HERMES_REASONING_EFFORT": "high",
            }
        )

        self.assertEqual(adapter.model, "gpt-5.6-sol")
        self.assertEqual(adapter.provider, "openai-codex")
        self.assertEqual(adapter.reasoning_effort, "high")

    def test_reasoning_effort_must_be_supported_by_hermes(self) -> None:
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            HermesBrainAdapter(
                endpoint="http://127.0.0.1:8642/v1/chat/completions",
                api_key="secret-key",
                reasoning_effort="fastest",
            )

    def test_from_environment_requires_the_existing_api_server_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "API_SERVER_KEY"):
            HermesBrainAdapter.from_environment({})


if __name__ == "__main__":
    unittest.main()
