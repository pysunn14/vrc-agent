import json
from urllib.error import HTTPError, URLError

import pytest

from tests.test_provider_adapters import Recorder, Response
from vrc_ardy_agent.providers.discovery import discover_items
from vrc_ardy_agent.runner.control import dispatch


def discover(body, *, provider="openai-compatible", kind="models", capability="llm", **kwargs):
    opener = Recorder(Response(body))
    result = discover_items(provider, {"base_url": "http://example.test/prefix/v1"},
                            kind=kind, capability=capability, environment={}, opener=opener, **kwargs)
    return result, opener


def test_unsaved_connection_discovers_models_without_claiming_capabilities():
    result, opener = discover({"data": [{"id": "only-model", "owned_by": "organization"}]})
    assert result["state"] == "ready"
    assert result["items"] == [{"id": "only-model", "label": "only-model"}]
    request, options = opener.calls[0]
    assert request.full_url == "http://example.test/prefix/v1/models"
    assert request.get_method() == "GET"
    assert options["timeout"] <= 5


@pytest.mark.parametrize("body,state", [({"data": []}, "empty"), ({"error": "bad"}, "invalid-response"),
                                       ({"data": [{"name": "no-id"}]}, "invalid-response"),
                                       ({"data": [{"id": "a"}, {"id": "a"}]}, "ready")])
def test_empty_malformed_and_duplicate_inventory(body, state):
    result, _ = discover(body)
    assert result["state"] == state
    if state == "ready": assert len(result["items"]) == 1


@pytest.mark.parametrize("code,state", [(401, "auth-error"), (403, "auth-error"), (404, "unsupported"),
                                       (405, "unsupported"), (429, "http-error"), (500, "http-error")])
def test_http_errors_are_distinguishable(code, state):
    def fail(request, **kwargs): raise HTTPError(request.full_url, code, "private body", {}, None)
    result = discover_items("openai", {}, kind="models", environment={"OPENAI_API_KEY": "secret"}, opener=fail)
    assert result["state"] == state
    assert "secret" not in json.dumps(result)
    assert "private body" not in json.dumps(result)


def test_missing_auth_is_observed_before_network_and_only_presence_is_returned():
    def fail(*args, **kwargs): pytest.fail("missing credentials must not contact the server")
    result = discover_items("hermes", {}, kind="models", environment={}, opener=fail)
    assert result["state"] == "needs-auth"
    assert result["credential_env"] == "API_SERVER_KEY"


def test_timeout_and_invalid_json_are_not_unsupported():
    def timeout(*args, **kwargs): raise URLError(TimeoutError())
    assert discover_items("openai-compatible", {"base_url": "http://example.test"}, kind="models",
                          environment={}, opener=timeout)["state"] == "network-error"
    assert discover(b"not json")[0]["state"] == "invalid-response"


def test_explicit_error_status_in_transport_is_preserved():
    response = Response({})
    response.status = 503
    result = discover_items("openai-compatible", {"base_url": "http://example.test"}, kind="models",
                            environment={}, opener=Recorder(response))
    assert result["state"] == "http-error"


def test_elevenlabs_filters_explicitly_incompatible_models_and_keeps_unknown():
    result = discover_items("elevenlabs", {}, kind="models", capability="tts",
        environment={"ELEVENLABS_API_KEY": "secret"}, opener=Recorder(Response([
            {"model_id": "voice-conversion", "can_do_text_to_speech": False},
            {"model_id": "speech", "name": "Speech", "can_do_text_to_speech": True},
            {"model_id": "unknown"}])))
    assert [row["id"] for row in result["items"]] == ["speech", "unknown"]


def test_voice_page_uses_v2_under_the_same_proxy_prefix_and_preserves_cursor():
    opener = Recorder(Response({"voices": [{"voice_id": "v1", "name": "Voice"}],
                                "has_more": True, "next_page_token": "next/2"}))
    result = discover_items("elevenlabs", {"base_url": "http://example.test/prefix/v1"},
        kind="voices", environment={"ELEVENLABS_API_KEY": "secret"}, opener=opener, cursor="first/1")
    assert result["next_cursor"] == "next/2"
    assert result["items"] == [{"id": "v1", "label": "Voice"}]
    assert opener.calls[0][0].full_url == "http://example.test/prefix/v2/voices?page_size=100&next_page_token=first%2F1"
    assert opener.calls[0][0].get_header("Xi-api-key") == "secret"


def test_discovery_and_draft_progress_are_available_without_an_active_profile(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    result = dispatch("connection.discover", {"provider": "hermes", "config": {}, "kind": "models"}, path=path)
    assert result["state"] == "needs-auth"
    dispatch("draft.save", {"profile": {"version": 3}, "revision": None, "completed": ["hosts"]}, path=path)
    assert json.loads(path.with_name(path.name + ".draft").read_text())["completed"] == ["hosts"]
