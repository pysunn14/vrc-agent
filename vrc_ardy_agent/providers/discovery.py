"""Read-only inventory for unsaved connections, shared by setup and the CLI.

Inventory access proves neither inference nor model capabilities. Unknown
capabilities stay unknown; a model name or its owner is not a routing policy.
Each call fetches one bounded page so the UI can offer more without an unbounded
pagination loop. No credentials or remote error bodies leave this boundary.
"""
import os
from http.client import HTTPException
from urllib.error import URLError
from urllib.parse import urlencode

from .catalog import definition, validate_connection
from .credentials import resolve_key, CredentialError
from .http import HttpClient, ProviderHTTPError
from .presets import OPENAI_VOICES


def discover_items(provider, config, *, kind, capability=None, cursor=None, environment=None, opener=None):
    config = validate_connection(provider, config)
    spec = definition(provider, config)
    if kind not in ("models", "voices"): raise ValueError("kind must be models or voices")
    if capability is not None and capability not in spec["operations"]: raise ValueError("unsupported capability")
    if cursor is not None and (not isinstance(cursor, str) or not cursor or len(cursor) > 4096):
        raise ValueError("invalid inventory cursor")
    environment = os.environ if environment is None else environment
    def result(state, detail, **extra):
        return {"state": state, "detail": detail, "items": [], "next_cursor": None, "source": "server", **extra}
    if spec["protocol"] == "whisper-local":
        return result("unsupported", "Local Whisper requires an installed model directory; there is no inventory API.")
    if kind == "voices" and spec['inventory']['voices'] == 'openai':
        return result('ready', 'Documented built-in voices; verify with the selected model.',
                      source='preset', items=[{'id':v,'label':v} for v in OPENAI_VOICES])
    if kind == 'voices' and spec['protocol'] != 'elevenlabs' and not config.get('voices_path'):
        return result('unsupported', 'This connection has no declared voice inventory path; enter a voice ID or configure its path.')
    try: api_key = resolve_key(provider, config, environment=environment)
    except CredentialError as exc:
        return result(exc.state, str(exc), credential_env=config.get("api_key_env", ""))
    path = (config.get("voices_path") or kind) if kind == "voices" else kind
    if spec["protocol"] == "elevenlabs" and kind == "voices":
        # ElevenLabs serves inference/models on v1 and paginated voices on v2.
        # Preserve a reverse proxy prefix instead of replacing the URL origin.
        if not config["base_url"].endswith("/v1"):
            return result("unsupported", "ElevenLabs voice discovery requires an API base URL ending in /v1.")
        config = config | {"base_url": config["base_url"][:-3] + "/v2"}
        path += "?" + urlencode({"page_size": 100, **({"next_page_token": cursor} if cursor else {})})
    elif cursor:
        raise ValueError("this inventory does not support pagination")
    try:
        body = HttpClient(config, environment=environment, auth=spec["auth"], opener=opener, api_key=api_key).json(
            path, timeout=min(config["timeout_seconds"], 5))
    except RuntimeError as exc:
        cause = exc.__cause__
        if isinstance(exc, ProviderHTTPError):
            state = "auth-error" if exc.status in (401, 403) else "unsupported" if exc.status in (404, 405) else "http-error"
            return result(state, f"Inventory request returned HTTP {exc.status}.")
        if isinstance(cause, (URLError, OSError, TimeoutError, HTTPException)):
            return result("network-error", "Inventory request failed. Check the API address and whether the service is running.")
        return result("invalid-response", str(exc))
    try:
        items, next_cursor = normalize_inventory(body, adapter=spec["protocol"], kind=kind, capability=capability)
    except ValueError as exc:
        return result("invalid-response", str(exc))
    return result("ready" if items else "empty", "Inventory only; inference has not been tested.",
                  items=items, next_cursor=next_cursor)


def normalize_inventory(body, *, adapter, kind, capability):
    if adapter == "elevenlabs" and kind == "models":
        rows, key = body, "model_id"
    elif isinstance(body, dict):
        rows, key = body.get("data", body.get(kind)), "id"
    else:
        rows, key = body, "id"
    if not isinstance(rows, list) or len(rows) > 10000: raise ValueError("Invalid inventory list.")
    items = {}
    for row in rows:
        if isinstance(row, str) and kind == "voices": row = {"id": row}
        if not isinstance(row, dict): raise ValueError("Invalid inventory entry.")
        identifier = row.get("voice_id", row.get("id")) if kind == "voices" else row.get(key)
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 4096:
            raise ValueError("Inventory entry is missing a valid ID.")
        if adapter == "elevenlabs" and capability == "tts" and row.get("can_do_text_to_speech") is False: continue
        label = row.get("name", identifier)
        if not isinstance(label, str) or not label.strip(): label = identifier
        items.setdefault(identifier, {"id": identifier, "label": label[:4096]})
    next_cursor = None
    if isinstance(body, dict) and body.get("has_more"):
        next_cursor = body.get("next_page_token")
        if not isinstance(next_cursor, str) or not next_cursor or len(next_cursor) > 4096:
            raise ValueError("Paginated inventory is missing a valid next page token.")
        if adapter != "elevenlabs" or kind != "voices":
            raise ValueError("This provider's inventory pagination format is not supported.")
    return list(items.values()), next_cursor
