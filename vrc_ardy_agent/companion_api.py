from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from .interaction_runtime import InteractionRuntime
from .status_serialization import to_jsonable


MAX_REQUEST_BYTES = 64 * 1024


class CompanionApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"companion API returned {status}: {detail}")
        self.status = status
        self.detail = detail


class CompanionControlService:
    def __init__(
        self,
        runtime: InteractionRuntime,
        *,
        enable_test_controls: bool = False,
        status_sources: Mapping[str, object] | None = None,
        screenshot_provider: Callable[[], bytes] | None = None,
    ) -> None:
        self.runtime = runtime
        self.enable_test_controls = enable_test_controls
        self.status_sources = dict(status_sources or {})
        self.screenshot_provider = screenshot_provider
        if not all(
            isinstance(name, str) and name.strip() for name in self.status_sources
        ):
            raise ValueError("status source names must be non-empty strings")

    def health(self) -> dict[str, object]:
        snapshot = self.runtime.snapshot()
        return {
            "status": "ok",
            "heartbeat_monotonic": snapshot.heartbeat_monotonic,
        }

    def status(self) -> dict[str, object]:
        status = to_jsonable(self.runtime.snapshot())
        if not isinstance(status, dict):
            raise TypeError("companion snapshot is not serializable")
        if self.status_sources:
            status["components"] = {
                name: self._component_snapshot(source)
                for name, source in sorted(self.status_sources.items())
            }
        return status

    @staticmethod
    def _component_snapshot(source: object) -> object:
        try:
            snapshot = getattr(source, "snapshot", None)
            if callable(snapshot):
                value = snapshot()
            elif callable(source):
                value = source()
            else:
                raise TypeError("status source must be callable or expose snapshot()")
            converted = to_jsonable(value)
            if not isinstance(converted, (dict, list)):
                raise TypeError("status source returned an unsupported value")
            return converted
        except Exception as exc:
            return {"status_error": f"{type(exc).__name__}: {exc}"}

    def actions(self) -> dict[str, object]:
        return {
            "actions": {
                "say": {"required": ["text"]},
                "motion": {
                    "resource": "FULL_BODY_POSE",
                    "names": list(self.runtime.behaviors),
                    "behaviors": [spec.describe() for spec in self.runtime.behaviors.values()],
                },
                "ardy_motion": {
                    "required": ["prompt", "duration_seconds"],
                    "duration_seconds": {"minimum": 1, "maximum": 10},
                },
            },
            "fast_stop": "멈춰",
        }

    def stop(self, payload: object) -> dict[str, object]:
        body = self._require_object(payload)
        extra = set(body) - {"reason"}
        if extra:
            raise ValueError(f"unknown stop fields: {sorted(extra)}")
        reason = body.get("reason", "agentctl")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        return asdict(self.runtime.halt_all(reason=reason.strip()))

    def test_utterance(self, payload: object) -> dict[str, object]:
        self._require_test_controls()
        body = self._require_exact_object(payload, {"text"})
        if not isinstance(body["text"], str) or not body["text"].strip():
            raise ValueError("text must be a non-empty string")
        return asdict(
            self.runtime.handle_utterance(transcript=body["text"], screenshot=None)
        )

    def test_say(self, payload: object) -> dict[str, object]:
        self._require_test_controls()
        body = self._require_exact_object(payload, {"text"})
        return asdict(
            self.runtime.apply_action_payload(
                {"actions": [{"type": "say", "text": body["text"]}]},
                source="agentctl.test_say",
            )
        )

    def test_motion(self, payload: object) -> dict[str, object]:
        self._require_test_controls()
        body = self._require_exact_object(
            payload,
            {"prompt", "duration_seconds"},
        )
        return asdict(
            self.runtime.apply_action_payload(
                {
                    "actions": [
                        {
                            "type": "ardy_motion",
                            "prompt": body["prompt"],
                            "duration_seconds": body["duration_seconds"],
                        }
                    ]
                },
                source="agentctl.test_motion",
            )
        )

    def test_bundle(self, payload: object) -> dict[str, object]:
        self._require_test_controls()
        return asdict(
            self.runtime.apply_action_payload(
                payload,
                source="agentctl.test_bundle",
            )
        )

    def test_screenshot(self) -> bytes:
        self._require_test_controls()
        if self.screenshot_provider is None:
            raise RuntimeError("screenshot provider is unavailable")
        jpeg = self.screenshot_provider()
        if not isinstance(jpeg, bytes) or not jpeg:
            raise RuntimeError("screenshot provider returned no bytes")
        return jpeg

    def _require_test_controls(self) -> None:
        if not self.enable_test_controls:
            raise PermissionError("test controls are disabled")

    @staticmethod
    def _require_object(payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) for key in payload
        ):
            raise ValueError("request body must be a JSON object")
        return payload

    @classmethod
    def _require_exact_object(
        cls,
        payload: object,
        expected: set[str],
    ) -> dict[str, Any]:
        body = cls._require_object(payload)
        if set(body) != expected:
            raise ValueError(f"request fields must be exactly {sorted(expected)}")
        return body


class CompanionHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: CompanionControlService,
    ) -> None:
        self.control_service = service
        super().__init__(server_address, CompanionRequestHandler)


class CompanionRequestHandler(BaseHTTPRequestHandler):
    server_version = "VrcAgentCompanion/0.1"

    @property
    def companion_server(self) -> CompanionHttpServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        service = self.companion_server.control_service
        if path == "/health":
            self._send_json(HTTPStatus.OK, service.health())
        elif path == "/status":
            self._send_json(HTTPStatus.OK, service.status())
        elif path == "/actions":
            self._send_json(HTTPStatus.OK, service.actions())
        elif path == "/test/screenshot":
            try:
                jpeg = service.test_screenshot()
            except PermissionError as exc:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                return
            self._send_bytes(HTTPStatus.OK, jpeg, content_type="image/jpeg")
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        try:
            payload = self._read_json()
            service = self.companion_server.control_service
            handlers = {
                "/stop": service.stop,
                "/test/say": service.test_say,
                "/test/utterance": service.test_utterance,
                "/test/ardy-motion": service.test_motion,
                "/test/bundle": service.test_bundle,
            }
            if hasattr(service, "calibrate"): handlers["/calibration"] = service.calibrate
            handler = handlers.get(path)
            if handler is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            result = handler(payload)
        except PermissionError as exc:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            return
        except (ValueError, TypeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def _read_json(self) -> object:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body is not valid JSON") from exc

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def create_companion_server(
    address: tuple[str, int],
    service: CompanionControlService,
) -> CompanionHttpServer:
    return CompanionHttpServer(address, service)


class CompanionApiClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 5.0) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def actions(self) -> dict[str, Any]:
        return self._request("GET", "/actions")

    def stop(self, *, reason: str = "agentctl") -> dict[str, Any]:
        return self._request("POST", "/stop", {"reason": reason})

    def test_utterance(self, text: str) -> dict[str, Any]:
        return self._request("POST", "/test/utterance", {"text": text})

    def test_say(self, text: str) -> dict[str, Any]:
        return self._request("POST", "/test/say", {"text": text})

    def test_motion(self, prompt: str, duration_seconds: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/test/ardy-motion",
            {"prompt": prompt, "duration_seconds": duration_seconds},
        )

    def test_bundle(self, payload: object) -> dict[str, Any]:
        return self._request("POST", "/test/bundle", payload)

    def test_screenshot(self) -> bytes:
        return self._request_bytes("GET", "/test/screenshot")

    def _request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        raw = self._send_request(method, path, data=data, headers=headers)
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompanionApiError(0, "response was not valid JSON") from exc
        if not isinstance(result, dict):
            raise CompanionApiError(0, "response must be a JSON object")
        return result

    def _request_bytes(self, method: str, path: str) -> bytes:
        return self._send_request(
            method,
            path,
            data=None,
            headers={"Accept": "image/jpeg"},
        )

    def _send_request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None,
        headers: Mapping[str, str],
    ) -> bytes:
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=dict(headers),
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            detail = self._error_detail(raw)
            raise CompanionApiError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise CompanionApiError(0, str(exc.reason)) from exc

    @staticmethod
    def _error_detail(raw: bytes) -> str:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw.decode("utf-8", errors="replace")
        if isinstance(payload, dict):
            return str(payload.get("error", payload))
        return str(payload)
