"""Bounded HTTP transport. Provider errors never trigger a different provider."""
import json
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit


class ProviderHTTPError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"provider returned HTTP {status}")


class HttpClient:
    def __init__(self, config, *, environment, auth="bearer", opener=None, api_key=None):
        self.base_url = config["base_url"].rstrip("/")
        self.timeout = config["timeout_seconds"]
        self.opener = opener or urlopen
        self.headers = {}
        key_name = config.get("api_key_env")
        if api_key is not None:
            if api_key: self.headers = {"xi-api-key": api_key} if auth == "elevenlabs" else {"Authorization": f"Bearer {api_key}"}
        elif key_name:
            key = environment.get(key_name, "").strip()
            if not key: raise ValueError(f"required environment variable is not set: {key_name}")
            self.headers = {"xi-api-key": key} if auth == "elevenlabs" else {"Authorization": f"Bearer {key}"}

    def request(self, path, *, payload=None, body=None, content_type="application/json", accept="application/json",
                limit=1024*1024, timeout=None):
        if payload is not None: body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        base = urlsplit(self.base_url)
        url = f"{base.scheme}://{base.netloc}{path}" if path.startswith("/") else self.base_url + "/" + path
        request = Request(url, data=body,
                          headers=self.headers | {"Content-Type": content_type, "Accept": accept},
                          method="POST" if body is not None else "GET")
        try:
            with self.opener(request, timeout=timeout if timeout is not None else self.timeout) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read(limit + 1)
                mime = str(getattr(response, "headers", {}).get("Content-Type", "")).split(";", 1)[0].lower()
        except HTTPError as exc: raise ProviderHTTPError(exc.code) from exc
        except (URLError, OSError, TimeoutError, HTTPException) as exc: raise RuntimeError(f"provider request failed: {type(exc).__name__}") from exc
        if not 200 <= status < 300: raise ProviderHTTPError(status)
        if len(raw) > limit: raise RuntimeError("provider response exceeds size limit")
        return raw, mime

    def json(self, path, **kwargs):
        raw, _ = self.request(path, **kwargs)
        try: return json.loads(raw)
        except (ValueError, UnicodeError) as exc: raise RuntimeError("provider response is not valid JSON") from exc
