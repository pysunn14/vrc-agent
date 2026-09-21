from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile

from filelock import FileLock
from platformdirs import user_config_path, user_state_path


def config_directory() -> Path:
    return user_config_path("vrc-agent", appauthor=False)


def state_directory() -> Path:
    return user_state_path("vrc-agent", appauthor=False)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class Preferences:
    language: str = "en"

    def __post_init__(self) -> None:
        if self.language not in ("en", "ko"):
            raise ValueError("language must be en or ko")


class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else config_directory() / "settings.json"

    def read(self) -> Preferences:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Preferences()
        if not isinstance(raw, dict) or set(raw) - {"language"}:
            raise ValueError("invalid settings object")
        return Preferences(**raw)

    def update(self, **changes: str) -> Preferences:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + ".lock", timeout=5):
            updated = Preferences(**(asdict(self.read()) | changes))
            atomic_json(self.path, asdict(updated))
            return updated
