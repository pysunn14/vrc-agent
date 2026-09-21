from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
import shutil
import tomllib
from urllib.parse import urlsplit

from .settings import config_directory


@dataclass(frozen=True)
class Service:
    id: str
    command: tuple[str, ...] = ()
    cwd: Path = field(default_factory=Path.cwd)
    health_url: str = ""
    health_status: str = ""
    required_files: tuple[Path, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    label: str = ""
    label_ko: str = ""
    timeout: float = 1.0

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", self.id):
            raise ValueError("service id must contain lowercase letters, digits, underscores or hyphens")
        if not isinstance(self.command, (list, tuple)) or any(not isinstance(x, str) or not x for x in self.command):
            raise ValueError("command must be an array of non-empty strings")
        if not isinstance(self.env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.env.items()):
            raise ValueError("env must map strings to strings")
        for name in ("health_url", "health_status", "label", "label_ko"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or not .1 <= self.timeout <= 10:
            raise ValueError("timeout must be between 0.1 and 10 seconds")
        if self.health_url:
            url = urlsplit(self.health_url)
            if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
                raise ValueError("health_url must be an HTTP(S) URL without credentials")
            _ = url.port
        elif self.health_status:
            raise ValueError("health_status requires health_url")

    def payload(self) -> dict:
        return asdict(self) | {"cwd": str(self.cwd), "required_files": [str(p) for p in self.required_files]}

    def executable(self) -> str | None:
        if not self.command or not self.cwd.is_dir():
            return None
        command = self.command[0]
        if "/" in command or "\\" in command:
            import os
            path = Path(command)
            path = path if path.is_absolute() else self.cwd / path
            return str(path) if path.is_file() and os.access(path, os.X_OK) else None
        return shutil.which(command, path=self.env.get("PATH"))


@dataclass(frozen=True)
class RunnerConfig:
    services: tuple[Service, ...]
    path: Path | None = None

    def __post_init__(self):
        if len(self.services) > 32 or len({s.id for s in self.services}) != len(self.services):
            raise ValueError("configure at most 32 services with unique ids")

    @classmethod
    def load(cls, path: Path | None = None) -> "RunnerConfig":
        path = path if path is not None else config_directory() / "runner.toml"
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"runner configuration not found: {path}; run vrc-agent init") from None
        if set(raw) - {"services"} or not isinstance(raw.get("services"), dict):
            raise ValueError("runner configuration must contain a services table")
        services = []
        for service_id, values in raw["services"].items():
            allowed = set(Service.__dataclass_fields__) - {"id"}
            if not isinstance(values, dict) or set(values) - allowed:
                raise ValueError(f"unknown configuration fields for {service_id}")
            values = dict(values)
            if "cwd" in values and not isinstance(values["cwd"], str):
                raise ValueError("cwd must be a path string")
            cwd = Path(values.get("cwd", ".")).expanduser()
            values["cwd"] = (path.parent / cwd).resolve()
            files = values.get("required_files", [])
            if not isinstance(files, list) or any(not isinstance(p, str) for p in files):
                raise ValueError("required_files must be an array of paths")
            values["required_files"] = tuple((values["cwd"] / Path(p).expanduser()).resolve() for p in files)
            if "command" in values and isinstance(values["command"], list):
                values["command"] = tuple(values["command"])
            services.append(Service(service_id, **values))
        return cls(tuple(services), path.resolve())

