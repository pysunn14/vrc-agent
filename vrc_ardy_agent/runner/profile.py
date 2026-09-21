"""Role-based local ownership with optimistic concurrent profile editing."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import ipaddress
import json
from pathlib import Path
import re

from filelock import FileLock
from .profile_fields import object_fields, identifier, string, validate_agent, validate_bridge
from .profile_upgrade import upgrade_profile
from .settings import atomic_json, config_directory


@dataclass(frozen=True)
class Profile:
    data: dict
    path: Path | None = None

    @classmethod
    def parse(cls, raw, path=None):
        data = upgrade_profile(raw)
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 3:
            raise ValueError("unsupported profile version")
        role = data.get("role")
        if role not in ("agent", "bridge", "combined"): raise ValueError("invalid device role")
        fields = ("version", "language", "role", "hosts", "runner_host", "game_host", "connection")
        if role != "bridge": fields += ("providers", "bindings", "runtime", "avatar", "behaviors", "autonomy")
        if role != "agent": fields += ("bridge",)
        object_fields(data, fields, fields, name="profile")
        if data["language"] not in ("en", "ko"): raise ValueError("language must be en or ko")
        hosts = data["hosts"]
        if not isinstance(hosts, dict) or not 1 <= len(hosts) <= 32: raise ValueError("configure 1..32 hosts")
        for key, host in hosts.items():
            identifier(key)
            object_fields(host, ("os", "address"), ("os", "address"), name=f"host {key}")
            if host["os"] not in ("macos", "linux", "windows"): raise ValueError("invalid host OS")
            string(host["address"], "host address")
            if not re.fullmatch(r"[A-Za-z0-9_.:-]+", host["address"]): raise ValueError("host address must be a hostname or IP")
        for key in ("runner_host", "game_host"):
            if not isinstance(data[key], str) or data[key] not in hosts: raise ValueError(f"unknown {key}")
        if hosts[data["game_host"]]["os"] != "windows": raise ValueError("game host must run Windows")
        same_host = data["runner_host"] == data["game_host"]
        if (role == "combined") != same_host: raise ValueError("combined role requires one Windows host; separate roles require two hosts")
        if not same_host:
            for host in (data["runner_host"], data["game_host"]):
                host_address = hosts[host]["address"]
                if host_address.lower().rstrip(".") == "localhost": raise ValueError("separate computers require reachable addresses, not localhost")
                try: ip = ipaddress.ip_address(host_address)
                except ValueError: continue  # Hostnames are resolved by the connection, not by profile validation.
                if ip.is_loopback or ip.is_unspecified:
                    raise ValueError("separate computers require reachable addresses, not loopback or wildcard addresses")
        ports = data["connection"]
        defaults = {"stream_port": 8766, "agent_port": 8765, "bridge_port": 8767}
        object_fields(ports, defaults, name="connection")
        for key, default in defaults.items():
            value = ports.setdefault(key, default)
            if type(value) is not int or not 1 <= value <= 65535: raise ValueError(f"{key}: invalid port")
        if ports["stream_port"] == ports["agent_port"]: raise ValueError("agent ports must differ")
        if same_host and len(set(ports.values())) != 3: raise ValueError("same-host ports must differ")
        if role != "bridge": validate_agent(data)
        if role != "agent": validate_bridge(data["bridge"], standalone=role == "bridge")
        return cls(data, Path(path).resolve() if path is not None else None)

    @classmethod
    def load(cls, path):
        return cls.parse(json.loads(Path(path).read_text(encoding="utf-8")), path)

    @property
    def has_agent(self): return self.data["role"] != "bridge"

    @property
    def has_bridge(self): return self.data["role"] != "agent"

    @property
    def local_host(self):
        return self.data["game_host"] if self.data["role"] == "bridge" else self.data["runner_host"]

    def host_address(self, host):
        return "127.0.0.1" if self.data["role"] == "combined" else self.data["hosts"][host]["address"]

    def require_agent(self):
        if not self.has_agent: raise ValueError("this operation requires an agent role; run it on the agent computer")

    def resolve(self, capability):
        self.require_agent()
        binding = self.data["bindings"][capability]
        return self.data["providers"][binding["instance"]], binding

    @property
    def bridge_address(self): return self.host_address(self.data["runner_host"])


class RevisionConflict(ValueError): pass


class ProfileStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else config_directory() / "agent.json"

    @property
    def draft_path(self): return self.path.with_name(self.path.name + ".draft")

    def read(self):
        try: raw = self.path.read_bytes()
        except FileNotFoundError: return {"profile": None, "revision": None}
        return {"profile": Profile.parse(json.loads(raw), self.path).data,
                "revision": hashlib.sha256(raw).hexdigest()}

    def save_draft(self, data, *, expected_revision, completed=()):
        if not isinstance(completed, (list, tuple)) or any(key not in
            ("language", "role", "hosts", "llm", "stt", "tts", "runtime", "avatar", "bridge") for key in completed):
            raise ValueError("invalid completed setup sections")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + ".lock", timeout=5):
            if self.read()["revision"] != expected_revision: raise RevisionConflict("profile changed; reopen setup")
            atomic_json(self.draft_path, {"profile": data, "revision": expected_revision, "completed": list(completed)})

    def commit(self, data, *, expected_revision):
        profile = Profile.parse(data, self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + ".lock", timeout=5):
            if self.read()["revision"] != expected_revision: raise RevisionConflict("profile changed; reopen setup")
            atomic_json(self.path, profile.data)
            self.draft_path.unlink(missing_ok=True)
            return self.read()
