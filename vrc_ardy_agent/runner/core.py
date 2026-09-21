from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import uuid4

from filelock import FileLock
import psutil

from .config import RunnerConfig, Service
from .records import owned_process, read_record, update_record
from .settings import atomic_json, state_directory


def probe(service: Service) -> dict:
    if not service.health_url:
        return {"ok": None, "detail": ""}
    try:
        with urlopen(service.health_url, timeout=service.timeout) as response:
            if response.status != 200:
                return {"ok": False, "detail": f"HTTP {response.status}"}
            if service.health_status:
                body = json.loads(response.read(65537))
                if not isinstance(body, dict) or body.get("status") != service.health_status:
                    return {"ok": False, "detail": "health status does not match configured value"}
            return {"ok": True, "detail": "HTTP 200"}
    except (HTTPError, URLError, OSError, ValueError) as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


class Runner:
    def __init__(self, config: RunnerConfig, directory: Path | None = None):
        self.config = config
        namespace = hashlib.sha256(str(config.path).encode()).hexdigest()[:12]
        self.directory = (directory if directory is not None else state_directory() / namespace).resolve()

    def service(self, service_id: str) -> Service:
        for service in self.config.services:
            if service.id == service_id:
                return service
        raise ValueError(f"unknown service: {service_id}")

    def record_path(self, service_id: str) -> Path:
        self.service(service_id)
        return self.directory / f"{service_id}.json"

    def observe(self, service_id: str, *, check_health: bool = True) -> dict:
        service = self.service(service_id)
        record = read_record(self.record_path(service_id))
        supervisor = owned_process(record.get("supervisor_pid"), record.get("supervisor_created"))
        child = owned_process(record.get("pid"), record.get("created"))
        health = probe(service) if check_health else {"ok": None, "detail": ""}
        result = {"id": service_id, "pid": child.pid if child else None,
                  "supervisor_pid": supervisor.pid if supervisor else None,
                  "heartbeat_at": record.get("heartbeat_at"), "exit_code": record.get("exit_code"),
                  "health": health, "error": record.get("error", ""),
                  "owned": bool(child or supervisor), "observed_at": time.time()}
        if child is not None:
            if supervisor is None:
                state, code = "failed", "supervisor_lost"
            elif record.get("phase") == "stopping":
                state, code = "stopping", "process_alive"
            elif health["ok"]:
                state, code = "ready", "endpoint_ok"
            else:
                state, code = "running", "endpoint_failed" if health["ok"] is False else "process_alive"
        elif supervisor and record.get("phase") in ("starting", "running", "stopping"):
            state, code = "starting", "process_starting"
        elif health["ok"]:
            state, code = "external", "external_process"
        elif record.get("phase") == "failed" or (record and record.get("phase") != "stopped"):
            state, code = "failed", "process_failed" if record.get("exit_code") is not None else "supervisor_lost"
        elif service.command:
            state, code = "stopped", "process_stopped"
        elif service.health_url:
            state, code = "unreachable", "endpoint_failed"
        else:
            state, code = "unconfigured", "missing_service"
        return result | {"state": state, "code": code}

    def status(self, progress=None) -> list[dict]:
        services = self.config.services
        if not services:
            return []
        rows = {}
        with ThreadPoolExecutor(max_workers=min(8, len(services))) as pool:
            jobs = {pool.submit(self.observe, s.id): s.id for s in services}
            for done, future in enumerate(as_completed(jobs), 1):
                rows[jobs[future]] = future.result()
                if progress:
                    progress(done, len(jobs))
        return [rows[s.id] for s in services]

    def start(self, service_id: str) -> dict:
        service = self.service(service_id)
        path = self.record_path(service_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".control.lock", timeout=15):
            current = self.observe(service_id)
            if current["owned"] or current["state"] == "external":
                return current
            if not service.command:
                raise ValueError("no start command configured for this service")
            executable = service.executable()
            if executable is None:
                raise ValueError("executable or working directory is unavailable")
            missing = [str(p) for p in service.required_files if not p.is_file()]
            if missing:
                raise ValueError(f"required files are missing: {missing}")
            ticket = uuid4().hex
            spec = service.payload()
            spec["command"] = [executable, *service.command[1:]]
            atomic_json(path, {"ticket": ticket, "phase": "starting", "service": spec,
                               "stop_requested": False, "started_at": time.time(), "exit_code": None})
            try:
                with path.with_suffix(".supervisor.log").open("ab") as output:
                    worker = subprocess.Popen(
                        [sys.executable, "-m", "vrc_ardy_agent.runner.worker", "--record", str(path), "--ticket", ticket],
                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                        start_new_session=os.name != "nt",
                        creationflags=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
                    )
                identity = psutil.Process(worker.pid)
                update_record(path, ticket, supervisor_pid=identity.pid, supervisor_created=identity.create_time())
                # Reap the child while this client lives; detached supervision
                # continues independently when the client exits.
                threading.Thread(target=worker.wait, daemon=True).start()
            except Exception as exc:
                update_record(path, ticket, phase="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            return self.observe(service_id, check_health=False)

    def stop(self, service_id: str) -> dict:
        path = self.record_path(service_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".control.lock", timeout=15):
            current = self.observe(service_id)
            if current["state"] == "external":
                raise ValueError("service was started outside this runner")
            if not current["owned"]:
                return current
            record = read_record(path)
            if current["supervisor_pid"] is None:
                raise RuntimeError("supervisor is unavailable; inspect the recorded process before recovery")
            update_record(path, record["ticket"], stop_requested=True)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                observed = self.observe(service_id, check_health=False)
                if not observed["owned"]:
                    return observed
                time.sleep(.05)
            raise TimeoutError("stop was requested but process termination has not been observed")

    def logs(self, service_id: str, *, limit: int = 65536) -> str:
        path = self.record_path(service_id).with_suffix(".log")
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - limit))
                return stream.read(limit).decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def doctor(self, progress=None, *, statuses: list[dict] | None = None) -> list[dict]:
        rows = []
        statuses = {row["id"]: row for row in (statuses if statuses is not None else self.status(progress=progress))}
        for service in self.config.services:
            def add(check, result, code, detail=""):
                rows.append({"service": service.id, "check": check, "result": result, "code": code, "detail": detail})
            if service.command:
                valid = service.executable() is not None
                add("command", "pass" if valid else "error", "command_ok" if valid else "command_invalid",
                    json.dumps(list(service.command), ensure_ascii=False))
            elif not service.health_url:
                add("command", "warning", "missing_service")
            for path in service.required_files:
                valid = path.is_file()
                add("files", "pass" if valid else "error", "asset_ok" if valid else "asset_missing", str(path))
            state = statuses[service.id]
            if service.health_url:
                ok = state["health"]["ok"]
                add("endpoint", "pass" if ok else "error", "endpoint_ok" if ok else "endpoint_failed", state["health"]["detail"])
            if state["state"] == "failed":
                add("process", "error", state["code"], state["error"] or f"exit_code={state['exit_code']}")
        return rows
