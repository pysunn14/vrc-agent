from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock
import psutil

from .settings import atomic_json


def read_record(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"invalid process record: {path}")
    return value


def update_record(path: Path, ticket: str, **changes) -> dict:
    with FileLock(str(path) + ".lock", timeout=5):
        value = read_record(path)
        if value.get("ticket") != ticket:
            raise RuntimeError("process record belongs to another run")
        value.update(changes)
        atomic_json(path, value)
        return value


def owned_process(pid: int | None, created: float | None) -> psutil.Process | None:
    if not pid or created is None:
        return None
    try:
        process = psutil.Process(pid)
        if process.create_time() != created or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except psutil.NoSuchProcess:
        return None

