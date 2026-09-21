"""Development-only process controls for service-level exploration."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import threading
import time

from .runner.cli import configure_parser, make_runner, run_command
from .runner.settings import SettingsStore, atomic_json


def configure_agentctl(parser):
    commands = configure_parser(parser)
    commands.add_parser("actions")
    schedule = commands.add_parser("schedule")
    schedule.add_argument("--payload", required=True)
    schedule.add_argument("--output", type=Path, required=True)


def schedule(runner, payload, output):
    if not isinstance(payload, list) or not 1 <= len(payload) <= 32:
        raise ValueError("schedule must contain between 1 and 32 operations")
    for item in payload:
        if not isinstance(item, dict) or set(item) - {"at_seconds", "command", "service"}:
            raise ValueError("invalid schedule fields")
        delay = item.get("at_seconds")
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or not 0 <= delay <= 30:
            raise ValueError("at_seconds must be between 0 and 30")
        if item.get("command") not in ("status", "doctor", "start", "stop"):
            raise ValueError("unsupported scheduled operation")
        if item["command"] in ("start", "stop"):
            runner.service(item.get("service"))
        elif "service" in item:
            raise ValueError("status and doctor do not accept a service")
    started = time.monotonic()
    results = [None] * len(payload)
    lock = threading.Lock()
    atomic_json(output, {"completed": 0, "total": len(payload), "results": results})

    def execute(index, item):
        time.sleep(max(0, started + item["at_seconds"] - time.monotonic()))
        row = {"command": item["command"], "started_seconds": time.monotonic() - started}
        try:
            method = getattr(runner, item["command"])
            row["result"] = method(item["service"]) if "service" in item else method()
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["finished_seconds"] = time.monotonic() - started
        with lock:
            results[index] = row
            completed = sum(value is not None for value in results)
            atomic_json(output, {"completed": completed, "total": len(payload), "results": results})
            print(json.dumps({"event": "progress", "completed": completed, "total": len(payload)}), flush=True)
    with ThreadPoolExecutor(max_workers=len(payload)) as pool:
        futures = [pool.submit(execute, index, item) for index, item in enumerate(payload)]
        for future in futures:
            future.result()
    return {"completed": len(results), "results": results}


def run_agentctl(args):
    if args.runner_command in ("actions", "schedule") and args.language:
        SettingsStore(args.settings_file).update(language=args.language)
    if args.runner_command == "actions":
        result = {"commands": ["status", "doctor", "start", "stop", "logs", "settings", "schedule"],
                  "languages": ["en", "ko"]}
    elif args.runner_command == "schedule":
        result = schedule(make_runner(args), json.loads(args.payload), args.output)
    else:
        return run_command(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.runner_command == "schedule" and any("error" in r for r in result["results"]) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="VRC AGENT runner exploration")
    configure_agentctl(parser)
    args = parser.parse_args(argv)
    try:
        return run_agentctl(args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
