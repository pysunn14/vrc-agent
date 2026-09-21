from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from .companion_api import CompanionApiClient


DEFAULT_COMPANION_URL = "http://127.0.0.1:8765"


def add_companion_arguments(groups: argparse._SubParsersAction) -> None:
    companion = groups.add_parser(
        "companion",
        help="Observe and control the running companion runtime",
    )
    commands = companion.add_subparsers(
        dest="companion_command",
        required=True,
    )
    for command in ("health", "status", "actions"):
        child = commands.add_parser(command)
        _add_url(child)

    stop = commands.add_parser("stop")
    _add_url(stop)
    stop.add_argument("--reason", default="agentctl")

    say = commands.add_parser("test-say")
    _add_url(say)
    say.add_argument("--text", required=True)

    utterance = commands.add_parser("test-utterance")
    _add_url(utterance)
    utterance.add_argument("--text", required=True)
    utterance.add_argument("--timeout-seconds", type=float, default=35.0)

    motion = commands.add_parser("test-motion")
    _add_url(motion)
    motion.add_argument("--prompt", required=True)
    motion.add_argument("--duration", type=float, required=True)

    screenshot = commands.add_parser(
        "screenshot",
        help="Capture the current Windows VRChat frame",
    )
    _add_url(screenshot)
    screenshot.add_argument("--output", type=Path, required=True)

    bundle = commands.add_parser("test-bundle")
    _add_url(bundle)
    bundle.add_argument("--payload", required=True)

    schedule = commands.add_parser(
        "schedule",
        help="Run a JSON timeline of control operations concurrently",
    )
    _add_url(schedule)
    schedule.add_argument("--payload", required=True)


def run_companion(args: argparse.Namespace) -> dict:
    if args.companion_command == "schedule":
        payload = _parse_json(args.payload, name="--payload")
        return run_companion_schedule(
            payload,
            client_factory=lambda: CompanionApiClient(args.url),
        )
    if args.companion_command == "test-utterance":
        return CompanionApiClient(
            args.url, timeout_seconds=args.timeout_seconds
        ).test_utterance(args.text)
    client = CompanionApiClient(args.url)
    if args.companion_command == "health":
        return client.health()
    if args.companion_command == "status":
        return client.status()
    if args.companion_command == "actions":
        return client.actions()
    if args.companion_command == "stop":
        return client.stop(reason=args.reason)
    if args.companion_command == "test-say":
        return client.test_say(args.text)
    if args.companion_command == "test-motion":
        return client.test_motion(args.prompt, args.duration)
    if args.companion_command == "screenshot":
        jpeg = client.test_screenshot()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(jpeg)
        return {
            "event": "screenshot_saved",
            "path": str(args.output.resolve()),
            "bytes": len(jpeg),
        }
    payload = _parse_json(args.payload, name="--payload")
    return client.test_bundle(payload)


def run_companion_schedule(
    payload: object,
    *,
    client_factory: Callable[[], Any],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    timeline = _validate_schedule(payload)
    started = clock()
    results: list[dict[str, object] | None] = [None] * len(timeline)

    def execute(index: int, item: dict[str, object]) -> None:
        remaining = started + float(item["at_seconds"]) - clock()
        if remaining > 0:
            sleep(remaining)
        operation_started = clock()
        row: dict[str, object] = {
            "index": index,
            "command": item["command"],
            "at_seconds": item["at_seconds"],
            "started_monotonic": operation_started,
        }
        try:
            row["result"] = _dispatch_scheduled(client_factory(), item)
            row["status"] = "completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["finished_monotonic"] = clock()
        results[index] = row

    threads = [
        threading.Thread(
            target=execute,
            args=(index, item),
            name=f"agentctl-companion-schedule-{index}",
        )
        for index, item in enumerate(timeline)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(result is None for result in results):
        raise RuntimeError("a scheduled operation terminated without a result")
    return {
        "scheduled": len(timeline),
        "results": results,
    }


def _validate_schedule(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("schedule payload must be a non-empty JSON array")
    if len(payload) > 64:
        raise ValueError("schedule payload may contain at most 64 operations")
    normalized: list[dict[str, object]] = []
    schemas = {
        **{name: ({"at_seconds", "command"}, set()) for name in ("health", "status", "actions")},
        "stop": ({"at_seconds", "command"}, {"reason"}),
        "test-say": ({"at_seconds", "command", "text"}, set()),
        "test-motion": (
            {"at_seconds", "command", "prompt", "duration_seconds"},
            set(),
        ),
        "test-bundle": ({"at_seconds", "command", "bundle"}, set()),
    }
    for index, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict) or not all(
            isinstance(key, str) for key in raw_item
        ):
            raise ValueError(f"schedule item {index} must be a JSON object")
        command = raw_item.get("command")
        if not isinstance(command, str) or command not in schemas:
            raise ValueError(f"schedule item {index} has an unknown command")
        required, optional = schemas[command]
        fields = set(raw_item)
        if not required <= fields or fields - required - optional:
            allowed = sorted(required | optional)
            raise ValueError(f"schedule item {index} fields must match {allowed}")
        at_seconds = raw_item.get("at_seconds")
        if (
            isinstance(at_seconds, bool)
            or not isinstance(at_seconds, (int, float))
            or not math.isfinite(at_seconds)
            or not 0 <= float(at_seconds) <= 30
        ):
            raise ValueError(
                f"schedule item {index} at_seconds must be between 0 and 30"
            )
        item = dict(raw_item)
        item["at_seconds"] = float(at_seconds)
        if command == "stop":
            reason = item.get("reason", "agentctl.schedule")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"schedule item {index} reason must not be empty")
            item["reason"] = reason.strip()
        elif command == "test-say":
            _require_schedule_text(item.get("text"), index=index, field="text")
        elif command == "test-motion":
            _require_schedule_text(item.get("prompt"), index=index, field="prompt")
            duration = item.get("duration_seconds")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or not 1 <= float(duration) <= 10
            ):
                raise ValueError(
                    f"schedule item {index} duration_seconds must be between 1 and 10"
                )
            item["duration_seconds"] = float(duration)
        normalized.append(item)
    return normalized


def _require_schedule_text(value: object, *, index: int, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"schedule item {index} {field} must not be empty")


def _dispatch_scheduled(client: Any, item: dict[str, object]) -> dict[str, object]:
    command = item["command"]
    if command in ("health", "status", "actions"):
        return getattr(client, command)()
    if command == "stop":
        return client.stop(reason=item["reason"])
    if command == "test-say":
        return client.test_say(item["text"])
    if command == "test-motion":
        return client.test_motion(item["prompt"], item["duration_seconds"])
    if command == "test-bundle":
        return client.test_bundle(item["bundle"])
    raise AssertionError(f"unreachable scheduled command: {command}")


def _parse_json(raw: str, *, name: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be valid JSON: {exc}") from exc


def _add_url(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=DEFAULT_COMPANION_URL)
