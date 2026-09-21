from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import RunnerConfig
from .core import Runner
from .settings import SettingsStore, config_directory


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--language", choices=("en", "ko"))
    commands = parser.add_subparsers(dest="runner_command")
    commands.add_parser("init", help="Create a configuration without overwriting an existing file")
    for name in ("status", "doctor"):
        child = commands.add_parser(name)
        child.add_argument("--json", action="store_true")
    settings = commands.add_parser("settings")
    settings.add_argument("setting_language", nargs="?", choices=("en", "ko"))
    settings.add_argument("--json", action="store_true")
    for name in ("start", "stop", "logs"):
        child = commands.add_parser(name)
        child.add_argument("service")
        child.add_argument("--json", action="store_true")
    return commands


def initialize(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("default.toml").read_text(encoding="utf-8")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(source)
    return path


def make_runner(args) -> Runner:
    path = args.config or config_directory() / "runner.toml"
    if args.config is None and not path.exists():
        try:
            initialize(path)
        except FileExistsError:
            pass  # Another client created the same default configuration.
    return Runner(RunnerConfig.load(path), args.state_dir)


def run_command(args) -> int:
    store = SettingsStore(args.settings_file)
    preferences = store.read()
    language = getattr(args, "setting_language", None) or args.language
    if language:
        preferences = store.update(language=language)
    command = args.runner_command
    if command == "init":
        print(initialize(args.config or config_directory() / "runner.toml"))
        return 0
    if command == "settings":
        print(json.dumps(asdict(preferences)))
        return 0
    runner = make_runner(args)
    if command is None:
        print("Use npm start for the interactive VRC AGENT console.")
        return 0
    if command in ("status", "doctor"):
        result = getattr(runner, command)()
    elif command == "logs":
        result = {"service": args.service, "output": runner.logs(args.service)}
    else:
        result = getattr(runner, command)(args.service)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if command == "doctor" and any(row["result"] == "error" for row in result):
        return 1
    if command in ("start", "stop") and result["state"] == "failed":
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VRC AGENT command-line runner")
    configure_parser(parser)
    args = parser.parse_args(argv)
    try:
        return run_command(args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        else:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
