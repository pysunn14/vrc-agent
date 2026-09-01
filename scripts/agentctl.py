#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.follow_protocol import TargetObservation, TargetSource
from vrc_ardy_agent.follow_receiver import UdpObservationReceiver
from vrc_ardy_agent.windows_capture import (
    WindowsGraphicsCaptureSource,
    list_windows,
    wait_for_window,
)
from vrc_ardy_agent.windows_perception import (
    UdpObservationSender,
    UltralyticsPersonTracker,
    WindowsPerceptionRunner,
)
from vrc_ardy_agent.nameplate import (
    AsyncNameplateTracker,
    EasyOcrTextReader,
    NameplateMatcher,
)
from vrc_ardy_agent.target_fusion import TargetFusionSelector


def _parse_hwnd(value: str) -> int:
    try:
        result = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HWND must be a decimal or 0x-prefixed integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("HWND must be positive")
    return result


def _json_default(value):
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"cannot encode {type(value).__name__}")


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":"), default=_json_default), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory control and observation CLI for the VRChat ARDY follower."
    )
    groups = parser.add_subparsers(dest="group", required=True)

    windows = groups.add_parser("windows", help="Inspect or run the Windows perception side")
    windows_commands = windows.add_subparsers(dest="windows_command", required=True)

    windows_list = windows_commands.add_parser("list", help="List top-level windows")
    windows_list.add_argument("--title")
    windows_list.add_argument("--json", action="store_true")

    screenshot = windows_commands.add_parser("screenshot", help="Capture one window frame")
    screenshot.add_argument("--hwnd", type=_parse_hwnd, required=True)
    screenshot.add_argument("--output", type=Path, required=True)
    screenshot.add_argument("--timeout", type=float, default=3.0)

    nameplate = windows_commands.add_parser(
        "nameplate",
        help="Capture one frame and inspect the configured display name",
    )
    nameplate.add_argument("--hwnd", type=_parse_hwnd, required=True)
    nameplate.add_argument("--target-name", required=True)
    nameplate.add_argument("--timeout", type=float, default=3.0)
    nameplate.add_argument("--nameplate-match-threshold", type=float, default=0.72)
    nameplate.add_argument("--nameplate-input-width", type=int, default=960)
    nameplate.add_argument("--nameplate-model-dir")

    track = windows_commands.add_parser("track", help="Track one person and stream target state")
    _add_tracking_arguments(track)

    follow = groups.add_parser("follow", help="Inject or observe follower target packets")
    follow_commands = follow.add_subparsers(dest="follow_command", required=True)

    inject = follow_commands.add_parser("inject", help="Send a scripted target state")
    inject.add_argument("--host", default="127.0.0.1")
    inject.add_argument("--port", type=int, default=9200)
    inject.add_argument("--center-x", type=float)
    inject.add_argument("--proximity", type=float)
    inject.add_argument(
        "--source",
        choices=[source.value for source in TargetSource],
        default=TargetSource.BODY.value,
    )
    inject.add_argument("--confidence", type=float, default=0.9)
    inject.add_argument("--repeat", type=int, default=1)
    inject.add_argument("--interval", type=float, default=0.05)
    inject.add_argument("--delay", type=float, default=0.0)
    inject.add_argument("--session")
    inject.add_argument("--json", action="store_true")

    listen = follow_commands.add_parser("listen", help="Observe received target state and health")
    listen.add_argument("--bind-host", default="0.0.0.0")
    listen.add_argument("--port", type=int, default=9200)
    listen.add_argument("--duration", type=float, default=10.0)
    listen.add_argument("--json", action="store_true")
    return parser


def _add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    window_selector = parser.add_mutually_exclusive_group(required=True)
    window_selector.add_argument("--hwnd", type=_parse_hwnd)
    window_selector.add_argument("--window-title")
    window_selector.add_argument("--window-process-id", type=_parse_hwnd)
    parser.add_argument("--window-wait-timeout", type=float, default=None)
    parser.add_argument("--mac-host", required=True)
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--capture-interval-ms", type=int, default=50)
    parser.add_argument("--reacquire-frames", type=int, default=6)
    parser.add_argument("--target-name")
    parser.add_argument("--nameplate-scan-interval", type=float, default=5.0)
    parser.add_argument("--nameplate-anchor-max-age", type=float, default=120.0)
    parser.add_argument("--nameplate-match-threshold", type=float, default=0.72)
    parser.add_argument("--nameplate-input-width", type=int, default=960)
    parser.add_argument("--nameplate-model-dir")
    parser.add_argument("--nameplate-hold-width-ratio", type=float, default=0.16)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)


def _windows_list(args: argparse.Namespace) -> None:
    windows = list_windows(title_filter=args.title)
    if args.json:
        for window in windows:
            _print_json({"event": "window", **asdict(window)})
        return
    for window in windows:
        print(
            f"0x{window.hwnd:X} pid={window.process_id} "
            f"{window.width}x{window.height} minimized={window.minimized} {window.title}"
        )


def _windows_screenshot(args: argparse.Namespace) -> None:
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is missing; install requirements-windows.txt") from exc
    capture = WindowsGraphicsCaptureSource(hwnd=args.hwnd)
    try:
        capture.start()
        frame = capture.read(timeout_seconds=args.timeout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), frame):
            raise RuntimeError(f"failed to write screenshot: {args.output}")
    finally:
        capture.close()
    _print_json(
        {
            "event": "screenshot_saved",
            "hwnd": args.hwnd,
            "path": str(args.output.resolve()),
        }
    )


def _windows_nameplate(args: argparse.Namespace) -> None:
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    capture = WindowsGraphicsCaptureSource(hwnd=args.hwnd)
    try:
        capture.start()
        frame = capture.read(timeout_seconds=args.timeout)
    finally:
        capture.close()
    regions = EasyOcrTextReader(
        input_width=args.nameplate_input_width,
        model_storage_directory=args.nameplate_model_dir,
    ).read(frame)
    match = NameplateMatcher(
        args.target_name,
        minimum_score=args.nameplate_match_threshold,
    ).match(regions)
    _print_json(
        {
            "event": "nameplate_result",
            "matched": match is not None,
            "regions_seen": len(regions),
            "match": asdict(match) if match is not None else None,
        }
    )


def _windows_track(args: argparse.Namespace) -> None:
    hwnd = args.hwnd
    if hwnd is None:
        selector = (
            f"title={args.window_title!r}"
            if args.window_title is not None
            else f"process_id={args.window_process_id}"
        )
        window = wait_for_window(
            title=args.window_title,
            process_id=args.window_process_id,
            timeout_seconds=args.window_wait_timeout,
            heartbeat=lambda elapsed: _print_json(
                {
                    "event": "window_wait_heartbeat",
                    "title": args.window_title,
                    "process_id": args.window_process_id,
                    "elapsed_seconds": elapsed,
                }
            ),
        )
        hwnd = window.hwnd
        _print_json({"event": "window_found", "selector": selector, **asdict(window)})

    nameplate_tracker = None
    if args.target_name:
        nameplate_tracker = AsyncNameplateTracker(
            reader=EasyOcrTextReader(
                input_width=args.nameplate_input_width,
                model_storage_directory=args.nameplate_model_dir,
            ),
            matcher=NameplateMatcher(
                args.target_name,
                minimum_score=args.nameplate_match_threshold,
            ),
            scan_interval_seconds=args.nameplate_scan_interval,
            max_anchor_age_seconds=args.nameplate_anchor_max_age,
        )

    runner = WindowsPerceptionRunner(
        capture=WindowsGraphicsCaptureSource(
            hwnd=hwnd,
            minimum_update_interval_ms=args.capture_interval_ms,
        ),
        tracker=UltralyticsPersonTracker(
            model_path=args.model,
            tracker_config=args.tracker,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
        ),
        selector=TargetFusionSelector(
            require_nameplate_identity=nameplate_tracker is not None,
            reacquire_after_missed_frames=args.reacquire_frames,
            desired_nameplate_width_ratio=args.nameplate_hold_width_ratio,
        ),
        sender=UdpObservationSender(host=args.mac_host, port=args.port),
        nameplate_tracker=nameplate_tracker,
    )

    def heartbeat(status) -> None:
        _print_json({"event": "heartbeat", **asdict(status)})

    try:
        status = runner.run(
            max_frames=args.max_frames,
            duration_seconds=args.duration,
            heartbeat=heartbeat,
        )
    except KeyboardInterrupt:
        _print_json({"event": "stopped", "reason": "keyboard_interrupt"})
        return
    _print_json({"event": "stopped", **asdict(status)})


def _follow_inject(args: argparse.Namespace) -> None:
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.interval < 0 or args.delay < 0:
        raise SystemExit("--interval and --delay must be non-negative")
    if args.delay:
        time.sleep(args.delay)

    session_id = args.session or str(uuid4())
    geometry = (args.center_x, args.proximity)
    if (args.center_x is None) != (args.proximity is None):
        raise SystemExit("--center-x and --proximity must be provided together")
    visible = args.center_x is not None
    sender = UdpObservationSender(host=args.host, port=args.port)
    try:
        for sequence in range(args.repeat):
            observation = TargetObservation(
                session_id=session_id,
                sequence=sequence,
                captured_at_ns=time.perf_counter_ns(),
                visible=visible,
                source=TargetSource(args.source) if visible else None,
                center_x=geometry[0] if visible else None,
                proximity=geometry[1] if visible else None,
                confidence=args.confidence if visible else 0.0,
            )
            sender.send(observation)
            if args.json:
                _print_json(
                    {
                        "event": "observation_sent",
                        "host": args.host,
                        "port": args.port,
                        **asdict(observation),
                    }
                )
            elif sequence == 0 or sequence + 1 == args.repeat:
                print(
                    f"sent seq={sequence} visible={observation.visible} "
                    f"to={args.host}:{args.port}",
                    flush=True,
                )
            if sequence + 1 < args.repeat and args.interval:
                time.sleep(args.interval)
    finally:
        sender.close()


def _follow_listen(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    receiver = UdpObservationReceiver(bind_host=args.bind_host, port=args.port)
    receiver.start()
    deadline = time.monotonic() + args.duration
    last_key: tuple[str, int] | None = None
    next_heartbeat = time.monotonic()
    try:
        while time.monotonic() < deadline:
            snapshot = receiver.store.snapshot()
            if snapshot is not None:
                observation = snapshot.observation
                key = (observation.session_id, observation.sequence)
                if key != last_key:
                    payload = {
                        "event": "observation",
                        "received_monotonic": snapshot.received_monotonic,
                        **asdict(observation),
                    }
                    if args.json:
                        _print_json(payload)
                    else:
                        print(
                            f"observation seq={observation.sequence} "
                            f"visible={observation.visible} source={observation.source} "
                            f"center_x={observation.center_x} proximity={observation.proximity}"
                        )
                    last_key = key
            now = time.monotonic()
            if now >= next_heartbeat:
                status = receiver.status
                if args.json:
                    _print_json({"event": "heartbeat", **asdict(status)})
                else:
                    print(
                        f"heartbeat packets={status.packets_accepted}/{status.packets_received} "
                        f"invalid={status.invalid_packets} error={status.last_error or '-'}"
                    )
                next_heartbeat = now + 1.0
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()


def main() -> None:
    args = build_parser().parse_args()
    if args.group == "windows":
        if args.windows_command == "list":
            _windows_list(args)
        elif args.windows_command == "screenshot":
            _windows_screenshot(args)
        elif args.windows_command == "nameplate":
            _windows_nameplate(args)
        else:
            _windows_track(args)
        return
    if args.follow_command == "inject":
        _follow_inject(args)
    else:
        _follow_listen(args)


if __name__ == "__main__":
    main()
