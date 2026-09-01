#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.windows_capture import WindowsGraphicsCaptureSource, list_windows
from vrc_ardy_agent.windows_perception import (
    PerceptionStatus,
    SingleTargetSelector,
    UdpObservationSender,
    UltralyticsPersonTracker,
    WindowsPerceptionRunner,
)


def parse_hwnd(value: str) -> int:
    try:
        hwnd = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HWND must be a decimal or 0x-prefixed integer") from exc
    if hwnd <= 0:
        raise argparse.ArgumentTypeError("HWND must be positive")
    return hwnd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture AgentAvatar's VRChat window, track one person, and stream target state to Mac."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-windows", help="List capturable top-level windows")
    list_parser.add_argument("--title", help="Case-insensitive title substring filter")

    run_parser = subparsers.add_parser("run", help="Run window capture and person tracking")
    run_parser.add_argument("--hwnd", required=True, type=parse_hwnd)
    run_parser.add_argument("--mac-host", required=True, help="Mac LAN or Tailscale address")
    run_parser.add_argument("--port", type=int, default=9200)
    run_parser.add_argument("--model", default="yolov8n.pt")
    run_parser.add_argument("--tracker", default="botsort.yaml")
    run_parser.add_argument("--confidence", type=float, default=0.25)
    run_parser.add_argument("--image-size", type=int, default=640)
    run_parser.add_argument("--device", default=None, help="Ultralytics device, such as cpu or 0")
    run_parser.add_argument("--capture-interval-ms", type=int, default=50)
    run_parser.add_argument("--reacquire-frames", type=int, default=6)
    run_parser.add_argument("--duration", type=float, default=None)
    run_parser.add_argument("--max-frames", type=int, default=None)
    return parser


def print_heartbeat(status: PerceptionStatus) -> None:
    inference = (
        "-"
        if status.last_inference_seconds is None
        else f"{status.last_inference_seconds * 1000.0:.1f}ms"
    )
    print(
        "heartbeat: "
        f"frames={status.frames_processed} "
        f"sent={status.observations_sent} "
        f"visible={status.target_visible} "
        f"inference={inference} "
        f"error={status.last_error or '-'}",
        flush=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list-windows":
        windows = list_windows(title_filter=args.title)
        if not windows:
            print("no matching windows")
            return
        for window in windows:
            state = "minimized" if window.minimized else "visible"
            print(
                f"hwnd=0x{window.hwnd:X} pid={window.process_id} "
                f"size={window.width}x{window.height} state={state} title={window.title}"
            )
        return

    capture = WindowsGraphicsCaptureSource(
        hwnd=args.hwnd,
        minimum_update_interval_ms=args.capture_interval_ms,
    )
    tracker = UltralyticsPersonTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
    )
    runner = WindowsPerceptionRunner(
        capture=capture,
        tracker=tracker,
        selector=SingleTargetSelector(
            reacquire_after_missed_frames=args.reacquire_frames,
        ),
        sender=UdpObservationSender(host=args.mac_host, port=args.port),
    )
    print(
        f"tracking hwnd=0x{args.hwnd:X}; target={args.mac_host}:{args.port}; "
        f"model={args.model}; tracker={args.tracker}",
        flush=True,
    )
    try:
        status = runner.run(
            max_frames=args.max_frames,
            duration_seconds=args.duration,
            heartbeat=print_heartbeat,
        )
    except KeyboardInterrupt:
        print("stopped by user", flush=True)
        return
    print_heartbeat(status)


if __name__ == "__main__":
    main()
