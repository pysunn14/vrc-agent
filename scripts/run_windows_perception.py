#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.windows_capture import (
    WindowsGraphicsCaptureSource,
    list_windows,
    wait_for_window,
)
from vrc_ardy_agent.windows_perception import (
    PerceptionStatus,
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
    window_selector = run_parser.add_mutually_exclusive_group(required=True)
    window_selector.add_argument("--hwnd", type=parse_hwnd)
    window_selector.add_argument(
        "--window-title",
        help="Case-insensitive title substring; waits until a matching window appears",
    )
    window_selector.add_argument(
        "--window-process-id",
        type=parse_hwnd,
        help="Process id whose top-level window should be captured",
    )
    run_parser.add_argument("--window-wait-timeout", type=float, default=None)
    run_parser.add_argument("--mac-host", required=True, help="Mac LAN or Tailscale address")
    run_parser.add_argument("--port", type=int, default=9200)
    run_parser.add_argument("--model", default="yolov8n.pt")
    run_parser.add_argument("--tracker", default="botsort.yaml")
    run_parser.add_argument("--confidence", type=float, default=0.25)
    run_parser.add_argument("--image-size", type=int, default=640)
    run_parser.add_argument("--device", default=None, help="Ultralytics device, such as cpu or 0")
    run_parser.add_argument("--capture-interval-ms", type=int, default=50)
    run_parser.add_argument("--reacquire-frames", type=int, default=6)
    _add_nameplate_arguments(run_parser)
    run_parser.add_argument("--duration", type=float, default=None)
    run_parser.add_argument("--max-frames", type=int, default=None)
    return parser


def _add_nameplate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-name",
        help="Exact VRChat display name; omit to use body tracking only",
    )
    parser.add_argument("--nameplate-scan-interval", type=float, default=5.0)
    parser.add_argument("--nameplate-anchor-max-age", type=float, default=120.0)
    parser.add_argument("--nameplate-match-threshold", type=float, default=0.72)
    parser.add_argument("--nameplate-input-width", type=int, default=960)
    parser.add_argument("--nameplate-model-dir")
    parser.add_argument("--nameplate-hold-width-ratio", type=float, default=0.16)


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
        f"person_skipped={status.person_inference_skipped} "
        f"visible={status.target_visible} "
        f"source={status.target_source.value if status.target_source else '-'} "
        f"identity={status.target_identity_acquired} "
        f"inference={inference} "
        f"ocr={'scanning' if status.nameplate_scanning else 'idle'} "
        f"ocr_scans={status.nameplate_scans_completed} "
        f"ocr_matches={status.nameplate_matches_found} "
        f"visual_matches={status.nameplate_visual_matches_found}/"
        f"{status.nameplate_visual_updates} "
        f"visual_score={status.last_nameplate_visual_score or 0.0:.2f} "
        f"ocr_error={status.nameplate_error or '-'} "
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

    hwnd = args.hwnd
    if hwnd is None:
        selector = (
            f"title={args.window_title!r}"
            if args.window_title is not None
            else f"process_id={args.window_process_id}"
        )
        print(f"waiting for window {selector}", flush=True)
        window = wait_for_window(
            title=args.window_title,
            process_id=args.window_process_id,
            timeout_seconds=args.window_wait_timeout,
            heartbeat=lambda elapsed: print(
                f"heartbeat: waiting_for_window={selector} elapsed={elapsed:.1f}s",
                flush=True,
            ),
        )
        hwnd = window.hwnd
        print(
            f"window found: hwnd=0x{hwnd:X} pid={window.process_id} "
            f"size={window.width}x{window.height} title={window.title}",
            flush=True,
        )

    capture = WindowsGraphicsCaptureSource(
        hwnd=hwnd,
        minimum_update_interval_ms=args.capture_interval_ms,
    )
    tracker = UltralyticsPersonTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
    )
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
        capture=capture,
        tracker=tracker,
        selector=TargetFusionSelector(
            require_nameplate_identity=nameplate_tracker is not None,
            reacquire_after_missed_frames=args.reacquire_frames,
            desired_nameplate_width_ratio=args.nameplate_hold_width_ratio,
        ),
        sender=UdpObservationSender(host=args.mac_host, port=args.port),
        nameplate_tracker=nameplate_tracker,
    )
    print(
        f"tracking hwnd=0x{hwnd:X}; target={args.mac_host}:{args.port}; "
        f"model={args.model}; tracker={args.tracker}; "
        f"nameplate={'enabled' if nameplate_tracker is not None else 'disabled'}",
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
