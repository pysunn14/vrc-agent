#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.ardy_runtime import ArdyHeadlessRuntime
from vrc_ardy_agent.live_control import IDLE_PROMPT, apply_control_line
from vrc_ardy_agent.live_session import LiveArdySession, LiveSessionStatus
from vrc_ardy_agent.live_sink import SixPointUdpSink
from vrc_ardy_agent.stream_bridge import SixPointStreamMapper
from vrc_ardy_agent.tracking_rig import AVATAR_PROFILES, get_avatar_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run resident ARDY body-motion generation directly into VRChat. "
            "Player following is owned by scripts/run_follow.py."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=IDLE_PROMPT,
        help="Initial motion prompt; defaults to a natural standing pose",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Windows host running VRChat/SteamVR")
    parser.add_argument("--duration", type=float, default=30.0, help="Live playback duration in seconds")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read prompt/idle/stop/quit commands from stdin and keep ARDY resident until quit or EOF",
    )
    parser.add_argument("--model", default="core", help="ARDY model name")
    parser.add_argument("--device", default="mps", help="ARDY inference device")
    parser.add_argument(
        "--checkpoints-dir",
        default=os.environ.get("CHECKPOINTS_DIR"),
        help="ARDY checkpoint root; defaults to CHECKPOINTS_DIR",
    )
    parser.add_argument(
        "--text-encoder-mode",
        default=os.environ.get("TEXT_ENCODER_MODE"),
        help="ARDY text encoder mode; defaults to TEXT_ENCODER_MODE",
    )
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help=(
            "ARDY history frames retained per generation step; use the model's "
            "token size for the fastest prompt adaptation"
        ),
    )
    parser.add_argument("--replan-threshold-frames", type=int, default=None)

    parser.add_argument("--opentrack-port", type=int, default=4242)
    parser.add_argument("--vmt-port", type=int, default=39570)
    parser.add_argument("--vrchat-port", type=int, default=9000)
    parser.add_argument("--left-vmt-index", type=int, default=1)
    parser.add_argument("--right-vmt-index", type=int, default=2)
    parser.add_argument("--hips-vmt-index", type=int, default=3)
    parser.add_argument("--left-foot-vmt-index", type=int, default=4)
    parser.add_argument("--right-foot-vmt-index", type=int, default=5)
    parser.add_argument("--left-enable", type=int, default=5)
    parser.add_argument("--right-enable", type=int, default=6)
    parser.add_argument("--body-enable", type=int, default=7)
    parser.add_argument("--no-locomotion", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate/map/play in real time without UDP sends")
    parser.add_argument("--park-head-on-exit", action="store_true")

    parser.add_argument(
        "--hmd-base",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 1.0, 0.0),
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument(
        "--avatar-profile",
        default="felis",
    )
    parser.add_argument("--full-stick-speed-mps", type=float, default=2.5)
    parser.add_argument("--locomotion-deadzone-mps", type=float, default=0.08)
    parser.add_argument("--full-turn-speed-dps", type=float, default=180.0)
    parser.add_argument("--turn-deadzone-dps", type=float, default=12.0)
    parser.add_argument("--heading-smoothing-seconds", type=float, default=0.5)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def print_heartbeat(
    status: LiveSessionStatus,
    *,
    mapper: SixPointStreamMapper | None = None,
) -> None:
    generation = "-" if status.last_generation_seconds is None else f"{status.last_generation_seconds:.3f}s"
    body_state = ""
    if mapper is not None:
        body = mapper.lower_body_monitor.snapshot()
        body_state = (
            f" body_frames={body.frames}"
            f" body_clamps={body.workspace_clamp_count + body.floor_clamp_count + body.leg_clamp_count + body.speed_clamp_count}"
        )
    print(
        "heartbeat: "
        f"played={status.played_frames} "
        f"buffered={status.buffered_frames} "
        f"chunks={status.generated_chunks} "
        f"paused={status.paused} "
        f"generating={status.generation_in_progress} "
        f"last_gen={generation} "
        f"underruns={status.underruns}"
        f"{body_state}",
        flush=True,
    )


def run_interactive_controls(session: LiveArdySession) -> None:
    print(
        "interactive controls: type a motion prompt, or use: idle | stop | quit",
        flush=True,
    )
    try:
        for line in sys.stdin:
            try:
                result = apply_control_line(session, line)
            except Exception as exc:
                print(f"control error: {exc}", flush=True)
                continue
            if result.message:
                print(f"control: {result.message}", flush=True)
            if result.quit_requested:
                return
    finally:
        session.request_stop()


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")

    mode = "dry-run" if args.dry_run else args.host
    print(
        f"loading resident ARDY model={args.model} device={args.device}; output={mode}",
        flush=True,
    )
    runtime = ArdyHeadlessRuntime.load(
        model_name=args.model,
        device=args.device,
        checkpoints_dir=args.checkpoints_dir,
        text_encoder_mode=args.text_encoder_mode,
        history_limit_frames=args.history_frames,
        postprocess=not args.no_postprocess,
        num_denoising_steps=args.diffusion_steps,
    )
    print(
        f"ARDY ready: fps={runtime.fps:g}, horizon={runtime.horizon_frames} frames, "
        f"history={runtime.history_limit_frames} frames",
        flush=True,
    )

    mapper = SixPointStreamMapper(
        avatar_profile=get_avatar_profile(args.avatar_profile),
        hmd_base=tuple(args.hmd_base),
        scale=args.scale,
        full_stick_speed_mps=args.full_stick_speed_mps,
        locomotion_deadzone_mps=args.locomotion_deadzone_mps,
        full_turn_speed_dps=args.full_turn_speed_dps,
        turn_deadzone_dps=args.turn_deadzone_dps,
        heading_smoothing_seconds=args.heading_smoothing_seconds,
    )
    sink = SixPointUdpSink(
        host=args.host,
        opentrack_port=args.opentrack_port,
        vmt_port=args.vmt_port,
        vrchat_port=args.vrchat_port,
        left_tracker_index=args.left_vmt_index,
        right_tracker_index=args.right_vmt_index,
        hips_tracker_index=args.hips_vmt_index,
        left_foot_tracker_index=args.left_foot_vmt_index,
        right_foot_tracker_index=args.right_foot_vmt_index,
        left_enable=args.left_enable,
        right_enable=args.right_enable,
        body_enable=args.body_enable,
        send_locomotion=not args.no_locomotion,
        park_head_on_close=args.park_head_on_exit,
        dry_run=args.dry_run,
    )
    session = LiveArdySession(
        runtime=runtime,
        mapper=mapper,
        sink=sink,
        replan_threshold_frames=args.replan_threshold_frames,
    )

    if args.interactive:
        print(f"starting interactive live motion: {args.prompt!r}", flush=True)
    else:
        print(f"starting live motion for {args.duration:g}s: {args.prompt!r}", flush=True)

    try:
        if args.interactive:
            session.start(args.prompt)
            control_thread = threading.Thread(
                target=run_interactive_controls,
                args=(session,),
                name="ardy-stdin-control",
                daemon=True,
            )
            control_thread.start()
            status = session.run_started(
                duration_seconds=None,
                realtime=True,
                heartbeat=lambda status: print_heartbeat(status, mapper=mapper),
            )
        else:
            status = session.run(
                prompt=args.prompt,
                duration_seconds=args.duration,
                realtime=True,
                heartbeat=lambda status: print_heartbeat(status, mapper=mapper),
            )
    except KeyboardInterrupt:
        session.request_stop()
        print("stopped by user", flush=True)
        return

    print(
        "live motion finished: "
        f"played={status.played_frames}, chunks={status.generated_chunks}, underruns={status.underruns}",
        flush=True,
    )


if __name__ == "__main__":
    main()
