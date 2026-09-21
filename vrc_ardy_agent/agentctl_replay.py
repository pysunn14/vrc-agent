"""Finite NPZ replay with an explicit, previously checked room-space idle pose."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from .ardy_runtime import ArdyMotionChunk
from .device_payloads import decode_pose_payload
from .live_sink import SixPointUdpSink
from .face_cue import FaceCueSink, FaceOscOutput
from .motion_replay import MotionReplayRunner
from .stream_bridge import SixPointStreamMapper
from .tracking_rig import AVATAR_PROFILES, get_avatar_profile


def add_replay_arguments(groups):
    parser = groups.add_parser(
        "replay",
        help="Replay a clip, return to a saved pose, then hold until interrupted",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion", type=Path)
    source.add_argument("--frames", type=Path, help="Already calibrated pose frames")
    parser.add_argument(
        "--face-track", type=Path, help="One intensity per motion frame; zero endpoints"
    )
    parser.add_argument("--face-osc-port", type=int, default=9000)
    parser.add_argument("--idle-pose", type=Path, required=True)
    parser.add_argument(
        "--avatar-profile", required=True
    )
    parser.add_argument("--hmd-base", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    parser.add_argument("--host", required=True)
    parser.add_argument("--transition-seconds", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the complete replay without sending packets",
    )
    parser.add_argument(
        "--idle-frames",
        type=int,
        default=None,
        help="Stop after this many idle frames; otherwise keep holding",
    )


def load_motion_frames(args):
    if getattr(args, "frames", None) is not None:
        return [decode_pose_payload(row) for row in json.loads(args.frames.read_text())]
    with np.load(args.motion, allow_pickle=False) as data:
        roots = data["smooth_root_pos"]
        chunk = ArdyMotionChunk(
            posed_joints=data["posed_joints"],
            global_rot_mats=data["global_rot_mats"],
            root_positions=roots,
            smooth_root_pos=roots,
            global_root_heading=data["global_root_heading"],
            foot_contacts=data["foot_contacts"],
            fps=float(data["fps"]),
            prompt="",
            generation_seconds=0.0,
        )
    mapper = SixPointStreamMapper(
        avatar_profile=get_avatar_profile(args.avatar_profile),
        hmd_base=tuple(args.hmd_base),
    )
    return mapper.map_chunk(chunk)


def run_replay(args):
    if args.idle_frames is not None and args.idle_frames <= 0:
        raise ValueError("--idle-frames must be positive")
    if not math.isfinite(args.transition_seconds) or args.transition_seconds <= 0:
        raise ValueError("--transition-seconds must be finite and positive")
    idle = decode_pose_payload(json.loads(args.idle_pose.read_text()))
    frames = load_motion_frames(args)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Dry runs deliberately finish without waiting in an infinite idle loop.
    idle_limit = (
        args.idle_frames
        if args.idle_frames is not None
        else (3 if args.dry_run else None)
    )
    weights = None
    if getattr(args, "face_track", None) is not None:
        weights = json.loads(args.face_track.read_text())
        if not isinstance(weights, list) or len(weights) != len(frames):
            raise ValueError("face track must match motion frame count")
        # Validate the complete input before acquiring any output resources.
        FaceCueSink(None, weights, lambda _: None)
    sink = SixPointUdpSink(
        host=args.host,
        send_locomotion=False,
        disable_trackers_on_close=True,
        dry_run=args.dry_run,
    )
    face_output = None
    try:
        if weights is not None:
            face_output = (
                None if args.dry_run else FaceOscOutput(args.host, args.face_osc_port)
            )
            sink = FaceCueSink(
                sink, weights, (lambda _: None) if face_output is None else face_output
            )
        runner = MotionReplayRunner(
            frames=frames,
            idle=idle,
            sink=sink,
            transition_seconds=args.transition_seconds,
        )
    except BaseException:
        try:
            sink.close()
        finally:
            if face_output is not None:
                face_output.close()
        raise
    last_state, next_report = None, 0.0

    def report(status, force=False):
        nonlocal last_state, next_report
        now = time.monotonic()
        if force or status.state != last_state or now >= next_report:
            payload = {
                **asdict(status),
                "dry_run": args.dry_run,
                "motion_progress": status.motion_frames_sent
                / status.motion_frames_total,
            }
            # Atomic snapshots are observation checkpoints, not automatic resume
            # points: replaying into an unknown current VR pose requires review.
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(args.checkpoint)
            print(json.dumps(payload), flush=True)
            last_state, next_report = status.state, now + 1.0
        if idle_limit is not None and status.idle_frames_sent >= idle_limit:
            runner.request_stop()

    previous_handlers = {}
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(
                number, lambda *_: runner.request_stop()
            )
        report(runner.status, force=True)
        runner.run(realtime=not args.dry_run, heartbeat=report)
    finally:
        # Also covers checkpoint/signal setup failure before runner.run owns it.
        try:
            sink.close()
        finally:
            if face_output is not None:
                face_output.close()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        report(runner.status, force=True)
    return runner.status
