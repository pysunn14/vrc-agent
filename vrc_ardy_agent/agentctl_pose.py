from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from vrc_ardy_agent.live_sink import SixPointUdpSink
from vrc_ardy_agent.pose_runtime import PoseStreamRunner
from vrc_ardy_agent.tracking_rig import (
    AVATAR_PROFILES,
    BodyTrackingMode,
    HandSelection,
    NeutralPoseConfig,
    RigCalibration,
    build_neutral_tracking_frame,
    get_avatar_profile,
)

def _print_json(payload):
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _add_pose_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--avatar-profile", default="felis"
    )
    parser.add_argument("--pose-rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--hands",
        choices=[selection.value for selection in HandSelection],
        default=HandSelection.NONE.value,
    )
    parser.add_argument(
        "--body-trackers",
        choices=[mode.value for mode in BodyTrackingMode],
        default=BodyTrackingMode.PROFILE.value,
    )
    parser.add_argument("--body-enable", type=int, default=7)
    parser.add_argument(
        "--hmd-base",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 1.0, 0.0),
    )
    parser.add_argument("--hand-reach-fraction", type=float, default=0.85)
    parser.add_argument(
        "--hand-direction",
        type=float,
        nargs=3,
        metavar=("OUT", "DOWN", "FORWARD"),
        default=(0.25, 0.94, 0.20),
    )
    parser.add_argument(
        "--left-hand-euler",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(90.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--right-hand-euler",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(90.0, 0.0, 0.0),
    )


def _build_neutral_pose(args: argparse.Namespace):
    profile = get_avatar_profile(args.avatar_profile)
    return build_neutral_tracking_frame(
        profile,
        NeutralPoseConfig(
            hmd_base=tuple(args.hmd_base),
            fps=args.pose_rate_hz,
            hands=HandSelection(args.hands),
            reach_fraction=args.hand_reach_fraction,
            hand_direction=tuple(args.hand_direction),
            left_hand_euler_deg=tuple(args.left_hand_euler),
            right_hand_euler_deg=tuple(args.right_hand_euler),
            body_tracking=BodyTrackingMode(args.body_trackers),
            left_enable=getattr(args, "left_enable", 5),
            right_enable=getattr(args, "right_enable", 6),
            body_enable=args.body_enable,
        ),
    )


def _pose_show(args: argparse.Namespace) -> None:
    pose = _build_neutral_pose(args)
    profile = get_avatar_profile(args.avatar_profile)
    calibration = RigCalibration.from_avatar_profile(profile)
    payload = {
        "event": "neutral_pose",
        "avatar_profile": args.avatar_profile,
        "hmd_base": tuple(args.hmd_base),
        "left": {
            "enabled": pose.left_enable != 0,
            "position": pose.left.position,
            "quaternion_xyzw": pose.left.quaternion_xyzw,
            "shoulder": calibration.shoulder_position(tuple(args.hmd_base), side="left"),
        },
        "right": {
            "enabled": pose.right_enable != 0,
            "position": pose.right.position,
            "quaternion_xyzw": pose.right.quaternion_xyzw,
            "shoulder": calibration.shoulder_position(tuple(args.hmd_base), side="right"),
        },
        "arm_reach_m": calibration.scaled_arm_reach(args.hmd_base[1]),
        "used_reach_fraction": args.hand_reach_fraction,
        "body_trackers": {
            "mode": args.body_trackers,
            "enabled": pose.body_enable != 0,
            "enable": pose.body_enable,
            "hips": {
                "position": pose.frame.hips.position,
                "quaternion_xyzw": pose.frame.hips.quaternion_xyzw,
            },
            "left_foot": {
                "position": pose.frame.left_foot.position,
                "quaternion_xyzw": pose.frame.left_foot.quaternion_xyzw,
            },
            "right_foot": {
                "position": pose.frame.right_foot.position,
                "quaternion_xyzw": pose.frame.right_foot.quaternion_xyzw,
            },
        },
    }
    if getattr(args, "json", False):
        _print_json(payload)
        return
    print(
        f"avatar_profile={args.avatar_profile} body_trackers={args.body_trackers} "
        f"arm_reach={payload['arm_reach_m']:.4f}m used={args.hand_reach_fraction:.2f}",
        flush=True,
    )
    for side in ("left", "right"):
        hand = payload[side]
        print(
            f"{side}: enabled={hand['enabled']} position={hand['position']} "
            f"quaternion={hand['quaternion_xyzw']}",
            flush=True,
        )
    if pose.body_enable != 0:
        for name in ("hips", "left_foot", "right_foot"):
            tracker = payload["body_trackers"][name]
            print(
                f"{name}: enabled=True position={tracker['position']} "
                f"quaternion={tracker['quaternion_xyzw']}",
                flush=True,
            )


def _pose_hold(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    pose = _build_neutral_pose(args)
    sink = SixPointUdpSink(
        host=args.host,
        opentrack_port=args.opentrack_port,
        vmt_port=args.vmt_port,
        left_tracker_index=args.left_vmt_index,
        right_tracker_index=args.right_vmt_index,
        hips_tracker_index=args.hips_vmt_index,
        left_foot_tracker_index=args.left_foot_vmt_index,
        right_foot_tracker_index=args.right_foot_vmt_index,
        left_enable=pose.left_enable,
        right_enable=pose.right_enable,
        body_enable=pose.body_enable,
        send_locomotion=False,
        lock_head_rotation=True,
        # A bounded exploratory hold must not leave synthetic body trackers
        # alive at their final packet after the command exits.
        disable_trackers_on_close=True,
        dry_run=args.dry_run,
    )
    runner = PoseStreamRunner(
        frame=pose.frame,
        sink=sink,
        rate_hz=args.pose_rate_hz,
    )
    _pose_show(args)
    try:
        status = runner.run(
            duration_seconds=args.duration,
            heartbeat=lambda current: _print_json(
                {"event": "pose_heartbeat", **asdict(current)}
            ),
        )
    except KeyboardInterrupt:
        runner.request_stop()
        status = runner.status
    _print_json({"event": "pose_stopped", **asdict(status)})


