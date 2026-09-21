#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.follow_control import FollowConfig, FollowController
from vrc_ardy_agent.follow_receiver import LatestObservationStore, UdpObservationReceiver
from vrc_ardy_agent.follow_runtime import FollowDecisionLoop, LatestDecisionStore
from vrc_ardy_agent.follow_sink import VmtLocomotionSink, VrchatLocomotionSink
from vrc_ardy_agent.live_sink import SixPointUdpSink
from vrc_ardy_agent.pose_runtime import PoseStreamRunner, PoseStreamStatus
from vrc_ardy_agent.tracking_rig import (
    AVATAR_PROFILES,
    HandSelection,
    NeutralPoseConfig,
    build_neutral_tracking_frame,
    get_avatar_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Follow one observed VRChat player with head-only tracking. "
            "This runner does not load ARDY."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="Windows host running VRChat/SteamVR")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds; defaults to running until interrupted",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heartbeat-interval", type=float, default=1.0)

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

    parser.add_argument("--follow-bind-host", default="0.0.0.0")
    parser.add_argument("--follow-port", type=int, default=9200)
    parser.add_argument("--follow-tick-hz", type=float, default=20.0)
    parser.add_argument("--follow-stale-seconds", type=float, default=1.0)
    parser.add_argument(
        "--follow-active-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--follow-search-delay", type=float, default=0.75)
    parser.add_argument("--follow-search-sweep-seconds", type=float, default=6.0)
    parser.add_argument("--follow-search-turn", type=float, default=0.35)
    parser.add_argument(
        "--follow-output",
        choices=("vmt", "vrchat-osc"),
        default="vrchat-osc",
    )
    parser.add_argument(
        "--follow-osc-turn-buttons",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--follow-turn-pulse-seconds", type=float, default=0.5)
    parser.add_argument("--follow-relocate-turn-seconds", type=float, default=1.0)
    parser.add_argument("--follow-relocate-forward-seconds", type=float, default=1.0)
    parser.add_argument("--follow-relocate-forward", type=float, default=0.2)
    parser.add_argument("--follow-align-enter", type=float, default=0.2)
    parser.add_argument("--follow-align-exit", type=float, default=0.1)
    parser.add_argument("--follow-resume-height", type=float, default=0.35)
    parser.add_argument("--follow-hold-height", type=float, default=0.45)
    parser.add_argument("--follow-turn-gain", type=float, default=1.0)
    parser.add_argument("--follow-max-turn", type=float, default=0.6)
    parser.add_argument("--follow-forward-gain", type=float, default=1.0)
    parser.add_argument("--follow-max-forward", type=float, default=0.5)
    parser.add_argument("--follow-smoothing-alpha", type=float, default=0.35)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.pose_rate_hz <= 0:
        raise ValueError("--pose-rate-hz must be positive")
    if args.follow_tick_hz <= 0:
        raise ValueError("--follow-tick-hz must be positive")
    if args.heartbeat_interval <= 0:
        raise ValueError("--heartbeat-interval must be positive")
    if args.follow_output == "vmt" and args.hands != HandSelection.BOTH.value:
        raise ValueError("VMT locomotion requires both hands so both controller inputs stay active")


def _build_follow_config(args: argparse.Namespace) -> FollowConfig:
    return FollowConfig(
        stale_after_seconds=args.follow_stale_seconds,
        active_search=args.follow_active_search,
        search_delay_seconds=args.follow_search_delay,
        search_sweep_seconds=args.follow_search_sweep_seconds,
        search_turn=args.follow_search_turn,
        relocate_turn_seconds=args.follow_relocate_turn_seconds,
        relocate_forward_seconds=args.follow_relocate_forward_seconds,
        relocate_forward=args.follow_relocate_forward,
        align_enter_error=args.follow_align_enter,
        align_exit_error=args.follow_align_exit,
        resume_follow_below_height=args.follow_resume_height,
        hold_above_height=args.follow_hold_height,
        turn_gain=args.follow_turn_gain,
        max_turn=args.follow_max_turn,
        forward_gain=args.follow_forward_gain,
        max_forward=args.follow_max_forward,
        smoothing_alpha=args.follow_smoothing_alpha,
    )


def _print_heartbeat(
    pose_status: PoseStreamStatus,
    *,
    receiver: UdpObservationReceiver,
    loop: FollowDecisionLoop,
    decisions: LatestDecisionStore,
) -> None:
    receiver_status = receiver.status
    loop_status = loop.status
    decision = decisions.snapshot()
    age = (
        "-"
        if decision.observation_age_seconds is None
        else f"{decision.observation_age_seconds:.3f}s"
    )
    print(
        "heartbeat: "
        f"pose_frames={pose_status.frames_sent} "
        f"follow={decision.state.value} "
        f"vertical={decision.vertical:.3f} "
        f"turn={decision.look_horizontal:.3f} "
        f"observation_age={age} "
        f"packets={receiver_status.packets_accepted}/{receiver_status.packets_received} "
        f"invalid={receiver_status.invalid_packets} "
        f"control_ticks={loop_status.ticks} "
        f"pose_error={pose_status.last_error or '-'} "
        f"receiver_error={receiver_status.last_error or '-'} "
        f"controller_error={loop_status.last_error or '-'}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        pose = build_neutral_tracking_frame(
            get_avatar_profile(args.avatar_profile),
            NeutralPoseConfig(
                hmd_base=tuple(args.hmd_base),
                fps=args.pose_rate_hz,
                hands=HandSelection(args.hands),
                reach_fraction=args.hand_reach_fraction,
                hand_direction=tuple(args.hand_direction),
                left_hand_euler_deg=tuple(args.left_hand_euler),
                right_hand_euler_deg=tuple(args.right_hand_euler),
                left_enable=args.left_enable,
                right_enable=args.right_enable,
            ),
        )
        follow_config = _build_follow_config(args)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    pose_sink = SixPointUdpSink(
        host=args.host,
        opentrack_port=args.opentrack_port,
        vmt_port=args.vmt_port,
        vrchat_port=args.vrchat_port,
        left_tracker_index=args.left_vmt_index,
        right_tracker_index=args.right_vmt_index,
        hips_tracker_index=args.hips_vmt_index,
        left_foot_tracker_index=args.left_foot_vmt_index,
        right_foot_tracker_index=args.right_foot_vmt_index,
        left_enable=pose.left_enable,
        right_enable=pose.right_enable,
        # Zero is sent every frame so stale full-body trackers cannot keep
        # deforming the avatar after switching from an ARDY experiment.
        body_enable=pose.body_enable,
        send_locomotion=False,
        lock_head_rotation=True,
        dry_run=args.dry_run,
    )
    pose_runner = PoseStreamRunner(
        frame=pose.frame,
        sink=pose_sink,
        rate_hz=args.pose_rate_hz,
    )

    observation_store = LatestObservationStore(
        session_takeover_after_seconds=args.follow_stale_seconds,
    )
    receiver = UdpObservationReceiver(
        bind_host=args.follow_bind_host,
        port=args.follow_port,
        store=observation_store,
    )
    decisions = LatestDecisionStore()
    if args.follow_output == "vmt":
        locomotion_sink = VmtLocomotionSink(
            host=args.host,
            port=args.vmt_port,
            left_controller_index=args.left_vmt_index,
            right_controller_index=args.right_vmt_index,
            dry_run=args.dry_run,
            turn_pulse_interval_seconds=args.follow_turn_pulse_seconds,
        )
    else:
        locomotion_sink = VrchatLocomotionSink(
            host=args.host,
            port=args.vrchat_port,
            dry_run=args.dry_run,
            turn_buttons=args.follow_osc_turn_buttons,
            turn_pulse_interval_seconds=args.follow_turn_pulse_seconds,
        )
    follow_loop = FollowDecisionLoop(
        observation_store=observation_store,
        controller=FollowController(follow_config),
        decision_store=decisions,
        on_decision=locomotion_sink.send,
        tick_hz=args.follow_tick_hz,
    )

    target = "dry-run" if args.dry_run else args.host
    print(
        "starting follower without ARDY: "
        f"target={target} avatar_profile={args.avatar_profile} hands={args.hands} "
        f"left={pose.left.position} right={pose.right.position} body_trackers=off",
        flush=True,
    )
    try:
        receiver.start()
        follow_loop.start()
        pose_runner.run(
            duration_seconds=args.duration,
            heartbeat_interval_seconds=args.heartbeat_interval,
            heartbeat=lambda status: _print_heartbeat(
                status,
                receiver=receiver,
                loop=follow_loop,
                decisions=decisions,
            ),
        )
    except KeyboardInterrupt:
        pose_runner.request_stop()
        print("stopped by user", flush=True)
    finally:
        follow_loop.stop()
        locomotion_sink.close()
        receiver.stop()

    status = pose_runner.status
    print(
        f"follower finished: pose_frames={status.frames_sent} error={status.last_error or '-'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
