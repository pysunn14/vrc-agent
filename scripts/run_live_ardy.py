#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.ardy_runtime import ArdyHeadlessRuntime
from vrc_ardy_agent.follow_control import FollowConfig, FollowController
from vrc_ardy_agent.follow_receiver import LatestObservationStore, UdpObservationReceiver
from vrc_ardy_agent.follow_runtime import (
    FollowDecisionLoop,
    FollowPromptRouter,
    LatestDecisionStore,
)
from vrc_ardy_agent.follow_sink import VmtLocomotionSink, VrchatLocomotionSink
from vrc_ardy_agent.live_control import IDLE_PROMPT, apply_control_line
from vrc_ardy_agent.live_session import LiveArdySession, LiveSessionStatus
from vrc_ardy_agent.live_sink import SixPointUdpSink
from vrc_ardy_agent.stream_bridge import SixPointStreamMapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resident ARDY autoregressive generation directly into VRChat without NPZ files."
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
        "--follow",
        action="store_true",
        help="Receive target observations and let the follow controller own VRChat locomotion",
    )
    parser.add_argument("--follow-bind-host", default="0.0.0.0")
    parser.add_argument("--follow-port", type=int, default=9200)
    parser.add_argument("--follow-tick-hz", type=float, default=20.0)
    parser.add_argument("--follow-stale-seconds", type=float, default=1.0)
    parser.add_argument(
        "--follow-active-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Actively scan and relocate after losing the target",
    )
    parser.add_argument("--follow-search-delay", type=float, default=0.75)
    parser.add_argument("--follow-search-sweep-seconds", type=float, default=6.0)
    parser.add_argument("--follow-search-turn", type=float, default=0.35)
    parser.add_argument(
        "--follow-output",
        choices=("vmt", "vrchat-osc"),
        default="vmt",
        help="Drive the active VMT hand controllers or VRChat OSC input",
    )
    parser.add_argument(
        "--follow-osc-turn-buttons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pulse VRChat turn buttons when --follow-output=vrchat-osc",
    )
    parser.add_argument(
        "--follow-turn-pulse-seconds",
        type=float,
        default=0.5,
        help="Interval between comfort-turn input pulses",
    )
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
    parser.add_argument(
        "--follow-walking-prompt",
        default="A person walks forward naturally.",
    )
    parser.add_argument("--follow-idle-prompt", default=IDLE_PROMPT)

    parser.add_argument(
        "--hmd-base",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 1.0, 0.0),
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--full-stick-speed-mps", type=float, default=2.5)
    parser.add_argument("--locomotion-deadzone-mps", type=float, default=0.08)
    parser.add_argument("--full-turn-speed-dps", type=float, default=180.0)
    parser.add_argument("--turn-deadzone-dps", type=float, default=12.0)
    parser.add_argument("--heading-smoothing-seconds", type=float, default=0.5)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def print_heartbeat(status: LiveSessionStatus) -> None:
    generation = "-" if status.last_generation_seconds is None else f"{status.last_generation_seconds:.3f}s"
    print(
        "heartbeat: "
        f"played={status.played_frames} "
        f"buffered={status.buffered_frames} "
        f"chunks={status.generated_chunks} "
        f"paused={status.paused} "
        f"generating={status.generation_in_progress} "
        f"last_gen={generation} "
        f"underruns={status.underruns}",
        flush=True,
    )


def print_follow_heartbeat(
    status: LiveSessionStatus,
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
        f"played={status.played_frames} "
        f"buffered={status.buffered_frames} "
        f"follow={decision.state.value} "
        f"vertical={decision.vertical:.3f} "
        f"turn={decision.look_horizontal:.3f} "
        f"observation_age={age} "
        f"packets={receiver_status.packets_accepted}/{receiver_status.packets_received} "
        f"invalid={receiver_status.invalid_packets} "
        f"control_ticks={loop_status.ticks} "
        f"receiver_error={receiver_status.last_error or '-'} "
        f"controller_error={loop_status.last_error or '-'}",
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
    if args.follow and args.interactive:
        raise SystemExit("--follow and --interactive cannot own ARDY prompts at the same time")
    if args.follow and args.no_locomotion:
        raise SystemExit("--follow requires locomotion output")

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
        postprocess=not args.no_postprocess,
        num_denoising_steps=args.diffusion_steps,
    )
    print(
        f"ARDY ready: fps={runtime.fps:g}, horizon={runtime.horizon_frames} frames, "
        f"history={runtime.history_limit_frames} frames",
        flush=True,
    )

    mapper = SixPointStreamMapper(
        hmd_base=tuple(args.hmd_base),
        scale=args.scale,
        full_stick_speed_mps=args.full_stick_speed_mps,
        locomotion_deadzone_mps=args.locomotion_deadzone_mps,
        full_turn_speed_dps=args.full_turn_speed_dps,
        turn_deadzone_dps=args.turn_deadzone_dps,
        heading_smoothing_seconds=args.heading_smoothing_seconds,
    )
    base_sink = SixPointUdpSink(
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
        send_locomotion=not args.no_locomotion and not args.follow,
        # Follow perception observes the HMD view itself. Generated head rotation
        # would move that sensor frame and can point the camera away from its target.
        lock_head_rotation=args.follow,
        park_head_on_close=args.park_head_on_exit,
        dry_run=args.dry_run,
    )

    receiver: UdpObservationReceiver | None = None
    follow_loop: FollowDecisionLoop | None = None
    decision_store: LatestDecisionStore | None = None
    locomotion_sink: VrchatLocomotionSink | VmtLocomotionSink | None = None
    if args.follow:
        observation_store = LatestObservationStore(
            session_takeover_after_seconds=args.follow_stale_seconds,
        )
        receiver = UdpObservationReceiver(
            bind_host=args.follow_bind_host,
            port=args.follow_port,
            store=observation_store,
        )
        decision_store = LatestDecisionStore()
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
    else:
        observation_store = None

    session = LiveArdySession(
        runtime=runtime,
        mapper=mapper,
        sink=base_sink,
        replan_threshold_frames=args.replan_threshold_frames,
    )

    if args.follow:
        assert observation_store is not None
        assert decision_store is not None
        assert locomotion_sink is not None
        prompt_router = FollowPromptRouter(
            set_prompt=session.set_prompt,
            initial_prompt=args.prompt,
            walking_prompt=args.follow_walking_prompt,
            idle_prompt=args.follow_idle_prompt,
        )
        follow_loop = FollowDecisionLoop(
            observation_store=observation_store,
            controller=FollowController(
                FollowConfig(
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
            ),
            decision_store=decision_store,
            on_decision=locomotion_sink.send,
            on_state_change=prompt_router,
            tick_hz=args.follow_tick_hz,
        )

    if args.interactive:
        print(f"starting interactive live motion: {args.prompt!r}", flush=True)
    else:
        print(
            f"starting live motion for {args.duration:g}s: {args.prompt!r}",
            flush=True,
        )

    follow_session_started = False
    try:
        if args.follow:
            assert receiver is not None
            assert follow_loop is not None
            assert decision_store is not None
            print(
                f"starting follow mode: udp={args.follow_bind_host}:{args.follow_port}",
                flush=True,
            )
            session.start(args.prompt)
            follow_session_started = True
            receiver.start()
            follow_loop.start()
            status = session.run_started(
                duration_seconds=args.duration,
                realtime=True,
                heartbeat=lambda current: print_follow_heartbeat(
                    current,
                    receiver=receiver,
                    loop=follow_loop,
                    decisions=decision_store,
                ),
            )
        elif args.interactive:
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
                heartbeat=print_heartbeat,
            )
        else:
            status = session.run(
                prompt=args.prompt,
                duration_seconds=args.duration,
                realtime=True,
                heartbeat=print_heartbeat,
            )
    except KeyboardInterrupt:
        session.request_stop()
        print("stopped by user", flush=True)
        return
    finally:
        if follow_loop is not None:
            follow_loop.stop()
        if locomotion_sink is not None:
            locomotion_sink.close()
        if receiver is not None:
            receiver.stop()
        if follow_session_started and session.status.running:
            session.stop()

    print(
        "live motion finished: "
        f"played={status.played_frames}, chunks={status.generated_chunks}, underruns={status.underruns}",
        flush=True,
    )


if __name__ == "__main__":
    main()
