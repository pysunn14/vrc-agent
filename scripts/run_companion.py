#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.action_contracts import ActionType
from vrc_ardy_agent.ardy_runtime import ArdyHeadlessRuntime
from vrc_ardy_agent.audio_interaction_pipeline import AudioInteractionPipeline
from vrc_ardy_agent.character_supervisor import CharacterSupervisor
from vrc_ardy_agent.companion_api import CompanionControlService, create_companion_server
from vrc_ardy_agent.companion_app import MacCompanionApp
from vrc_ardy_agent.companion_http_runtime import CompanionHttpRuntime
from vrc_ardy_agent.energy_vad import EnergyVadSegmenter
from vrc_ardy_agent.providers.factory import build_provider
from vrc_ardy_agent.runner.launch import apply_runtime_profile
from vrc_ardy_agent.runner.settings import config_directory
from vrc_ardy_agent.interaction_runtime import InteractionRuntime
from vrc_ardy_agent.mac_bridge_server import MacBridgeServer
from vrc_ardy_agent.mac_input_transport import MacInputTransport
from vrc_ardy_agent.mac_output_transport import MacOutputTransport
from vrc_ardy_agent.humanoid_retargeting import RetargetingMonitor
from vrc_ardy_agent.lower_body_retargeting import LowerBodyRetargetingMonitor
from vrc_ardy_agent.motion_director import MotionDirector
from vrc_ardy_agent.anchored_motion import AnchoredMotionSession
from vrc_ardy_agent.motion_assets import MotionAssets
from vrc_ardy_agent.motion_stream import DEFAULT_POSE_TTL_MS, CALIBRATED_IDLE
from vrc_ardy_agent.speech_executor import SpeechExecutor
from vrc_ardy_agent.status_serialization import to_jsonable
from vrc_ardy_agent.stream_bridge import RetargetingMode, SixPointStreamMapper
from vrc_ardy_agent.tracking_rig import get_avatar_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the agent's configured providers, speech, and ARDY motion runtime."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=config_directory() / "agent.json",
    )
    parser.add_argument("--bridge-host", default="0.0.0.0")
    parser.add_argument("--bridge-port", type=int)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--enable-test-controls", action="store_true")
    parser.add_argument("--manual-actions-only", action="store_true",
                        help="Receive sensor data without automatically reacting to speech")
    parser.add_argument("--heartbeat-interval", type=float, default=2.0)

    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
    )
    parser.add_argument(
        "--text-encoder-mode",
        default=os.environ.get("TEXT_ENCODER_MODE"),
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
    parser.add_argument("--transition-seconds", type=float, default=1.)
    parser.add_argument("--replan-threshold-frames", type=int, default=None)
    parser.add_argument(
        "--pose-ttl-ms",
        type=int,
        default=DEFAULT_POSE_TTL_MS,
    )
    parser.add_argument("--full-stick-speed-mps", type=float, default=2.5)
    parser.add_argument("--locomotion-deadzone-mps", type=float, default=0.08)
    parser.add_argument("--full-turn-speed-dps", type=float, default=180.0)
    parser.add_argument("--turn-deadzone-dps", type=float, default=12.0)
    parser.add_argument("--heading-smoothing-seconds", type=float, default=0.5)
    parser.add_argument(
        "--retargeting-mode",
        choices=tuple(mode.value for mode in RetargetingMode),
        default=RetargetingMode.FULL.value,
    )

    parser.add_argument("--max-observations", type=int, default=2)
    parser.add_argument("--observation-timeout-seconds", type=float, default=30.)
    parser.add_argument("--observation-max-frame-age", type=float, default=15.)
    parser.add_argument("--vad-rms-threshold", type=int, default=500)
    parser.add_argument("--vad-start-ms", type=int, default=60)
    parser.add_argument("--vad-end-silence-ms", type=int, default=600)
    parser.add_argument("--vad-pre-roll-ms", type=int, default=200)
    parser.add_argument("--vad-minimum-speech-ms", type=int, default=200)
    parser.add_argument("--vad-maximum-utterance-ms", type=int, default=15000)
    return parser



def _load_ardy(args: argparse.Namespace) -> ArdyHeadlessRuntime:
    result: list[ArdyHeadlessRuntime] = []
    failures: list[BaseException] = []

    def load() -> None:
        try:
            result.append(
                ArdyHeadlessRuntime.load(
                    model_name=args.model,
                    device=args.device,
                    checkpoints_dir=args.checkpoints_dir,
                    text_encoder_mode=args.text_encoder_mode,
                    history_limit_frames=args.history_frames,
                    postprocess=not args.no_postprocess,
                    num_denoising_steps=args.diffusion_steps,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=load, name="ardy-model-loader", daemon=True)
    print("startup 1/4: loading resident ARDY model", flush=True)
    started = time.monotonic()
    worker.start()
    while worker.is_alive():
        worker.join(timeout=5.0)
        if worker.is_alive():
            print(
                f"heartbeat: ardy_loading elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if failures:
        raise failures[0]
    if not result:
        raise RuntimeError("ARDY loader returned no runtime")
    print(
        f"startup 2/4: ARDY ready fps={result[0].fps:g} "
        f"horizon={result[0].horizon_frames} "
        f"history={result[0].history_limit_frames}",
        flush=True,
    )
    return result[0]


def _jsonable(value: object) -> object:
    return to_jsonable(value)


def load_motion_assets(args, profile):
    if not 0 < args.transition_seconds <= 5:
        raise ValueError("transition duration must be in (0, 5]")
    avatar = profile.data["avatar"]
    return MotionAssets.load(args.idle_pose, profile.data["behaviors"], base_dir=profile.path.parent,
                             face_channels=avatar["face_channels"],
                             avatar_signature=avatar["calibration"]["signature"])


def create_motion_director(args, runtime, output, mapper_factory, assets):
    return MotionDirector(
        runtime=runtime, output=output, mapper_factory=mapper_factory,
        motion_assets=assets, idle_prompt=CALIBRATED_IDLE,
        session_factory=partial(AnchoredMotionSession, assets=assets,
                                transition_seconds=args.transition_seconds),
        replan_threshold_frames=args.replan_threshold_frames,
        pose_ttl_ms=args.pose_ttl_ms,
    )


def route_audio(message, *, pipeline, manual_actions_only):
    if not manual_actions_only:
        pipeline.accept_audio(message)


def run(args: argparse.Namespace) -> None:
    if args.heartbeat_interval <= 0:
        raise ValueError("heartbeat interval must be positive")
    profile = apply_runtime_profile(args)
    if profile.data["runtime"].get("text_encoder_mode") == "local":
        from vrc_ardy_agent.runner.runtime_setup.verify import runtime_environment
        os.environ.update(runtime_environment(profile.data["runtime"]))
        args.text_encoder_mode = "local"
    assets = load_motion_assets(args, profile)
    avatar_rig = get_avatar_profile(args.avatar_profile)
    from vrc_ardy_agent.avatar_config import signature
    if signature(profile.data["avatar"], profile.path.parent) != profile.data["avatar"]["calibration"]["signature"]:
        raise ValueError("avatar assets changed during startup")
    brain = build_provider(profile, "llm")
    stt = build_provider(profile, "stt")
    tts = build_provider(profile, "tts")
    print(
        "brain: "
        f"provider={brain.provider} model={brain.model} "
        f"reasoning_effort={brain.reasoning_effort}",
        flush=True,
    )
    ardy_runtime = _load_ardy(args)
    output = MacOutputTransport()
    retargeting_monitor = RetargetingMonitor()
    lower_body_monitor = LowerBodyRetargetingMonitor()
    mapper_factory = partial(
        SixPointStreamMapper,
        avatar_profile=avatar_rig,
        retargeting_monitor=retargeting_monitor,
        lower_body_monitor=lower_body_monitor,
        hmd_base=tuple(args.hmd_base),
        scale=assets.idle.scale,
        full_stick_speed_mps=args.full_stick_speed_mps,
        locomotion_deadzone_mps=args.locomotion_deadzone_mps,
        full_turn_speed_dps=args.full_turn_speed_dps,
        turn_deadzone_dps=args.turn_deadzone_dps,
        heading_smoothing_seconds=args.heading_smoothing_seconds,
        retargeting_mode=RetargetingMode(args.retargeting_mode),
    )
    motion_director = create_motion_director(args, ardy_runtime, output, mapper_factory, assets)

    supervisor = CharacterSupervisor(
        executors={
            ActionType.SAY: SpeechExecutor(
                synthesizer=tts,
                output=output,
            ),
        },
        output_authority=output,
    )
    interaction = InteractionRuntime(
        brain=brain,
        supervisor=supervisor,
        motion_director=motion_director,
        behaviors=assets.behaviors,
        autonomy=profile.data["autonomy"] | ({"enabled": False} if args.manual_actions_only else {}),
    )
    pipeline_holder: list[AudioInteractionPipeline] = []
    input_transport = MacInputTransport(
        audio_handler=lambda message: route_audio(message, pipeline=pipeline_holder[0],
                                                  manual_actions_only=args.manual_actions_only)
    )
    interaction.configure_observation(
        input_transport.request_screenshot,
        max_observations=args.max_observations,
        timeout_seconds=args.observation_timeout_seconds,
        max_frame_age=args.observation_max_frame_age,
    )
    vad = EnergyVadSegmenter(
        rms_threshold=args.vad_rms_threshold,
        start_trigger_ms=args.vad_start_ms,
        end_silence_ms=args.vad_end_silence_ms,
        pre_roll_ms=args.vad_pre_roll_ms,
        minimum_speech_ms=args.vad_minimum_speech_ms,
        maximum_utterance_ms=args.vad_maximum_utterance_ms,
    )
    pipeline = AudioInteractionPipeline(
        vad=vad,
        stt=stt,
        runtime=interaction,
        on_input_activity=interaction.note_input_activity,
    )
    pipeline_holder.append(pipeline)
    bridge = MacBridgeServer(
        output=output,
        input_transport=input_transport,
        host=args.bridge_host,
        port=args.bridge_port,
        heartbeat_interval_seconds=args.heartbeat_interval,
        on_session_opened=motion_director.attach_output_session,
        on_session_closed=lambda session_id, reason: (
            motion_director.detach_output_session(session_id, reason=reason)
        ),
    )
    service = CompanionControlService(
        interaction,
        enable_test_controls=args.enable_test_controls,
        screenshot_provider=input_transport.request_screenshot,
        status_sources={
            "brain": brain,
            "observation": interaction.observation_service,
            "bridge": bridge,
            "input": input_transport,
            "motion": motion_director,
            "output": output,
            "pipeline": pipeline,
            "retargeting": retargeting_monitor,
            "lower_body_retargeting": lower_body_monitor,
            "interaction_mode": lambda: {"automatic_reactions_enabled": not args.manual_actions_only},
            "motion_assets": assets,
            "autonomy": interaction.autonomy_snapshot,
            "pose_pipeline": lambda: {
                "retargeting_mode": args.retargeting_mode,
            },
            "vad": vad,
        },
    )
    control = CompanionHttpRuntime(
        create_companion_server(
            (args.control_host, args.control_port),
            service,
        )
    )
    app = MacCompanionApp(
        bridge=bridge,
        control=control,
        pipeline=pipeline,
        interaction=interaction,
        supervisor=supervisor,
        motion_director=motion_director,
    )
    print("startup 3/4: components assembled", flush=True)
    app.start()
    print(
        f"startup 4/4: ready bridge={args.bridge_host}:{bridge.bound_port} "
        f"control=http://{args.control_host}:{control.snapshot().bound_port}",
        flush=True,
    )
    try:
        while True:
            time.sleep(args.heartbeat_interval)
            pipeline.tick_autonomy()
            status = service.status()
            print(
                "heartbeat: "
                + json.dumps(_jsonable(status), ensure_ascii=False, separators=(",", ":")),
                flush=True,
            )
            if not bridge.snapshot().running:
                raise RuntimeError("agent bridge server stopped")
            if not control.snapshot().running:
                raise RuntimeError("companion control server stopped")
    except KeyboardInterrupt:
        print("stopped by user", flush=True)
    finally:
        app.stop()


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
