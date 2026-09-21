#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.companion_api import create_companion_server
from vrc_ardy_agent.companion_http_runtime import CompanionHttpRuntime
from vrc_ardy_agent.device_gateway import DeviceGateway
from vrc_ardy_agent.calibration_session import CalibrationSession
from vrc_ardy_agent.calibration_target import CalibrationTargetSession
from vrc_ardy_agent.live_sink import BodyPoseTransport, SixPointUdpSink
from vrc_ardy_agent.status_serialization import to_jsonable
from vrc_ardy_agent.tracker_device_controller import TrackerDeviceController
from vrc_ardy_agent.tracking_rig import (
    BodyTrackingMode,
    HandSelection,
    NeutralPoseConfig,
    build_neutral_tracking_frame,
    get_avatar_profile,
)
from vrc_ardy_agent.windows_bridge_client import WindowsBridgeClient
from vrc_ardy_agent.windows_capture import (
    WindowInfo,
    WindowsGraphicsCaptureSource,
    list_windows,
    wait_for_window,
)
from vrc_ardy_agent.windows_companion_api import WindowsCompanionControlService
from vrc_ardy_agent.windows_companion_runtime import WindowsCompanionRuntime
from vrc_ardy_agent.windows_companion_supervisor import WindowsCompanionSupervisor
from vrc_ardy_agent.windows_device_plane import WindowsDevicePlane
from vrc_ardy_agent.windows_device_sink import WindowsDeviceSink
from vrc_ardy_agent.windows_input_hub import WindowsInputHub
from vrc_ardy_agent.windows_microphone_audio import WasapiMicrophoneAudioSource
from vrc_ardy_agent.windows_output_controller import WindowsOutputController
from vrc_ardy_agent.windows_process_audio import NativeProcessAudioSource
from vrc_ardy_agent.windows_screenshot import (
    VrchatObserverScreenshotProvider,
    WindowsJpegScreenshotProvider,
)
from vrc_ardy_agent.windows_sensor_session import VrchatSensorSession
from vrc_ardy_agent.windows_vrchat_identity import VrchatIdentityResolver
from vrc_ardy_agent.windows_wave_player import WaveAudioPlayer


def positive_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def audio_device(value: str) -> str | int:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("audio device must not be empty")
    try:
        index = int(normalized, 10)
    except ValueError:
        return normalized
    if index < 0:
        raise argparse.ArgumentTypeError("audio device index must be non-negative")
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Windows sensory and VRChat device bridge."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list-windows")
    list_parser.add_argument("--title")
    commands.add_parser("list-audio-devices")

    run = commands.add_parser("run")
    selector = run.add_mutually_exclusive_group(required=True)
    selector.add_argument("--hwnd", type=positive_int)
    selector.add_argument("--window-title")
    selector.add_argument("--window-process-id", type=positive_int)
    selector.add_argument("--vrchat-user-name")
    selector.add_argument("--vrchat-user-id")
    run.add_argument("--window-wait-timeout", type=float, default=None)
    run.add_argument("--vrchat-log-dir")
    run.add_argument("--vrchat-window-title", default="VRChat")
    run.add_argument("--capture-interval-ms", type=int, default=50)
    run.add_argument("--screenshot-timeout", type=float, default=2.0)
    run.add_argument("--screenshot-quality", type=int, default=85)
    run.add_argument("--enable-test-controls", action="store_true")
    run.add_argument("--calibration-only", action="store_true",
                     help="Attach VRChat identity and pose output without audio or capture")
    observer_selector = run.add_mutually_exclusive_group()
    observer_selector.add_argument("--debug-observer-vrchat-user-name")
    observer_selector.add_argument("--debug-observer-vrchat-user-id")
    # Existing bridge launchers consume --mac-host; both flags select one host.
    run.add_argument("--agent-host", "--mac-host", dest="agent_host", required=True)
    run.add_argument("--bridge-port", type=int, default=8766)
    run.add_argument("--virtual-mic-device", required=True, type=audio_device)
    run.add_argument(
        "--audio-source",
        choices=("process", "microphone"),
        default="process",
    )
    run.add_argument("--microphone-device", type=audio_device)
    run.add_argument("--process-audio-helper")
    run.add_argument("--vrchat-host", default="127.0.0.1")
    run.add_argument("--opentrack-port", type=int, default=4242)
    run.add_argument("--vmt-port", type=int, default=39570)
    run.add_argument("--vrchat-port", type=int, default=9000)
    run.add_argument(
        "--body-output",
        choices=tuple(transport.value for transport in BodyPoseTransport),
        default=BodyPoseTransport.VRCHAT_OSC.value,
    )
    run.add_argument("--no-locomotion", action="store_true")
    run.add_argument("--dry-run-pose", action="store_true")
    run.add_argument(
        "--avatar-profile",
        required=True,
        help="Bundled rig name or a Windows path to an exported avatar rig JSON",
    )
    run.add_argument(
        "--hmd-base",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 1.0, 0.0),
    )
    run.add_argument("--pose-rate-hz", type=float, default=20.0)
    run.add_argument("--heartbeat-interval", type=float, default=2.0)
    run.add_argument("--control-host", default="127.0.0.1")
    run.add_argument("--control-port", type=int, default=8767)
    return parser


def _resolve_window(args: argparse.Namespace) -> WindowInfo:
    if args.hwnd is not None:
        match = next(
            (window for window in list_windows() if window.hwnd == args.hwnd),
            None,
        )
        if match is None:
            raise RuntimeError(f"HWND 0x{args.hwnd:X} is not a capturable window")
        return match
    return wait_for_window(
        title=args.window_title,
        process_id=args.window_process_id,
        timeout_seconds=args.window_wait_timeout,
        heartbeat=lambda elapsed: print(
            f"heartbeat: waiting_for_vrchat_window elapsed={elapsed:.1f}s",
            flush=True,
        ),
    )


def _list_audio_devices() -> None:
    try:
        import sounddevice
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is missing; install requirements-windows.txt"
        ) from exc
    for index, device in enumerate(sounddevice.query_devices()):
        print(
            f"index={index} inputs={device['max_input_channels']} "
            f"outputs={device['max_output_channels']} "
            f"name={device['name']}"
        )


def _jsonable(value: object) -> object:
    return to_jsonable(value)


def _build_device_plane(
    args: argparse.Namespace,
) -> tuple[WindowsDevicePlane, WindowsInputHub]:
    neutral = build_neutral_tracking_frame(
        get_avatar_profile(args.avatar_profile),
        NeutralPoseConfig(
            hmd_base=tuple(args.hmd_base),
            fps=args.pose_rate_hz,
            hands=HandSelection.NONE,
            body_tracking=BodyTrackingMode.PROFILE,
        ),
    )
    raw_pose_sink = SixPointUdpSink(
        host=args.vrchat_host,
        opentrack_port=args.opentrack_port,
        vmt_port=args.vmt_port,
        vrchat_port=args.vrchat_port,
        body_pose_transport=BodyPoseTransport(args.body_output),
        osc_head_position=tuple(args.hmd_base),
        send_locomotion=not args.no_locomotion,
        send_face=True,
        lock_head_rotation=True,
        dry_run=args.dry_run_pose,
    )
    tracker = TrackerDeviceController(
        sink=raw_pose_sink,
        safe_frame=neutral.frame,
        rate_hz=args.pose_rate_hz,
    )
    input_hub = WindowsInputHub()
    wave_player = WaveAudioPlayer(device=args.virtual_mic_device)
    devices = WindowsDeviceSink(
        pose_sink=tracker,
        wave_player=wave_player,
    )
    gateway = DeviceGateway(sink=devices)
    calibration = CalibrationSession(gateway, tracker, hmd_base=args.hmd_base)
    bridge = WindowsBridgeClient(
        url=f"ws://{('[' + args.agent_host + ']') if ':' in args.agent_host else args.agent_host}:{args.bridge_port}/companion",
        controller=WindowsOutputController(gateway=gateway),
        completion_target=devices,
        input_transport=input_hub,
        client_instance_id=f"windows-device-plane-{uuid.uuid4().hex}",
    )
    return (
        WindowsDevicePlane(
            tracker_controller=tracker,
            calibration=calibration,
            bridge=bridge,
            devices=devices,
            status_sources={
                "tracker": tracker,
                "bridge": bridge,
                "input": input_hub,
                "gateway": gateway,
                "wave": lambda: {"last_error": wave_player.last_error},
            },
        ),
        input_hub,
    )


def _build_sensor_session(
    args: argparse.Namespace,
    *,
    window: WindowInfo,
    epoch: int,
    input_hub: WindowsInputHub,
) -> VrchatSensorSession:
    sensor_id = f"vrchat-{window.process_id}-{epoch}"
    if args.calibration_only:
        return CalibrationTargetSession(sensor_id)
    capture = WindowsGraphicsCaptureSource(
        hwnd=window.hwnd,
        minimum_update_interval_ms=args.capture_interval_ms,
    )
    screenshot = WindowsJpegScreenshotProvider(
        source=capture,
        jpeg_quality=args.screenshot_quality,
        timeout_seconds=args.screenshot_timeout,
    )

    def publish_audio(pcm: bytes, captured_at: int) -> bool:
        return input_hub.publish_audio(
            sensor_id,
            pcm,
            captured_monotonic_ns=captured_at,
        )

    if args.audio_source == "microphone":
        if args.microphone_device is None:
            raise ValueError(
                "--microphone-device is required when --audio-source=microphone"
            )
        if args.process_audio_helper is not None:
            raise ValueError("--process-audio-helper requires --audio-source=process")
        audio = WasapiMicrophoneAudioSource(
            device=args.microphone_device,
            frame_handler=publish_audio,
        )
    else:
        if args.microphone_device is not None:
            raise ValueError("--microphone-device requires --audio-source=microphone")
        audio = NativeProcessAudioSource(
            pid=window.process_id,
            frame_handler=publish_audio,
            helper_path=args.process_audio_helper,
        )
    return VrchatSensorSession(
        sensor_id=sensor_id,
        capture=capture,
        screenshot_provider=screenshot,
        audio=audio,
        input_hub=input_hub,
        status_sources={
            "audio": audio,
            "screenshot": screenshot,
            "capture": lambda: {"frames_captured": capture.frames_captured},
        },
    )


def _build_debug_observer_screenshot(
    args: argparse.Namespace,
) -> VrchatObserverScreenshotProvider | None:
    if (
        args.debug_observer_vrchat_user_name is None
        and args.debug_observer_vrchat_user_id is None
    ):
        return None
    resolver = VrchatIdentityResolver(
        user_name=args.debug_observer_vrchat_user_name,
        user_id=args.debug_observer_vrchat_user_id,
        log_dir=args.vrchat_log_dir,
        window_title=args.vrchat_window_title,
    )
    return VrchatObserverScreenshotProvider(
        resolve_target=resolver.resolve,
        resolution_diagnostics=resolver.diagnostics,
        source_factory=lambda hwnd: WindowsGraphicsCaptureSource(
            hwnd=hwnd,
            minimum_update_interval_ms=args.capture_interval_ms,
        ),
        jpeg_quality=args.screenshot_quality,
        timeout_seconds=args.screenshot_timeout,
    )


def _run_once(args: argparse.Namespace) -> None:
    window = _resolve_window(args)
    device_plane, input_hub = _build_device_plane(args)
    sensors = _build_sensor_session(
        args,
        window=window,
        epoch=1,
        input_hub=input_hub,
    )
    runtime = WindowsCompanionRuntime(
        device_plane=device_plane,
        sensor_supervisor=sensors,
    )
    debug_observer_screenshot = _build_debug_observer_screenshot(args)
    shutdown_requested = threading.Event()
    control = CompanionHttpRuntime(
        create_companion_server(
            (args.control_host, args.control_port),
            WindowsCompanionControlService(
                runtime,
                shutdown_requested=shutdown_requested,
                enable_test_controls=args.enable_test_controls,
                screenshot_provider=debug_observer_screenshot,
                calibration=device_plane.calibration,
            ),
        )
    )
    print(
        f"VRChat window ready: hwnd=0x{window.hwnd:X} pid={window.process_id} "
        f"size={window.width}x{window.height}; bridge={args.agent_host}:{args.bridge_port}",
        flush=True,
    )
    runtime.start()
    try:
        control.start()
        print(
            f"control=http://{args.control_host}:{control.snapshot().bound_port}",
            flush=True,
        )
        while not shutdown_requested.wait(args.heartbeat_interval):
            snapshot = runtime.snapshot()
            print(
                "heartbeat: "
                + json.dumps(
                    _jsonable(snapshot), ensure_ascii=False, separators=(",", ":")
                ),
                flush=True,
            )
            if not snapshot.running:
                raise RuntimeError("Windows companion stopped")
    except KeyboardInterrupt:
        print("stopped by user", flush=True)
    finally:
        control.stop()
        runtime.stop()


def _run_supervised(args: argparse.Namespace) -> None:
    if args.window_wait_timeout is not None:
        raise ValueError(
            "--window-wait-timeout cannot be used with a VRChat account selector"
        )
    resolver = VrchatIdentityResolver(
        user_name=args.vrchat_user_name,
        user_id=args.vrchat_user_id,
        log_dir=args.vrchat_log_dir,
        window_title=args.vrchat_window_title,
    )
    device_plane, input_hub = _build_device_plane(args)
    supervisor = WindowsCompanionSupervisor(
        resolve_target=resolver.resolve,
        target_is_current=resolver.is_current,
        runtime_factory=lambda target, epoch: _build_sensor_session(
            args,
            window=target.window,
            epoch=epoch,
            input_hub=input_hub,
        ),
    )
    runtime = WindowsCompanionRuntime(
        device_plane=device_plane,
        sensor_supervisor=supervisor,
    )
    debug_observer_screenshot = _build_debug_observer_screenshot(args)
    shutdown_requested = threading.Event()
    control = CompanionHttpRuntime(
        create_companion_server(
            (args.control_host, args.control_port),
            WindowsCompanionControlService(
                runtime,
                shutdown_requested=shutdown_requested,
                enable_test_controls=args.enable_test_controls,
                screenshot_provider=debug_observer_screenshot,
                calibration=device_plane.calibration,
            ),
        )
    )
    runtime.start()
    try:
        control.start()
        print(
            f"control=http://{args.control_host}:{control.snapshot().bound_port}",
            flush=True,
        )
        while not shutdown_requested.wait(args.heartbeat_interval):
            snapshot = runtime.snapshot()
            print(
                "heartbeat: "
                + json.dumps(
                    _jsonable(snapshot),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if not snapshot.running:
                raise RuntimeError("Windows companion stopped")
            if not supervisor.snapshot().running:
                raise RuntimeError("Windows companion supervisor stopped")
    except KeyboardInterrupt:
        print("stopped by user", flush=True)
    finally:
        control.stop()
        runtime.stop()


def _run(args: argparse.Namespace) -> None:
    if args.heartbeat_interval <= 0:
        raise ValueError("heartbeat interval must be positive")
    has_debug_observer = (
        args.debug_observer_vrchat_user_name is not None
        or args.debug_observer_vrchat_user_id is not None
    )
    if has_debug_observer and not args.enable_test_controls:
        raise ValueError("debug observer selection requires --enable-test-controls")
    if args.vrchat_user_name is not None or args.vrchat_user_id is not None:
        _run_supervised(args)
        return
    if args.vrchat_log_dir is not None and not has_debug_observer:
        raise ValueError(
            "--vrchat-log-dir requires --vrchat-user-name or --vrchat-user-id"
        )
    _run_once(args)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "list-windows":
        for window in list_windows(title_filter=args.title):
            state = "minimized" if window.minimized else "visible"
            print(
                f"hwnd=0x{window.hwnd:X} pid={window.process_id} "
                f"size={window.width}x{window.height} state={state} title={window.title}"
            )
        return
    if args.command == "list-audio-devices":
        _list_audio_devices()
        return
    _run(args)


if __name__ == "__main__":
    main()
