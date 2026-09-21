"""Configuration checks and explicitly requested provider probes; no game actions."""
import json
import os
import platform
from pathlib import Path
import subprocess
import time

from ..providers.catalog import definition
from .launch import local_path, runner_config, endpoint
from .core import Runner


def current_os():
    return {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system(), platform.system().lower())


def check(service, name, result, detail):
    return {"service": service, "check": name, "result": result, "detail": detail}


def agent_checks(profile):
    rows = []
    from .avatar_setup import inspect_assets
    assets = inspect_assets(profile)
    rows.append(check("avatar", "calibration", "pass" if assets["avatar"]["ready"] else "error", assets["avatar"]["detail"]))
    rows.append(check("behaviors", "catalog", "pass", f'{len(assets["available"])} available behaviors'))
    for key, spec in assets["available"].items():
        rows.append(check("behavior:" + key, "assets", "pass", spec.source))
    for key, error in assets["unavailable"].items():
        rows.append(check("behavior:" + key, "assets", "warning", error))
    policy = profile.data["autonomy"]
    available_pool = [key for key in policy["behaviors"] if key in assets["available"]]
    rows.append(check("autonomy", "policy", "warning" if policy["enabled"] and not available_pool else "pass",
                      "disabled" if not policy["enabled"] else f"available pool: {available_pool}"))
    for key, connection in profile.data["providers"].items():
        from ..providers.credentials import resolve_key, CredentialError
        try:
            resolve_key(connection["provider"], connection["config"])
            rows.append(check(key, "credentials", "pass", connection["config"].get("credential_source", "environment")))
        except CredentialError as exc:
            rows.append(check(key, "credentials", "error", str(exc)))
        rows.append(check(key, "inference", "pending", "use providers test with an explicit text or audio sample"))
    return rows


def agent_runtime_checks(profile):
    rows = []
    runtime = profile.data["runtime"]
    try:
        observed_runtime = runtime_probe(profile)
        ready = observed_runtime["ardy_importable"] and all(observed_runtime["packages"].get(name) for name in ("torch", "numpy", "websockets"))
        rows.append(check("agent", "runtime", "pass" if ready else "error", json.dumps(observed_runtime)))
        device = runtime["device"]
        if device != "auto" and not observed_runtime["devices"].get(device):
            rows.append(check("agent", "device", "error", f"{device} is unavailable in the configured Python environment"))
        for key, connection in profile.data["providers"].items():
            if connection["provider"] == "whisper-local" and not observed_runtime["packages"].get("faster-whisper"):
                rows.append(check(key, "runtime", "error", "install the local-stt extra in the configured runtime Python"))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        rows.append(check("agent", "runtime", "error", str(exc)))
    return rows


def doctor(profile, *, offline=False, state_dir=None, progress=None):
    rows = [check("profile", "schema", "pass", f"version 3; role={profile.data['role']}")]
    expected_os = profile.data["hosts"][profile.local_host]["os"]
    local_matches = current_os() == expected_os
    rows.append(check("runner", "platform", "pass" if local_matches else "error",
                      f"observed={current_os()} configured={expected_os}"))
    config = runner_config(profile)
    for service in config.services:
        if service.command:
            valid = local_matches and service.executable() is not None
            rows.append(check(service.id, "command", "pass" if valid else "error", "executable and working directory"))
        for path in service.required_files:
            rows.append(check(service.id, "file", "pass" if path.is_file() else "error", str(path)))
    if profile.has_agent: rows.extend(agent_checks(profile))
    if profile.has_bridge:
        from .bridge_diagnostics import bridge_checks
        rows.extend(bridge_checks(profile, probe=not offline and local_matches))
    if offline:
        rows.append(check("connections", "network", "pending", "offline: no services contacted"))
        return rows
    if profile.has_agent and local_matches: rows.extend(agent_runtime_checks(profile))
    runner = Runner(config, state_dir)
    for status in runner.status(progress=progress):
        if runner.service(status["id"]).health_url:
            rows.append(check(status["id"], "endpoint", "pass" if status["health"]["ok"] else "error", status["health"]["detail"]))
    # Read the local peer's session state. A healthy HTTP listener alone does
    # not establish that the agent and the Windows bridge are communicating.
    from urllib.request import urlopen
    service = "agent" if profile.has_agent else "bridge"
    try:
        with urlopen(endpoint(profile, service) + "/status", timeout=2) as response:
            status = json.loads(response.read(1024*1024))
        if service == "agent":
            bridge_state = status.get("components", {}).get("bridge", {})
        else:
            plane = status.get("device_plane", {})
            bridge_state = plane.get("components", {}).get("bridge", {})
            rows.append(check("bridge", "tracking", "pass" if plane.get("registration_verified") is True else "pending",
                              "tracker registration verified" if plane.get("registration_verified") is True else "tracker registration requires VR verification"))
            sensor = status.get("sensor_supervisor", {})
            rows.append(check("bridge", "vrchat", "pass" if sensor.get("state") == "RUNNING" else "error", json.dumps(sensor)))
        rows.append(check("bridge", "session", "pass" if bridge_state.get("connected") is True else "error", json.dumps(bridge_state)))
    except (OSError, ValueError, AttributeError) as exc:
        rows.append(check("bridge", "session", "error", f"session state unavailable: {type(exc).__name__}"))
    return rows


def runtime_probe(profile):
    """Run on the configured Python, not on the UI's lightweight interpreter."""
    profile.require_agent()
    probe = Path(__file__).with_name("host_probe.py")
    completed = subprocess.run([profile.data["runtime"]["python"], str(probe)],
        cwd=local_path(profile, profile.data["runtime"]["project_dir"]), capture_output=True, text=True, timeout=45)
    if completed.returncode: raise RuntimeError(f"runtime probe failed: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def provider_probe(profile, capability, *, text=None, audio_file=None):
    from ..providers.factory import build_provider
    from ..providers.adapters import validate_wav
    from ..interaction_runtime import BrainRequest, BodyActivity, SpeechActivity
    from io import BytesIO
    import wave
    adapter = build_provider(profile, capability)
    started = time.monotonic()
    if capability == "llm":
        if not text: raise ValueError("LLM probe requires text")
        from .avatar_setup import inspect_assets
        behaviors = inspect_assets(profile)["available"]
        result = adapter.propose(BrainRequest("provider-probe", 1, text, None, BodyActivity.IDLE, SpeechActivity.IDLE, (), None, available_motions=tuple(spec.describe() for spec in behaviors.values())))
        # Use the production decoder; a JSON object alone is not a usable plan.
        from ..action_contracts import validate_action_bundle
        if result.get("type") == "observe_scene":
            if not isinstance(result.get("query"), str) or not result["query"].strip():
                raise ValueError("observation decision requires a non-empty query")
        else:
            validate_action_bundle({"actions": result.get("actions")}, behaviors=behaviors)
    elif capability == "stt":
        if not audio_file: raise ValueError("STT probe requires a 16 kHz mono PCM WAV file")
        if Path(audio_file).stat().st_size > 32*1024*1024:
            raise ValueError("STT probe audio exceeds 32 MiB")
        raw = Path(audio_file).read_bytes()
        validate_wav(raw)
        with wave.open(BytesIO(raw)) as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1: raise ValueError("STT sample must be 16 kHz mono")
            pcm = wav.readframes(wav.getnframes())
        result = {"transcript": adapter.transcribe_pcm(pcm)}
    else:
        if not text: raise ValueError("TTS probe requires text")
        raw = validate_wav(adapter.synthesize(text))
        with wave.open(BytesIO(raw)) as wav:
            result = {"wav_bytes": len(raw), "sample_rate": wav.getframerate(), "duration_seconds": wav.getnframes()/wav.getframerate()}
    return {"capability": capability, "result": result, "elapsed_seconds": time.monotonic()-started,
            "executed_in_game": False}


def provider_probe_on_runtime(profile, payload):
    if payload.get("capability") not in ("llm", "stt", "tts"): raise ValueError("unknown probe capability")
    from ..providers.credentials import runtime_environment
    source = Path(__file__).with_name("probe_worker.py")
    completed = subprocess.run([profile.data["runtime"]["python"], str(source), str(profile.path)],
        cwd=local_path(profile, profile.data["runtime"]["project_dir"]), input=json.dumps(payload),
        capture_output=True, text=True, timeout=630, env=runtime_environment(profile, os.environ))
    if completed.returncode: raise RuntimeError(f"provider probe failed: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def list_provider_items(profile, instance, kind, cursor=None):
    from ..providers.discovery import discover_items
    connection = profile.data["providers"].get(instance)
    if connection is None: raise ValueError("unknown provider instance")
    return discover_items(connection["provider"], connection["config"], kind=kind, cursor=cursor)
