"""Strict validation of role-owned profile sections."""
import re
from ..avatar_config import validate_avatar
from ..behavior_catalog import BehaviorSpec
from ..asset_contract import number
from ..providers.catalog import definition, validate_binding, validate_connection


def object_fields(value, allowed, required=(), *, name):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{name}: missing, unknown or invalid fields")


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,38}", value):
        raise ValueError("IDs must use lowercase letters, digits, underscores or hyphens")


def string(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()) or "\0" in value:
        raise ValueError(f"{name} must be a non-empty string")


def validate_agent(data):
    hosts = data["hosts"]
    providers = data["providers"]
    if not isinstance(providers, dict) or not 1 <= len(providers) <= 30: raise ValueError("configure 1..30 provider connections")
    for key, connection in providers.items():
        identifier(key)
        object_fields(connection, ("provider", "label", "config", "deployment"),
                      ("provider", "label", "config", "deployment"), name=key)
        string(connection["label"], "connection label")
        provider = definition(connection["provider"])
        connection["config"] = validate_connection(connection["provider"], connection["config"])
        deploy = connection["deployment"]
        object_fields(deploy, ("kind", "host", "command", "cwd", "health_url"), ("kind",), name=f"{key}.deployment")
        if deploy["kind"] not in provider["deployment_kinds"]: raise ValueError(f"{key}: invalid deployment kind")
        if deploy["kind"] == "api":
            if set(deploy) != {"kind"}: raise ValueError("API deployments do not own a host or command")
        else:
            if not isinstance(deploy.get("host"), str) or deploy["host"] not in hosts: raise ValueError(f"{key}: unknown execution host")
            if deploy["kind"] in ("managed", "embedded") and deploy["host"] != data["runner_host"]:
                raise ValueError("managed and embedded providers must execute on the runner host")
            if deploy["kind"] == "managed":
                command = deploy.get("command")
                if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
                    raise ValueError("managed provider requires a foreground command array")
                string(deploy.get("cwd"), "managed working directory")
            elif "command" in deploy or "cwd" in deploy:
                raise ValueError("only managed providers may have a command")
            if "health_url" in deploy:
                validate_connection("openai-compatible", {"base_url": deploy["health_url"]})
            if provider["protocol"] == "whisper-local" and connection["config"]["device"] == "cuda" and hosts[deploy["host"]]["os"] == "macos":
                raise ValueError("CUDA Whisper cannot execute on a macOS host")
    bindings = data["bindings"]
    object_fields(bindings, ("llm", "stt", "tts"), ("llm", "stt", "tts"), name="bindings")
    for capability, binding in bindings.items():
        if not isinstance(binding, dict) or not isinstance(binding.get("instance"), str) or binding["instance"] not in providers:
            raise ValueError(f"{capability}: unknown provider instance")
        instance = binding["instance"]
        bindings[capability] = {"instance": instance, **validate_binding(providers[instance]["provider"], capability,
            {k: v for k, v in binding.items() if k != "instance"}, config=providers[instance]["config"])}
    runtime = data["runtime"]
    required = ("python", "project_dir", "device")
    optional = ("model", "checkpoints_dir", "hf_cache_dir", "text_encoder_mode", "installation_root")
    object_fields(runtime, required + optional, required, name="runtime")
    for key in required + ("model", "checkpoints_dir", "hf_cache_dir", "installation_root"):
        if key in runtime: string(runtime[key], key, empty=key == "checkpoints_dir")
    if runtime.get("text_encoder_mode", "local") != "local":
        raise ValueError("managed runtime text_encoder_mode must be local")
    if runtime["device"] not in ("auto", "cpu", "mps", "cuda"): raise ValueError("invalid ARDY device")
    if runtime["device"] == "mps" and hosts[data["runner_host"]]["os"] != "macos":
        raise ValueError("MPS requires a macOS execution host")
    if runtime["device"] == "cuda" and hosts[data["runner_host"]]["os"] == "macos":
        raise ValueError("CUDA requires a Linux or Windows execution host")
    validate_avatar(data["avatar"])
    if not isinstance(data["behaviors"], dict) or len(data["behaviors"]) > 100:
        raise ValueError("behaviors must contain at most 100 definitions")
    for key, raw in data["behaviors"].items(): BehaviorSpec.parse(key, raw)
    autonomy = data["autonomy"]
    object_fields(autonomy, ("enabled", "interval_seconds", "behaviors"),
                  ("enabled", "interval_seconds", "behaviors"), name="autonomy")
    if type(autonomy["enabled"]) is not bool: raise ValueError("autonomy.enabled must be boolean")
    number(autonomy["interval_seconds"], 1, 86400)
    pool = autonomy["behaviors"]
    if not isinstance(pool, list) or any(not isinstance(key, str) or key not in data["behaviors"] for key in pool):
        raise ValueError("autonomy must reference registered behaviors")
    if len(set(pool)) != len(pool): raise ValueError("autonomy behaviors must be distinct")
    if autonomy["enabled"] and not pool: raise ValueError("enabled autonomy requires behaviors")


def validate_bridge(bridge, *, standalone=False, legacy=False):
    required = ("python", "project_dir", "virtual_mic_device")
    geometry = ("avatar_rig", "hmd_base") if standalone else ()
    allowed = required + geometry + ("body_output", "audio_source", "vrchat_pid", "vrchat_user_name", "microphone_device", "process_audio_helper")
    if legacy: allowed += ("mode", "control_port", "avatar_rig")
    object_fields(bridge, allowed, required + geometry, name="bridge")
    for key in required: string(bridge[key], key, empty=key == "virtual_mic_device")
    if bridge.get("body_output", "vrchat-osc") not in ("vrchat-osc", "vmt"): raise ValueError("invalid body output")
    if bridge.get("audio_source", "process") not in ("process", "microphone"): raise ValueError("invalid audio source")
    for key in ("vrchat_user_name", "microphone_device", "process_audio_helper", "avatar_rig"):
        if key in bridge: string(bridge[key], key, empty=key == "avatar_rig")
    if standalone:
        origin = bridge["hmd_base"]
        if not isinstance(origin, list) or len(origin) != 3: raise ValueError("bridge.hmd_base must be [x, y, z]")
        for value in origin: number(value, -100, 100)
        if origin[1] <= 0: raise ValueError("HMD height must be positive")
    if bridge.get("audio_source") == "microphone" and not bridge.get("microphone_device"):
        raise ValueError("microphone audio source requires microphone_device")
    if "vrchat_user_name" in bridge and "vrchat_pid" in bridge: raise ValueError("select one VRChat account or PID")
    if "vrchat_pid" in bridge and (type(bridge["vrchat_pid"]) is not int or bridge["vrchat_pid"] <= 0):
        raise ValueError("invalid VRChat PID")
