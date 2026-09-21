"""Compile a device role into local ownership and remote observations."""
from pathlib import Path, PureWindowsPath

from .config import RunnerConfig, Service
from .profile import Profile


def address(host):
    return f"[{host}]" if ":" in host else host


def local_path(profile, value):
    base = profile.path.parent if profile.path else Path.cwd()
    return (base / Path(value).expanduser()).resolve()


def endpoint(profile, service):
    host = profile.data["runner_host" if service == "agent" else "game_host"]
    return f"http://{address(profile.host_address(host))}:{profile.data['connection'][service + '_port']}"


def bridge_geometry(profile):
    if profile.data["role"] == "combined":
        return profile.data["avatar"]["rig"], profile.data["avatar"]["hmd_base"]
    bridge = profile.data["bridge"]
    return bridge["avatar_rig"], bridge["hmd_base"]


def bridge_command(profile, *, allow_incomplete=False):
    if not profile.has_bridge: raise ValueError("run bridge command on the Windows bridge computer; use connection info for its connection settings")
    bridge, ports = profile.data["bridge"], profile.data["connection"]
    rig, origin = bridge_geometry(profile)
    if not allow_incomplete and (not rig or not bridge["virtual_mic_device"]):
        raise ValueError("configure the local bridge avatar rig and virtual microphone before exporting its command")
    bundled = Path(__file__).parents[1] / "rig_profiles" / f"{rig}.avatar-rig.json"
    if rig and not bundled.is_file(): rig = str(local_path(profile, rig))
    project = bridge["project_dir"]
    if not PureWindowsPath(project).is_absolute(): project = str(local_path(profile, project))
    script = str(PureWindowsPath(project) / "scripts" / "run_windows_companion.py").replace("\\", "/")
    command = [bridge["python"], script, "run", "--agent-host", profile.bridge_address,
               "--bridge-port", str(ports["stream_port"]), "--control-port", str(ports["bridge_port"]),
               "--control-host", profile.host_address(profile.data["game_host"]),
               "--audio-source", bridge.get("audio_source", "process"),
               "--body-output", bridge.get("body_output", "vrchat-osc"),
               "--hmd-base", *map(str, origin)]
    # Incomplete setup remains inspectable by status/doctor. Required CLI
    # arguments stay absent; preflight reports their absence before a start.
    if rig: command += ["--avatar-profile", rig]
    if bridge["virtual_mic_device"]: command += ["--virtual-mic-device", bridge["virtual_mic_device"]]
    if bridge.get("vrchat_pid"):
        command += ["--window-process-id", str(bridge["vrchat_pid"])]
    elif bridge.get("vrchat_user_name"):
        command += ["--vrchat-user-name", bridge["vrchat_user_name"]]
    else:
        command += ["--window-title", "VRChat"]
    for key in ("microphone_device", "process_audio_helper"):
        if bridge.get(key): command += ["--" + key.replace("_", "-"), bridge[key]]
    return command


def connection_info(profile):
    data = profile.data
    return {"role": data["role"], "local_host": profile.local_host,
            "agent": {"os": data["hosts"][data["runner_host"]]["os"], "address": profile.bridge_address},
            "game": {"os": "windows", "address": profile.host_address(data["game_host"])},
            "ports": data["connection"]}


def runner_config(profile):
    data = profile.data
    services = []
    # Both peers are observable, but only this profile's role can own a process.
    # No remote command, path, or Python environment is needed for observation.
    if profile.has_agent:
        runtime = data["runtime"]
        root = local_path(profile, runtime["project_dir"])
        assets = (local_path(profile, data["avatar"]["base_pose"]),) if data["avatar"]["base_pose"] else ()
        command = (runtime["python"], "-u", str(root / "scripts" / "run_companion.py"), "--profile", str(profile.path),
                   "--control-host", profile.bridge_address)
        services.append(Service("agent", command=command, cwd=root,
                                env={"VRC_AGENT_CREDENTIAL_PROFILE": str(profile.path)},
                                required_files=(root / "scripts" / "run_companion.py", *assets),
                                health_url=endpoint(profile, "agent") + "/health", health_status="ok"))
    else:
        services.append(Service("agent", health_url=endpoint(profile, "agent") + "/health", health_status="ok"))
    if profile.has_bridge:
        root = local_path(profile, data["bridge"]["project_dir"])
        services.append(Service("bridge", command=tuple(bridge_command(profile, allow_incomplete=True)), cwd=root,
                                required_files=(root / "scripts" / "run_windows_companion.py",),
                                health_url=endpoint(profile, "bridge") + "/health", health_status="ok"))
    else:
        services.append(Service("bridge", health_url=endpoint(profile, "bridge") + "/health", health_status="ok"))
    for key, connection in data.get("providers", {}).items():
        deploy = connection["deployment"]
        if deploy["kind"] in ("api", "embedded"): continue
        services.append(Service("provider-" + key, command=tuple(deploy.get("command", ())),
                                cwd=local_path(profile, deploy["cwd"]) if "cwd" in deploy else local_path(profile, data["runtime"]["project_dir"]),
                                health_url=deploy.get("health_url", ""), label=connection["label"]))
    return RunnerConfig(tuple(services), profile.path)


def apply_runtime_profile(args):
    profile = Profile.load(args.profile)
    profile.require_agent()
    runtime, ports = profile.data["runtime"], profile.data["connection"]
    for key, value in {"device": runtime["device"], "bridge_port": ports["stream_port"], "control_port": ports["agent_port"]}.items():
        if getattr(args, key, None) is None: setattr(args, key, value)
    if args.checkpoints_dir is None and runtime.get("checkpoints_dir"):
        args.checkpoints_dir = local_path(profile, runtime["checkpoints_dir"])
    if args.model is None: args.model = runtime.get("model", "core")
    avatar = profile.data["avatar"]
    from ..avatar_config import inspect_avatar, rig_path
    base = profile.path.parent
    observed = inspect_avatar(avatar, base)
    if not observed["ready"]: raise ValueError(observed["detail"])
    args.avatar_profile = str(rig_path(avatar, base))
    args.idle_pose = local_path(profile, avatar["base_pose"])
    args.hmd_base = tuple(avatar["hmd_base"])
    return profile
