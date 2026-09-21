"""Bridge checks operate on Windows files and audio devices, never on ARDY."""
import json
from pathlib import Path
import subprocess

from .launch import local_path, bridge_geometry


def bridge_probe(profile):
    if not profile.has_bridge: raise ValueError("this operation requires a local bridge role")
    bridge = profile.data["bridge"]
    return inspect_bridge_environment(bridge["python"], local_path(profile, bridge["project_dir"]))


def inspect_bridge_environment(python, project_dir):
    completed = subprocess.run([python, str(Path(__file__).with_name("bridge_probe.py"))],
                              cwd=project_dir, capture_output=True, text=True, timeout=15)
    if completed.returncode: raise RuntimeError(f"bridge probe failed: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def bridge_checks(profile, *, probe=False):
    from .diagnostics import check
    from ..avatar_config import rig_path
    from ..avatar_rig_profile import AvatarRigProfile
    bridge = profile.data["bridge"]
    rows = []
    rig, _ = bridge_geometry(profile)
    try:
        AvatarRigProfile.load(rig_path({"rig": rig}, profile.path.parent if profile.path else Path.cwd()))
        rows.append(check("bridge", "rig", "pass", "local avatar rig parsed"))
    except (OSError, ValueError, TypeError) as exc:
        rows.append(check("bridge", "rig", "error", str(exc)))
    if not bridge["virtual_mic_device"]:
        rows.append(check("bridge", "audio_device", "error", "virtual microphone output device is not configured"))
    if bridge.get("process_audio_helper"):
        helper = local_path(profile, bridge["process_audio_helper"])
        rows.append(check("bridge", "audio_helper", "pass" if helper.is_file() else "error", str(helper)))
    rows.append(check("bridge", "tracking", "pending", "verify SteamVR, VMT and avatar tracking on the Windows computer"))
    if not probe: return rows
    try:
        observed = bridge_probe(profile)
        ready = observed["platform"] == "Windows" and all(observed["packages"].values())
        rows.append(check("bridge", "runtime", "pass" if ready else "error", json.dumps(observed)))
        if bridge.get("audio_source", "process") == "process":
            helper = local_path(profile, bridge["process_audio_helper"]) if bridge.get("process_audio_helper") else None
            observed_helper = observed.get("process_audio_helper") or {}
            exists = helper.is_file() if helper else observed_helper.get("exists") is True
            helper_path = str(helper) if helper else observed_helper.get("path", "")
            ready = exists and (not helper_path.endswith(".dll") or bool(observed.get("dotnet")))
            rows.append(check("bridge", "process_audio", "pass" if ready else "error",
                              helper_path if ready else observed.get("helper_error") or "build the process audio helper and install its .NET runtime, or select a microphone source"))
        if observed.get("audio_error"):
            rows.append(check("bridge", "audio_devices", "error", observed["audio_error"]))
        for key, channel in (("virtual_mic_device", "max_output_channels"), ("microphone_device", "max_input_channels")):
            if key not in bridge or not bridge[key]: continue
            selected = bridge[key]
            # PortAudio resolves names by substring; duplicate names must be
            # selected by index instead of reporting an arbitrary match as OK.
            matches = [device for device in observed["audio_devices"] if device.get(channel, 0) > 0 and
                       (str(device["index"]) == selected if selected.isdecimal() else selected.casefold() in device["name"].casefold())]
            rows.append(check("bridge", key, "pass" if len(matches) == 1 else "error", f"{len(matches)} matching devices for {selected}"))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        rows.append(check("bridge", "runtime", "error", str(exc)))
    return rows
