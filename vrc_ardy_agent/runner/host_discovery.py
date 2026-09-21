"""Read local Tailscale membership. Discovery never installs or reconfigures it."""
import ipaddress
import json
import os
from pathlib import Path
import shutil
import subprocess


def tailscale_executable(os_name=None):
    if os_name is None:
        from .diagnostics import current_os
        os_name = current_os()
    if os_name == "macos":
        bundled = Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale")
        if bundled.is_file(): return str(bundled)
    found = shutil.which("tailscale.exe" if os_name == "windows" else "tailscale")
    if found: return found
    if os_name == "windows":
        for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            if os.environ.get(variable):
                candidate = Path(os.environ[variable]) / "Tailscale" / "tailscale.exe"
                if candidate.is_file(): return str(candidate)
    return None


def peer_info(raw):
    if not isinstance(raw, dict): raise ValueError("invalid Tailscale peer object")
    ips = raw.get("TailscaleIPs") or []
    if not isinstance(ips, list): raise ValueError("invalid Tailscale IP list")
    parsed = [ipaddress.ip_address(ip) for ip in ips]
    if not parsed: return None
    parsed.sort(key=lambda ip: ip.version)
    os_name = str(raw.get("OS", "")).lower()
    return {"id": str(raw.get("ID") or parsed[0]), "name": str(raw.get("HostName") or raw.get("DNSName") or parsed[0]),
            "os": {"darwin": "macos"}.get(os_name, os_name), "address": str(parsed[0]),
            "addresses": list(map(str, parsed)), "online": raw.get("Online") is True}


def discover_hosts():
    executable = tailscale_executable()
    result = {"state": "unavailable", "detail": "Tailscale CLI was not found", "self": None, "peers": []}
    if not executable: return result
    try:
        completed = subprocess.run([executable, "status", "--json"], capture_output=True, text=True, timeout=8)
        if completed.returncode:
            raise ValueError(f"Tailscale status exited {completed.returncode}: {completed.stderr.strip()[-1000:]}")
        raw = json.loads(completed.stdout)
        if not isinstance(raw, dict): raise ValueError("invalid Tailscale status object")
        state = raw.get("BackendState")
        if state != "Running":
            return result | {"state": {"NeedsLogin": "needs-login", "Stopped": "stopped"}.get(state, "error"),
                             "detail": f"Tailscale BackendState={state}"}
        peers = raw.get("Peer") or {}
        if not isinstance(peers, dict): raise ValueError("invalid Tailscale peer collection")
        own = peer_info(raw.get("Self", {}))
        if own is None: raise ValueError("Tailscale has no address for this computer")
        candidates = [peer for value in peers.values() if (peer := peer_info(value)) is not None and peer["id"] != own["id"]]
        candidates.sort(key=lambda peer: (not peer["online"], peer["name"].casefold(), peer["id"]))
        return result | {"state": "ready", "detail": "Tailscale membership observed; application connectivity is not yet verified",
                         "self": own, "peers": candidates}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return result | {"state": "error", "detail": str(exc)}
