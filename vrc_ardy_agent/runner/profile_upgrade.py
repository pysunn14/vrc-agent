"""Read existing v2 installations and unfinished drafts without rewriting them.

The old CLI only ran on the agent host. That fact, rather than the current OS,
determines ownership when importing an existing profile. Publication remains an
explicit, revision-checked save; reading a profile never changes it on disk.
"""
from copy import deepcopy

from .profile_fields import object_fields, validate_bridge


def upgrade_profile(raw, *, draft=False):
    data = deepcopy(raw)
    if not isinstance(data, dict) or data.get("version") != 2: return data
    fields = ("version", "language", "hosts", "runner_host", "game_host", "providers", "bindings",
              "runtime", "bridge", "avatar", "behaviors", "autonomy")
    object_fields(data, fields, () if draft else fields, name="profile v2")
    same_host = data.get("runner_host") == data.get("game_host")
    bridge = data.get("bridge", {})
    runtime = data.get("runtime", {})
    if not isinstance(bridge, dict) or not isinstance(runtime, dict): raise ValueError("invalid runtime or bridge")
    if not draft:
        validate_bridge(bridge, legacy=True)
        if bridge.get("mode") not in ("external", "managed"): raise ValueError("invalid bridge mode")
        if bridge["mode"] == "managed" and not same_host:
            raise ValueError("a remote Windows bridge must be managed on its own host")
    data.update(version=3, role="combined" if same_host else "agent", connection={
        "stream_port": runtime.pop("bridge_port", 8766), "agent_port": runtime.pop("control_port", 8765),
        "bridge_port": bridge.pop("control_port", 8767)})
    if same_host:
        bridge.pop("mode", None)
        bridge.pop("avatar_rig", None)
    else:
        data.pop("bridge", None)
    return data
