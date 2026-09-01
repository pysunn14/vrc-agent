from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IDLE_PROMPT = "A person stands still naturally."


@dataclass(frozen=True)
class ControlResult:
    action: str
    message: str
    quit_requested: bool = False


def apply_control_line(session: Any, line: str) -> ControlResult:
    command = line.strip()
    if not command:
        return ControlResult(action="ignored", message="")

    keyword = command.lower()
    if keyword == "idle":
        session.set_prompt(IDLE_PROMPT)
        return ControlResult(action="idle", message=f"prompt -> {IDLE_PROMPT!r}")
    if keyword == "stop":
        session.pause()
        return ControlResult(action="stop", message="playback paused; ARDY remains resident")
    if keyword == "quit":
        session.request_stop()
        return ControlResult(action="quit", message="shutdown requested", quit_requested=True)

    session.set_prompt(command)
    return ControlResult(action="prompt", message=f"prompt -> {command!r}")
