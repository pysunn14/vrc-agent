from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, TypeAlias


class PlanValidationError(ValueError):
    """Raised when a brain response is not the exact executable contract."""


class ActionType(StrEnum):
    SAY = "say"
    ARDY_MOTION = "ardy_motion"
    MOTION = "motion"


class ControlResource(StrEnum):
    VOICE_OUTPUT = "VOICE_OUTPUT"
    FULL_BODY_POSE = "FULL_BODY_POSE"


class ActionState(StrEnum):
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class SayAction:
    text: str

    @property
    def action_type(self) -> ActionType:
        return ActionType.SAY

    @property
    def resource(self) -> ControlResource:
        return ControlResource.VOICE_OUTPUT


@dataclass(frozen=True, slots=True)
class ArdyMotionAction:
    prompt: str
    duration_seconds: float

    @property
    def action_type(self) -> ActionType:
        return ActionType.ARDY_MOTION

    @property
    def resource(self) -> ControlResource:
        return ControlResource.FULL_BODY_POSE


@dataclass(frozen=True, slots=True)
class PresetMotionAction:
    name: str
    duration_seconds: float | None = None

    @property
    def action_type(self) -> ActionType:
        return ActionType.MOTION

    @property
    def resource(self) -> ControlResource:
        return ControlResource.FULL_BODY_POSE


Action: TypeAlias = SayAction | ArdyMotionAction | PresetMotionAction


@dataclass(frozen=True, slots=True)
class ActionBundle:
    actions: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class ResourceLease:
    resource: ControlResource
    token: int
    action_id: str | None


@dataclass(frozen=True, slots=True)
class ExecutionCommand:
    action_id: str
    turn_id: str
    action: Action
    resource: ControlResource
    lease_token: int


@dataclass(frozen=True, slots=True)
class OutputEnvelope:
    session_id: str
    action_id: str
    resource: ControlResource
    lease_token: int
    sequence: int
    ttl_ms: int
    payload: object


_SENTENCE_TERMINATOR = re.compile(r"[.!?。！？]+")


def _require_mapping(value: object, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanValidationError(f"{location} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise PlanValidationError(f"{location} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    location: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    raise PlanValidationError(f"{location} has invalid fields ({', '.join(details)})")


def _validate_say(raw: Mapping[str, Any], *, location: str) -> SayAction:
    _require_exact_keys(raw, {"type", "text"}, location=location)
    text = raw["text"]
    if not isinstance(text, str):
        raise PlanValidationError(f"{location}.text must be a string")
    text = text.strip()
    if not text:
        raise PlanValidationError(f"{location}.text must not be empty")
    if len(text) > 500:
        raise PlanValidationError(f"{location}.text must be short")
    sentence_count = len(_SENTENCE_TERMINATOR.findall(text))
    if sentence_count > 2:
        raise PlanValidationError(f"{location}.text must contain at most two sentences")
    return SayAction(text=text)


def _validate_ardy_motion(
    raw: Mapping[str, Any],
    *,
    location: str,
) -> ArdyMotionAction:
    _require_exact_keys(
        raw,
        {"type", "prompt", "duration_seconds"},
        location=location,
    )
    prompt = raw["prompt"]
    if not isinstance(prompt, str):
        raise PlanValidationError(f"{location}.prompt must be a string")
    prompt = prompt.strip()
    if not prompt:
        raise PlanValidationError(f"{location}.prompt must not be empty")
    if len(prompt) > 500:
        raise PlanValidationError(f"{location}.prompt must be short")
    if not prompt.isascii() or not any(character.isalpha() for character in prompt):
        raise PlanValidationError(f"{location}.prompt must be English ASCII text")

    duration = raw["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise PlanValidationError(f"{location}.duration_seconds must be a number")
    duration = float(duration)
    if not 1.0 <= duration <= 10.0:
        raise PlanValidationError(
            f"{location}.duration_seconds must be between 1 and 10"
        )
    return ArdyMotionAction(prompt=prompt, duration_seconds=duration)


def _validate_preset(raw, *, location, behaviors):
    name = raw.get("name")
    if not isinstance(name, str) or name not in behaviors:
        raise PlanValidationError(f"{location}.name is not a registered motion")
    spec = behaviors[name]
    expected = {"type", "name"}
    if not spec.finite and "duration_seconds" in raw: expected.add("duration_seconds")
    _require_exact_keys(raw, expected, location=location)
    duration = raw.get("duration_seconds")
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 1 <= duration <= 10:
            raise PlanValidationError("motion duration must be between 1 and 10 seconds")
        duration = float(duration)
    elif "duration_seconds" in raw:
        raise PlanValidationError("duration_seconds must be a number")
    return PresetMotionAction(name, duration)


def validate_action_bundle(payload: object, *, behaviors=None) -> ActionBundle:
    root = _require_mapping(payload, location="response")
    _require_exact_keys(root, {"actions"}, location="response")
    raw_actions = root["actions"]
    if not isinstance(raw_actions, list):
        raise PlanValidationError("response.actions must be an array")
    if not 1 <= len(raw_actions) <= 2:
        raise PlanValidationError("response.actions must contain one or two actions")

    actions: list[Action] = []
    seen_resources: set[ControlResource] = set()
    for index, value in enumerate(raw_actions):
        location = f"response.actions[{index}]"
        raw = _require_mapping(value, location=location)
        action_type = raw.get("type")
        if action_type == ActionType.SAY.value:
            parsed = _validate_say(raw, location=location)
        elif action_type == ActionType.ARDY_MOTION.value:
            parsed = _validate_ardy_motion(raw, location=location)
        elif action_type == ActionType.MOTION.value:
            parsed = _validate_preset(raw, location=location, behaviors=behaviors or {})
        else:
            raise PlanValidationError(f"{location}.type is not supported")
        if parsed.resource in seen_resources:
            raise PlanValidationError(f"duplicate {parsed.action_type.value} action")
        seen_resources.add(parsed.resource)
        actions.append(parsed)

    return ActionBundle(actions=tuple(actions))
