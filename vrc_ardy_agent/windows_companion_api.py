from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .status_serialization import to_jsonable


class WindowsCompanionControlService:
    """Read-only diagnostics plus cooperative shutdown for the Windows side."""

    def __init__(
        self,
        runtime: Any,
        *,
        shutdown_requested: threading.Event,
        enable_test_controls: bool = False,
        screenshot_provider: Callable[[], bytes] | None = None,
        calibration: Any | None = None,
    ) -> None:
        if not isinstance(shutdown_requested, threading.Event):
            raise TypeError("shutdown_requested must be a threading.Event")
        self._calibration_lock = threading.RLock()
        self._calibration_target = None
        self._calibration = calibration
        self._runtime = runtime
        self._shutdown_requested = shutdown_requested
        self._enable_test_controls = bool(enable_test_controls)
        self._screenshot_provider = screenshot_provider

    def health(self) -> dict[str, object]:
        status = self.status()
        return {
            "status": "ok" if status.get("running") else "stopped",
            "heartbeat_monotonic": status.get("heartbeat_monotonic"),
        }

    def status(self) -> dict[str, object]:
        value = to_jsonable(self._runtime.snapshot())
        if not isinstance(value, dict):
            raise TypeError("Windows companion snapshot is not serializable")
        observer_snapshot = getattr(self._screenshot_provider, "snapshot", None)
        if callable(observer_snapshot):
            value["debug_observer"] = to_jsonable(observer_snapshot())
        if self._calibration is not None: value["calibration"] = self._calibration.status()
        return value

    def calibrate(self, payload):
        if self._calibration is None:
            raise RuntimeError("calibration is unavailable; update the Windows bridge")
        if not isinstance(payload, dict):
            raise ValueError("calibration payload must be an object")
        action = payload.get('action')
        if action not in ('start', 'update', 'heartbeat', 'finish', 'status'):
            raise ValueError("unknown calibration action")
        body = {k: v for k, v in payload.items() if k != 'action'}
        with self._calibration_lock:
            requires_target = action in ('start', 'update', 'heartbeat') or (
                action == 'finish' and payload.get('verified') is True)
            if requires_target:
                status = self.status()
                sensors = status.get('sensor_supervisor', {})
                ready = status.get('running') and sensors.get('running', True)
                ready = ready and sensors.get('state', 'RUNNING') == 'RUNNING'
                target = (sensors.get('epoch'), sensors.get('target'))
                if action != 'start' and (not ready or target != self._calibration_target):
                    current = self._calibration.status({'token': body.get('token')})
                    if current.get('state') == 'active':
                        self._calibration.finish(dict(token=body['token'],
                            revision=current['revision'], verified=False))
                    raise RuntimeError('VRChat target changed; calibration canceled, restart preview')
                if not ready:
                    raise RuntimeError('target VRChat is not attached; open the configured account first')
            result = getattr(self._calibration, action)(body)
            if action == 'start':
                self._calibration_target = target
            return result

    @staticmethod
    def actions() -> dict[str, object]:
        return {
            "actions": {},
            "role": "windows_device_gateway",
            "note": "actions are scheduled only by the Mac companion",
        }

    def stop(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict) or set(payload) - {"reason"}:
            raise ValueError("stop body may contain only reason")
        reason = payload.get("reason", "agentctl")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        self._shutdown_requested.set()
        return {
            "reason": reason.strip(),
            "shutdown_requested": True,
        }

    @staticmethod
    def test_utterance(_payload: object) -> dict[str, object]:
        raise PermissionError("Windows does not schedule character actions")

    @staticmethod
    def test_say(_payload: object) -> dict[str, object]:
        raise PermissionError("Windows does not schedule character actions")

    @staticmethod
    def test_motion(_payload: object) -> dict[str, object]:
        raise PermissionError("Windows does not schedule character actions")

    @staticmethod
    def test_bundle(_payload: object) -> dict[str, object]:
        raise PermissionError("Windows does not schedule character actions")

    def test_screenshot(self) -> bytes:
        if not self._enable_test_controls:
            raise PermissionError("test controls are disabled")
        if self._screenshot_provider is None:
            raise RuntimeError("observer screenshot provider is unavailable")
        jpeg = self._screenshot_provider()
        if not isinstance(jpeg, bytes) or not jpeg:
            raise RuntimeError("observer screenshot provider returned no bytes")
        return jpeg
