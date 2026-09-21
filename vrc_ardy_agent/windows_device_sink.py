from __future__ import annotations

from collections.abc import Callable, Mapping
import threading

from .action_contracts import ControlResource, OutputEnvelope
from .device_payloads import decode_pose_payload, decode_wav_payload
from .stream_protocol import OutputCompletedMessage


class WindowsDeviceSink:
    """The only adapter that touches VRChat pose and virtual-mic devices."""

    def __init__(
        self,
        *,
        pose_sink: object,
        wave_player: object,
        completion_sender: Callable[[OutputCompletedMessage], None] | None = None,
    ) -> None:
        self._pose_sink = pose_sink
        self._wave_player = wave_player
        self._lock = threading.Lock()
        self._completion_sender = completion_sender

    def set_completion_sender(
        self,
        sender: Callable[[OutputCompletedMessage], None] | None,
    ) -> None:
        with self._lock:
            self._completion_sender = sender

    def apply(self, envelope: OutputEnvelope) -> None:
        payload = envelope.payload
        if not isinstance(payload, Mapping):
            raise ValueError("device payload must be an object")
        kind = payload.get("kind")
        if envelope.resource == ControlResource.FULL_BODY_POSE:
            if kind != "six_point_pose":
                raise ValueError("FULL_BODY_POSE requires a pose payload")
            self._pose_sink.send(decode_pose_payload(payload))
            return
        if envelope.resource == ControlResource.VOICE_OUTPUT:
            if kind != "wav":
                raise ValueError("VOICE_OUTPUT requires a WAV payload")
            with self._lock:
                sender = self._completion_sender
            if sender is None:
                raise RuntimeError("speech completion sender is not connected")

            def completed(state: str, error: str | None) -> None:
                sender(
                    OutputCompletedMessage(
                        session_id=envelope.session_id,
                        action_id=envelope.action_id,
                        resource=envelope.resource,
                        lease_token=envelope.lease_token,
                        state=state,
                        error=error,
                    )
                )

            self._wave_player.play(decode_wav_payload(payload), completed)
            return
        raise ValueError(f"unsupported device resource: {envelope.resource}")

    def neutralize(self, resource: ControlResource) -> None:
        if resource == ControlResource.FULL_BODY_POSE:
            self._pose_sink.neutralize_pose()
            return
        if resource == ControlResource.VOICE_OUTPUT:
            self._wave_player.stop()
            return
        raise ValueError(f"unsupported device resource: {resource}")

    def close(self) -> None:
        errors: list[BaseException] = []
        for resource in ControlResource:
            try:
                self.neutralize(resource)
            except BaseException as exc:
                errors.append(exc)
        try:
            self._pose_sink.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError(
                "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            )
