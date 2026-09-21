from __future__ import annotations

from .action_contracts import ControlResource
from .device_gateway import DeviceGateway, GatewayDecision
from .stream_protocol import (
    AckMessage,
    AuthorizeMessage,
    OutputDataMessage,
    OutputNeutralizeMessage,
    SessionOpenMessage,
)


class OutputDataRejectedError(RuntimeError):
    """Raised when Windows and Mac disagree about active output ownership."""


class WindowsOutputController:
    """Maps validated bridge commands onto the sole Windows device gateway."""

    def __init__(self, *, gateway: DeviceGateway) -> None:
        self.gateway = gateway

    def open_session(self, message: SessionOpenMessage) -> None:
        if not isinstance(message, SessionOpenMessage):
            raise TypeError("open_session requires SessionOpenMessage")
        self.gateway.open_session(message.session_id)

    def close_session(self, session_id: str) -> bool:
        return self.gateway.close_session(session_id).accepted

    def handle(
        self,
        message: AuthorizeMessage | OutputNeutralizeMessage | OutputDataMessage,
    ) -> AckMessage | None:
        if isinstance(message, AuthorizeMessage):
            decision = self.gateway.authorize(message.session_id, message.lease)
            return _ack(message.message_id, decision)
        if isinstance(message, OutputNeutralizeMessage):
            decision = self.gateway.neutralize(
                message.session_id,
                message.resource,
                lease_token=message.lease_token,
            )
            return _ack(message.message_id, decision)
        if isinstance(message, OutputDataMessage):
            decision = self.gateway.apply_envelope(message.envelope)
            if not decision.accepted:
                reason = decision.reason.value if decision.reason is not None else "unknown"
                envelope = message.envelope
                raise OutputDataRejectedError(
                    f"{envelope.resource.value} output rejected: {reason}; "
                    f"lease={envelope.lease_token} sequence={envelope.sequence}"
                )
            return None
        raise TypeError(f"unsupported Windows output message: {type(message).__name__}")

    def poll_watchdog(self) -> tuple[ControlResource, ...]:
        return self.gateway.poll_expired()


def _ack(message_id: str, decision: GatewayDecision) -> AckMessage:
    return AckMessage(
        message_id=message_id,
        accepted=decision.accepted,
        reason=None if decision.reason is None else decision.reason.value,
    )
