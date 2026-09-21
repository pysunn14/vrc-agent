from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import threading
import time
from typing import Protocol

from .action_contracts import ControlResource, OutputEnvelope, ResourceLease


class GatewayDropReason(StrEnum):
    SESSION = "session"
    LEASE = "lease"
    SEQUENCE = "sequence"
    EXPIRED = "expired"
    INVALID = "invalid"
    SINK_ERROR = "sink_error"


class DeviceSink(Protocol):
    def apply(self, envelope: OutputEnvelope) -> None: ...

    def neutralize(self, resource: ControlResource) -> None: ...


@dataclass(frozen=True, slots=True)
class GatewayDecision:
    accepted: bool
    reason: GatewayDropReason | None = None


@dataclass(frozen=True, slots=True)
class GatewayResourceSnapshot:
    resource: ControlResource
    lease_token: int
    action_id: str | None
    last_sequence: int
    expires_monotonic: float | None


@dataclass(frozen=True, slots=True)
class GatewaySnapshot:
    session_id: str | None
    resources: tuple[GatewayResourceSnapshot, ...]
    dropped_messages: tuple[tuple[GatewayDropReason, int], ...]
    watchdog_expirations: int
    heartbeat_monotonic: float
    last_errors: tuple[str, ...]

    def resource(self, resource: ControlResource) -> GatewayResourceSnapshot:
        match = next(
            (item for item in self.resources if item.resource == resource),
            None,
        )
        if match is None:
            raise KeyError(resource)
        return match


@dataclass(slots=True)
class _ResourceState:
    lease_token: int = 0
    action_id: str | None = None
    last_sequence: int = -1
    expires_monotonic: float | None = None


class DeviceGateway:
    """The sole process boundary allowed to write final VRChat device state."""

    def __init__(
        self,
        *,
        sink: DeviceSink,
        monotonic: Callable[[], float] = time.monotonic,
        error_history_limit: int = 50,
    ) -> None:
        if error_history_limit <= 0:
            raise ValueError("error_history_limit must be positive")
        self._sink = sink
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._exclusive_session: str | None = None
        self._resources = {
            resource: _ResourceState() for resource in ControlResource
        }
        self._drops: Counter[GatewayDropReason] = Counter()
        self._watchdog_expirations = 0
        self._last_errors: deque[str] = deque(maxlen=error_history_limit)
        self._heartbeat_monotonic = monotonic()

    def open_session(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        with self._lock:
            if self._exclusive_session and self._exclusive_session != session_id:
                raise RuntimeError("calibration owns the Windows output; finish calibration first")
            if self._session_id == session_id:
                self._touch_locked()
                return
            self._neutralize_all_locked(context="session changed")
            self._session_id = session_id
            self._resources = {
                resource: _ResourceState() for resource in ControlResource
            }
            self._touch_locked()

    def claim_exclusive_session(self, session_id: str) -> None:
        """Atomically claim idle devices; never displace a connected agent."""
        with self._lock:
            if self._session_id is not None:
                raise RuntimeError("agent or calibration is connected; stop it before calibration")
            self.open_session(session_id)
            self._exclusive_session = session_id

    def release_exclusive_session(self, session_id: str) -> None:
        """Release a calibration only after its outputs were neutralized."""
        with self._lock:
            if self._exclusive_session != session_id or self._session_id != session_id:
                raise RuntimeError("exclusive output session changed")
            if any(state.action_id is not None for state in self._resources.values()):
                raise RuntimeError("neutralize owned outputs before releasing calibration")
            self._exclusive_session = self._session_id = None
            self._touch_locked()

    def close_session(self, session_id: str) -> GatewayDecision:
        with self._lock:
            if session_id != self._session_id:
                return self._drop_locked(GatewayDropReason.SESSION)
            self._neutralize_all_locked(context="session closed")
            for state in self._resources.values():
                state.action_id = None
                state.expires_monotonic = None
            self._session_id = None
            self._exclusive_session = None
            self._touch_locked()
            return GatewayDecision(accepted=True)

    def authorize(
        self,
        session_id: str,
        lease: ResourceLease,
    ) -> GatewayDecision:
        with self._lock:
            if session_id != self._session_id:
                return self._drop_locked(GatewayDropReason.SESSION)
            if lease.action_id is None or lease.token <= 0:
                return self._drop_locked(GatewayDropReason.INVALID)
            state = self._resources[lease.resource]
            if (
                lease.token == state.lease_token
                and lease.action_id == state.action_id
            ):
                self._touch_locked()
                return GatewayDecision(accepted=True)
            if lease.token <= state.lease_token:
                return self._drop_locked(GatewayDropReason.LEASE)
            state.lease_token = lease.token
            state.action_id = lease.action_id
            state.last_sequence = -1
            state.expires_monotonic = None
            self._touch_locked()
            return GatewayDecision(accepted=True)

    def neutralize(
        self,
        session_id: str,
        resource: ControlResource,
        *,
        lease_token: int,
    ) -> GatewayDecision:
        with self._lock:
            if session_id != self._session_id:
                return self._drop_locked(GatewayDropReason.SESSION)
            state = self._resources[resource]
            if lease_token < state.lease_token:
                return self._drop_locked(GatewayDropReason.LEASE)
            if lease_token > state.lease_token:
                state.lease_token = lease_token
                state.last_sequence = -1
            state.action_id = None
            state.expires_monotonic = None
            neutralized = self._safe_neutralize_locked(
                resource,
                context="explicit neutralize",
            )
            if not neutralized:
                return self._drop_locked(GatewayDropReason.SINK_ERROR)
            self._touch_locked()
            return GatewayDecision(accepted=True)

    def apply_envelope(
        self,
        envelope: OutputEnvelope,
        *,
        received_monotonic: float | None = None,
        now_monotonic: float | None = None,
    ) -> GatewayDecision:
        received = self._monotonic() if received_monotonic is None else received_monotonic
        now = self._monotonic() if now_monotonic is None else now_monotonic
        with self._lock:
            if envelope.session_id != self._session_id:
                return self._drop_locked(GatewayDropReason.SESSION)
            if (
                not envelope.action_id
                or envelope.sequence < 0
                or envelope.ttl_ms <= 0
                or now < received
            ):
                return self._drop_locked(GatewayDropReason.INVALID)

            state = self._resources[envelope.resource]
            if (
                envelope.lease_token != state.lease_token
                or envelope.action_id != state.action_id
            ):
                return self._drop_locked(GatewayDropReason.LEASE)
            if envelope.sequence <= state.last_sequence:
                return self._drop_locked(GatewayDropReason.SEQUENCE)

            expires = received + envelope.ttl_ms / 1000.0
            if now > expires:
                return self._drop_locked(GatewayDropReason.EXPIRED)

            try:
                self._sink.apply(envelope)
            except Exception as exc:
                self._last_errors.append(
                    f"{envelope.resource.value} apply failed: {exc}"
                )
                return self._drop_locked(GatewayDropReason.SINK_ERROR)

            state.last_sequence = envelope.sequence
            state.expires_monotonic = expires
            self._touch_locked()
            return GatewayDecision(accepted=True)

    def poll_expired(
        self,
        *,
        now_monotonic: float | None = None,
    ) -> tuple[ControlResource, ...]:
        now = self._monotonic() if now_monotonic is None else now_monotonic
        expired: list[ControlResource] = []
        with self._lock:
            for resource, state in self._resources.items():
                if (
                    state.action_id is None
                    or state.expires_monotonic is None
                    or now < state.expires_monotonic
                ):
                    continue
                # Revoking the owner prevents a late packet with the timed-out
                # lease from reactivating output. A higher token is required.
                state.action_id = None
                state.expires_monotonic = None
                self._safe_neutralize_locked(resource, context="TTL expired")
                self._watchdog_expirations += 1
                expired.append(resource)
            self._touch_locked()
        return tuple(expired)

    def snapshot(self) -> GatewaySnapshot:
        with self._lock:
            return GatewaySnapshot(
                session_id=self._session_id,
                resources=tuple(
                    GatewayResourceSnapshot(
                        resource=resource,
                        lease_token=state.lease_token,
                        action_id=state.action_id,
                        last_sequence=state.last_sequence,
                        expires_monotonic=state.expires_monotonic,
                    )
                    for resource, state in sorted(
                        self._resources.items(), key=lambda item: item[0].value
                    )
                ),
                dropped_messages=tuple(
                    sorted(self._drops.items(), key=lambda item: item[0].value)
                ),
                watchdog_expirations=self._watchdog_expirations,
                heartbeat_monotonic=self._heartbeat_monotonic,
                last_errors=tuple(self._last_errors),
            )

    def _drop_locked(self, reason: GatewayDropReason) -> GatewayDecision:
        self._drops[reason] += 1
        self._touch_locked()
        return GatewayDecision(accepted=False, reason=reason)

    def _neutralize_all_locked(self, *, context: str) -> None:
        for resource in ControlResource:
            self._safe_neutralize_locked(resource, context=context)

    def _safe_neutralize_locked(
        self,
        resource: ControlResource,
        *,
        context: str,
    ) -> bool:
        try:
            self._sink.neutralize(resource)
        except Exception as exc:
            self._last_errors.append(
                f"{resource.value} neutralize failed during {context}: {exc}"
            )
            return False
        return True

    def _touch_locked(self) -> None:
        self._heartbeat_monotonic = self._monotonic()
