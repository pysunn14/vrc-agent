from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time
import uuid


@dataclass(frozen=True, slots=True)
class TurnToken:
    turn_id: str
    version: int


@dataclass(frozen=True, slots=True)
class DialogueTurnSnapshot:
    latest_turn_id: str | None
    pending_turn_id: str | None
    version: int
    last_invalidation_reason: str | None
    heartbeat_monotonic: float


class DialogueTurnManager:
    """Owns the monotonic generation used to fence stale brain responses."""

    def __init__(self, *, id_factory: Callable[[], str] | None = None) -> None:
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._lock = threading.Lock()
        self._version = 0
        self._latest_turn_id: str | None = None
        self._pending: TurnToken | None = None
        self._last_invalidation_reason: str | None = None
        self._heartbeat_monotonic = time.monotonic()

    def begin_turn(self) -> TurnToken:
        turn_id = self._id_factory()
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("id_factory must return a non-empty string")
        with self._lock:
            self._version += 1
            token = TurnToken(turn_id=turn_id, version=self._version)
            self._latest_turn_id = turn_id
            self._pending = token
            self._last_invalidation_reason = None
            self._heartbeat_monotonic = time.monotonic()
            return token

    def is_current(self, token: TurnToken) -> bool:
        with self._lock:
            return self._pending == token

    def claim_current(self, token: TurnToken) -> bool:
        """Atomically accepts one response and rejects duplicates or stale results."""
        with self._lock:
            if self._pending != token:
                return False
            self._pending = None
            self._heartbeat_monotonic = time.monotonic()
            return True

    def apply_if_current(
        self,
        token: TurnToken,
        operation: Callable[[], None],
    ) -> bool:
        """Runs one side effect atomically against beginning a newer turn."""
        with self._lock:
            if self._pending != token:
                return False
            try:
                operation()
            finally:
                self._pending = None
                self._heartbeat_monotonic = time.monotonic()
            return True

    def invalidate_pending(self, *, reason: str) -> int:
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must not be empty")
        with self._lock:
            if self._pending is not None:
                self._version += 1
                self._pending = None
            self._last_invalidation_reason = reason
            self._heartbeat_monotonic = time.monotonic()
            return self._version

    def snapshot(self) -> DialogueTurnSnapshot:
        with self._lock:
            return DialogueTurnSnapshot(
                latest_turn_id=self._latest_turn_id,
                pending_turn_id=(
                    None if self._pending is None else self._pending.turn_id
                ),
                version=self._version,
                last_invalidation_reason=self._last_invalidation_reason,
                heartbeat_monotonic=self._heartbeat_monotonic,
            )
