from __future__ import annotations

from dataclasses import dataclass, replace
import socket
import threading
import time
from typing import Any, Callable

from .follow_protocol import FollowProtocolError, MAX_PACKET_BYTES, TargetObservation, decode_target_observation


@dataclass(frozen=True)
class ReceivedObservation:
    observation: TargetObservation
    received_monotonic: float


class LatestObservationStore:
    """Concurrency-safe store that only accepts forward progress from one sender session."""

    def __init__(self, *, session_takeover_after_seconds: float = 0.3) -> None:
        if session_takeover_after_seconds <= 0:
            raise ValueError("session_takeover_after_seconds must be positive")
        self.session_takeover_after_seconds = float(session_takeover_after_seconds)
        self._lock = threading.Lock()
        self._latest: ReceivedObservation | None = None

    def accept(self, observation: TargetObservation, *, received_monotonic: float) -> bool:
        received = ReceivedObservation(
            observation=observation,
            received_monotonic=float(received_monotonic),
        )
        with self._lock:
            current = self._latest
            if current is not None:
                same_session = current.observation.session_id == observation.session_id
                if same_session and observation.sequence <= current.observation.sequence:
                    return False
                if not same_session:
                    current_age = received.received_monotonic - current.received_monotonic
                    if current_age < self.session_takeover_after_seconds:
                        return False
            self._latest = received
            return True

    def snapshot(self) -> ReceivedObservation | None:
        with self._lock:
            return self._latest


@dataclass(frozen=True)
class UdpReceiverStatus:
    running: bool = False
    packets_received: int = 0
    packets_accepted: int = 0
    invalid_packets: int = 0
    last_sender: tuple[str, int] | None = None
    last_error: str | None = None
    heartbeat_monotonic: float = 0.0


class UdpObservationReceiver:
    """Receive complete target observations without allowing packet reordering to rewind state."""

    def __init__(
        self,
        *,
        bind_host: str = "0.0.0.0",
        port: int = 9200,
        store: LatestObservationStore | None = None,
        socket_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        self.bind_host = bind_host
        self.port = int(port)
        self.store = store or LatestObservationStore()
        self._socket_factory = socket_factory or (
            lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: Any | None = None
        self._bound_port: int | None = None
        self._status = UdpReceiverStatus()

    @property
    def bound_port(self) -> int:
        if self._bound_port is None:
            raise RuntimeError("receiver has not started")
        return self._bound_port

    @property
    def status(self) -> UdpReceiverStatus:
        with self._lock:
            return replace(self._status)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("receiver is already started")
            self._stop_event.clear()
            sock = self._socket_factory()
            try:
                sock.bind((self.bind_host, self.port))
                sock.settimeout(0.1)
                self._bound_port = int(sock.getsockname()[1])
            except BaseException:
                sock.close()
                raise
            self._socket = sock
            self._status = replace(
                self._status,
                running=True,
                last_error=None,
                heartbeat_monotonic=time.monotonic(),
            )
            self._thread = threading.Thread(
                target=self._run,
                name="follow-observation-receiver",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        sock = self._socket
        if sock is not None:
            sock.close()
        with self._lock:
            self._thread = None
            self._socket = None
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=time.monotonic(),
            )

    def _run(self) -> None:
        sock = self._socket
        if sock is None:
            return
        try:
            while not self._stop_event.is_set():
                try:
                    packet, sender = sock.recvfrom(MAX_PACKET_BYTES + 1)
                except socket.timeout:
                    with self._lock:
                        self._status = replace(
                            self._status,
                            heartbeat_monotonic=time.monotonic(),
                        )
                    continue

                now = time.monotonic()
                try:
                    observation = decode_target_observation(packet)
                except FollowProtocolError:
                    with self._lock:
                        self._status = replace(
                            self._status,
                            packets_received=self._status.packets_received + 1,
                            invalid_packets=self._status.invalid_packets + 1,
                            last_sender=(str(sender[0]), int(sender[1])),
                            heartbeat_monotonic=now,
                        )
                    continue

                accepted = self.store.accept(observation, received_monotonic=now)
                with self._lock:
                    self._status = replace(
                        self._status,
                        packets_received=self._status.packets_received + 1,
                        packets_accepted=self._status.packets_accepted + int(accepted),
                        last_sender=(str(sender[0]), int(sender[1])),
                        heartbeat_monotonic=now,
                    )
        except OSError as exc:
            if not self._stop_event.is_set():
                with self._lock:
                    self._status = replace(self._status, last_error=str(exc))
        finally:
            with self._lock:
                self._status = replace(
                    self._status,
                    running=False,
                    heartbeat_monotonic=time.monotonic(),
                )
