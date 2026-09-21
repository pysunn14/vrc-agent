from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import threading
import time
from typing import Any
import uuid

from .mac_input_transport import MacInputTransport
from .mac_output_transport import MacOutputTransport
from .stream_protocol import (
    AckMessage,
    AudioChunkMessage,
    HeartbeatMessage,
    HelloMessage,
    MAX_MESSAGE_BYTES,
    OutputCompletedMessage,
    ProtocolError,
    ScreenshotResponseMessage,
    SessionOpenMessage,
    StreamMessage,
    decode_message,
    encode_message,
)


@dataclass(frozen=True, slots=True)
class MacBridgeStatus:
    running: bool = False
    bound_host: str | None = None
    bound_port: int | None = None
    connected: bool = False
    client_instance_id: str | None = None
    session_id: str | None = None
    connections_opened: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    protocol_errors: int = 0
    heartbeat_monotonic: float = 0.0
    last_error: str | None = None


class _ConnectionSender:
    def __init__(self, connection: Any, sent: Callable[[], None]) -> None:
        self._connection = connection
        self._sent = sent
        self._lock = threading.Lock()

    def __call__(self, message: StreamMessage) -> None:
        encoded = encode_message(message)
        with self._lock:
            self._connection.send(encoded)
        self._sent()


class MacBridgeServer:
    """Accepts one Windows bridge and attaches it to MacOutputTransport."""

    def __init__(
        self,
        *,
        output: MacOutputTransport,
        input_transport: MacInputTransport,
        host: str = "0.0.0.0",
        port: int = 8766,
        handshake_timeout_seconds: float = 5.0,
        heartbeat_interval_seconds: float = 2.0,
        session_id_factory: Callable[[], str] | None = None,
        on_session_opened: Callable[[str], None] | None = None,
        on_session_closed: Callable[[str, str], None] | None = None,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        if handshake_timeout_seconds <= 0:
            raise ValueError("handshake_timeout_seconds must be positive")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.output = output
        self.input_transport = input_transport
        self.host = host
        self.port = int(port)
        self.handshake_timeout_seconds = float(handshake_timeout_seconds)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self._session_id_factory = session_id_factory or (
            lambda: uuid.uuid4().hex
        )
        self._on_session_opened = on_session_opened
        self._on_session_closed = on_session_closed
        self._condition = threading.Condition(threading.RLock())
        self._server: Any | None = None
        self._server_thread: threading.Thread | None = None
        self._active_connection: Any | None = None
        self._active_session_id: str | None = None
        self._status = MacBridgeStatus(heartbeat_monotonic=time.monotonic())

    @property
    def bound_port(self) -> int:
        with self._condition:
            if self._status.bound_port is None:
                raise RuntimeError("bridge server has not started")
            return self._status.bound_port

    def start(self) -> None:
        try:
            from websockets.sync.server import serve
        except ImportError as exc:
            raise RuntimeError("websockets is required for the Mac bridge") from exc

        with self._condition:
            if self._server is not None:
                raise RuntimeError("bridge server is already started")
            server = serve(
                self._handle_connection,
                self.host,
                self.port,
                compression=None,
                max_size=MAX_MESSAGE_BYTES,
                ping_interval=10,
                ping_timeout=10,
            )
            bound_host, bound_port = server.socket.getsockname()[:2]
            self._server = server
            self._status = replace(
                self._status,
                running=True,
                bound_host=str(bound_host),
                bound_port=int(bound_port),
                heartbeat_monotonic=time.monotonic(),
                last_error=None,
            )
            thread = threading.Thread(
                target=self._serve,
                args=(server,),
                name="companion-websocket-server",
                daemon=True,
            )
            self._server_thread = thread
            thread.start()

    def stop(self) -> None:
        with self._condition:
            server = self._server
            thread = self._server_thread
            connection = self._active_connection
            if server is None:
                return
        if connection is not None:
            try:
                connection.close(code=1001, reason="server stopping")
            except Exception:
                pass
        server.shutdown()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            error = "bridge server thread did not stop after shutdown"
            with self._condition:
                # Keep the live thread and running state observable. Clearing them
                # here would claim an authoritative stop that did not happen.
                self._status = replace(
                    self._status,
                    running=True,
                    heartbeat_monotonic=time.monotonic(),
                    last_error=error,
                )
                self._condition.notify_all()
            raise RuntimeError(error)
        with self._condition:
            self._server = None
            self._server_thread = None
            self._status = replace(
                self._status,
                running=False,
                connected=False,
                client_instance_id=None,
                session_id=None,
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()

    def wait_until_disconnected(self, *, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._status.connected:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def snapshot(self) -> MacBridgeStatus:
        with self._condition:
            return replace(self._status)

    def _serve(self, server: Any) -> None:
        try:
            server.serve_forever()
        except Exception as exc:
            with self._condition:
                self._status = replace(
                    self._status,
                    last_error=f"server failed: {type(exc).__name__}: {exc}",
                    heartbeat_monotonic=time.monotonic(),
                )
        finally:
            with self._condition:
                self._status = replace(
                    self._status,
                    running=False,
                    heartbeat_monotonic=time.monotonic(),
                )
                self._condition.notify_all()

    def _handle_connection(self, connection: Any) -> None:
        session_id: str | None = None
        reason = "connection closed"
        reserved = False
        try:
            first = connection.recv(timeout=self.handshake_timeout_seconds)
            self._count_received()
            hello = decode_message(first)
            if not isinstance(hello, HelloMessage):
                raise ProtocolError("first message must be hello")

            with self._condition:
                if self._active_connection is not None:
                    raise ProtocolError("another Windows bridge is already connected")
                self._active_connection = connection
                reserved = True

            session_id = self._session_id_factory()
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("session_id_factory returned an invalid id")
            sender = _ConnectionSender(connection, self._count_sent)
            self.output.open_session(session_id, sender)
            try:
                self.input_transport.open_session(session_id, sender)
            except BaseException:
                self.output.close_session(
                    session_id,
                    reason="input session failed to open",
                )
                raise
            with self._condition:
                self._active_session_id = session_id
                self._status = replace(
                    self._status,
                    connected=True,
                    client_instance_id=hello.client_instance_id,
                    session_id=session_id,
                    connections_opened=self._status.connections_opened + 1,
                    heartbeat_monotonic=time.monotonic(),
                    last_error=None,
                )
                self._condition.notify_all()
            sender(SessionOpenMessage(session_id=session_id))
            if self._on_session_opened is not None:
                self._on_session_opened(session_id)

            heartbeat_sequence = 0
            next_heartbeat = time.monotonic() + self.heartbeat_interval_seconds
            while True:
                now = time.monotonic()
                if now >= next_heartbeat:
                    sender(
                        HeartbeatMessage(
                            session_id=session_id,
                            sender="mac",
                            sequence=heartbeat_sequence,
                        )
                    )
                    heartbeat_sequence += 1
                    next_heartbeat = now + self.heartbeat_interval_seconds
                timeout = max(0.001, next_heartbeat - time.monotonic())
                try:
                    raw = connection.recv(timeout=timeout)
                except TimeoutError:
                    continue
                self._count_received()
                message = decode_message(raw)
                if isinstance(message, (AckMessage, OutputCompletedMessage)):
                    self.output.receive(message)
                    continue
                if isinstance(
                    message,
                    (AudioChunkMessage, ScreenshotResponseMessage, HeartbeatMessage),
                ):
                    self.input_transport.receive(message)
                    continue
                raise ProtocolError(
                    f"message {type(message).__name__} is invalid from Windows"
                )
        except ProtocolError as exc:
            reason = f"protocol error: {exc}"
            self._record_protocol_error(reason)
            try:
                connection.close(code=1002, reason="protocol error")
            except Exception:
                pass
        except Exception as exc:
            reason = f"connection ended: {type(exc).__name__}: {exc}"
            self._record_error(reason)
        finally:
            if session_id is not None:
                if self._on_session_closed is not None:
                    try:
                        self._on_session_closed(session_id, reason)
                    except BaseException as exc:
                        self._record_error(
                            "session close callback failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                self.input_transport.close_session(session_id, reason=reason)
                self.output.close_session(session_id, reason=reason)
            with self._condition:
                if reserved and self._active_connection is connection:
                    self._active_connection = None
                    self._active_session_id = None
                    self._status = replace(
                        self._status,
                        connected=False,
                        client_instance_id=None,
                        session_id=None,
                        heartbeat_monotonic=time.monotonic(),
                    )
                    self._condition.notify_all()

    def _count_received(self) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                messages_received=self._status.messages_received + 1,
                heartbeat_monotonic=time.monotonic(),
            )

    def _count_sent(self) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                messages_sent=self._status.messages_sent + 1,
                heartbeat_monotonic=time.monotonic(),
            )

    def _record_protocol_error(self, error: str) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                protocol_errors=self._status.protocol_errors + 1,
                heartbeat_monotonic=time.monotonic(),
                last_error=error,
            )

    def _record_error(self, error: str) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                heartbeat_monotonic=time.monotonic(),
                last_error=error,
            )
