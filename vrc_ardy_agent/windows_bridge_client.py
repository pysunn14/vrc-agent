from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Protocol

from .stream_protocol import (
    AuthorizeMessage,
    HelloMessage,
    MAX_MESSAGE_BYTES,
    OutputDataMessage,
    OutputNeutralizeMessage,
    ProtocolError,
    HeartbeatMessage,
    ScreenshotRequestMessage,
    SessionOpenMessage,
    StreamMessage,
    decode_message,
    encode_message,
)
from .windows_output_controller import WindowsOutputController
from .windows_input_transport import WindowsInputTransport


class CompletionTarget(Protocol):
    def set_completion_sender(
        self,
        sender: Callable[[StreamMessage], None] | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class WindowsBridgeStatus:
    running: bool = False
    connected: bool = False
    session_id: str | None = None
    connection_attempts: int = 0
    connections_opened: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    watchdog_expirations: int = 0
    protocol_errors: int = 0
    remote_heartbeat_sequence: int | None = None
    heartbeat_monotonic: float = 0.0
    last_error: str | None = None


class _ConnectionSender:
    def __init__(
        self,
        connection: Any,
        sent: Callable[[], None],
    ) -> None:
        self._connection = connection
        self._sent = sent
        self._lock = threading.Lock()

    def __call__(self, message: StreamMessage) -> None:
        encoded = encode_message(message)
        with self._lock:
            self._connection.send(encoded)
        self._sent()


class WindowsBridgeClient:
    """Maintains one fresh, non-replaying device session with the Mac."""

    def __init__(
        self,
        *,
        url: str,
        controller: WindowsOutputController,
        completion_target: CompletionTarget,
        input_transport: WindowsInputTransport,
        client_instance_id: str,
        reconnect_delay_seconds: float = 1.0,
        receive_poll_seconds: float = 0.1,
        handshake_timeout_seconds: float = 5.0,
        heartbeat_interval_seconds: float = 2.0,
        connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(url, str) or not url.startswith(("ws://", "wss://")):
            raise ValueError("url must use ws:// or wss://")
        client_instance_id = client_instance_id.strip()
        if not client_instance_id:
            raise ValueError("client_instance_id must not be empty")
        if reconnect_delay_seconds <= 0:
            raise ValueError("reconnect_delay_seconds must be positive")
        if receive_poll_seconds <= 0:
            raise ValueError("receive_poll_seconds must be positive")
        if handshake_timeout_seconds <= 0:
            raise ValueError("handshake_timeout_seconds must be positive")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.url = url
        self.controller = controller
        self.completion_target = completion_target
        self.input_transport = input_transport
        self.client_instance_id = client_instance_id
        self.reconnect_delay_seconds = float(reconnect_delay_seconds)
        self.receive_poll_seconds = float(receive_poll_seconds)
        self.handshake_timeout_seconds = float(handshake_timeout_seconds)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self._connection_factory = connection_factory or self._connect
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_connection: Any | None = None
        self._status = WindowsBridgeStatus(
            heartbeat_monotonic=time.monotonic()
        )

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("Windows bridge client is already started")
            self._stop_event.clear()
            self._status = replace(
                self._status,
                running=True,
                connected=False,
                session_id=None,
                remote_heartbeat_sequence=None,
                heartbeat_monotonic=time.monotonic(),
                last_error=None,
            )
            thread = threading.Thread(
                target=self._run,
                name="windows-companion-bridge",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        with self._condition:
            thread = self._thread
            connection = self._active_connection
            if thread is None:
                return
            self._stop_event.set()
        if connection is not None:
            try:
                connection.close(code=1001, reason="client stopping")
            except Exception:
                pass
        if thread is not threading.current_thread():
            thread.join(timeout=self.handshake_timeout_seconds + 2.0)
            if thread.is_alive():
                raise TimeoutError("Windows bridge client did not stop")
        with self._condition:
            if self._thread is thread:
                self._thread = None
            self._status = replace(
                self._status,
                running=False,
                connected=False,
                session_id=None,
                remote_heartbeat_sequence=None,
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()

    def wait_until_connected(self, *, timeout: float | None = None) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._status.connected,
                timeout=timeout,
            )

    def snapshot(self) -> WindowsBridgeStatus:
        with self._condition:
            return replace(self._status)

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._count_attempt()
                try:
                    with self._connection_factory() as connection:
                        with self._condition:
                            self._active_connection = connection
                        self._run_connection(connection)
                except ProtocolError as exc:
                    self._record_error(f"protocol error: {exc}", protocol=True)
                except Exception as exc:
                    if not self._stop_event.is_set():
                        self._record_error(
                            f"connection ended: {type(exc).__name__}: {exc}"
                        )
                finally:
                    with self._condition:
                        self._active_connection = None
                        self._status = replace(
                            self._status,
                            connected=False,
                            session_id=None,
                            remote_heartbeat_sequence=None,
                            heartbeat_monotonic=time.monotonic(),
                        )
                        self._condition.notify_all()

                # Reconnection creates a new authoritative session. There is
                # deliberately no outbound action queue to replay here.
                if self._stop_event.wait(self.reconnect_delay_seconds):
                    break
        finally:
            with self._condition:
                self._active_connection = None
                self._status = replace(
                    self._status,
                    running=False,
                    connected=False,
                    session_id=None,
                    remote_heartbeat_sequence=None,
                    heartbeat_monotonic=time.monotonic(),
                )
                self._condition.notify_all()

    def _run_connection(self, connection: Any) -> None:
        sender = _ConnectionSender(connection, self._count_sent)
        sender(HelloMessage(client_instance_id=self.client_instance_id))
        opened = decode_message(
            connection.recv(timeout=self.handshake_timeout_seconds)
        )
        self._count_received()
        if not isinstance(opened, SessionOpenMessage):
            raise ProtocolError("first Mac message must be session.open")

        self.controller.open_session(opened)
        try:
            self.input_transport.open_session(opened.session_id, sender)
        except BaseException:
            self.controller.close_session(opened.session_id)
            raise
        self.completion_target.set_completion_sender(sender)
        try:
            with self._condition:
                self._status = replace(
                    self._status,
                    connected=True,
                    session_id=opened.session_id,
                    remote_heartbeat_sequence=None,
                    connections_opened=self._status.connections_opened + 1,
                    heartbeat_monotonic=time.monotonic(),
                    last_error=None,
                )
                self._condition.notify_all()

            next_heartbeat = time.monotonic() + self.heartbeat_interval_seconds
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now >= next_heartbeat:
                    self.input_transport.send_heartbeat()
                    next_heartbeat = now + self.heartbeat_interval_seconds
                try:
                    raw = connection.recv(timeout=self.receive_poll_seconds)
                except TimeoutError:
                    expired = self.controller.poll_watchdog()
                    with self._condition:
                        self._status = replace(
                            self._status,
                            watchdog_expirations=(
                                self._status.watchdog_expirations + len(expired)
                            ),
                            heartbeat_monotonic=time.monotonic(),
                        )
                    continue

                message = decode_message(raw)
                self._count_received()
                if isinstance(
                    message,
                    (AuthorizeMessage, OutputNeutralizeMessage, OutputDataMessage),
                ):
                    response = self.controller.handle(message)
                    if response is not None:
                        sender(response)
                    continue
                if isinstance(message, ScreenshotRequestMessage):
                    self.input_transport.handle_screenshot_request(message)
                    continue
                if isinstance(message, HeartbeatMessage) and message.sender == "mac":
                    with self._condition:
                        previous = self._status.remote_heartbeat_sequence
                        if previous is not None and message.sequence <= previous:
                            raise ProtocolError("Mac heartbeat sequence is not increasing")
                        self._status = replace(
                            self._status,
                            remote_heartbeat_sequence=message.sequence,
                            heartbeat_monotonic=time.monotonic(),
                        )
                    continue
                raise ProtocolError(
                    f"message {type(message).__name__} is invalid from Mac"
                )
        finally:
            self.completion_target.set_completion_sender(None)
            self.input_transport.close_session(opened.session_id)
            self.controller.close_session(opened.session_id)
            with self._condition:
                self._status = replace(
                    self._status,
                    connected=False,
                    session_id=None,
                    remote_heartbeat_sequence=None,
                    heartbeat_monotonic=time.monotonic(),
                )
                self._condition.notify_all()

    def _connect(self) -> Any:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "websockets is required for the Windows bridge"
            ) from exc
        return connect(
            self.url,
            proxy=None,
            compression=None,
            max_size=MAX_MESSAGE_BYTES,
            open_timeout=self.handshake_timeout_seconds,
            ping_interval=10,
            ping_timeout=10,
        )

    def _count_attempt(self) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                connection_attempts=self._status.connection_attempts + 1,
                heartbeat_monotonic=time.monotonic(),
            )

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

    def _record_error(self, error: str, *, protocol: bool = False) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                protocol_errors=(
                    self._status.protocol_errors + (1 if protocol else 0)
                ),
                heartbeat_monotonic=time.monotonic(),
                last_error=error,
            )
