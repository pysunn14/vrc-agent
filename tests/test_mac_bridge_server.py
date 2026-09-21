from __future__ import annotations

import threading
import unittest
from dataclasses import replace

from websockets.sync.client import connect

from vrc_ardy_agent.action_contracts import ControlResource, ResourceLease
from vrc_ardy_agent.mac_bridge_server import MacBridgeServer
from vrc_ardy_agent.mac_input_transport import MacInputTransport
from vrc_ardy_agent.mac_output_transport import MacOutputTransport
from vrc_ardy_agent.stream_protocol import (
    AckMessage,
    AudioChunkMessage,
    AuthorizeMessage,
    HelloMessage,
    HeartbeatMessage,
    SessionOpenMessage,
    decode_message,
    encode_message,
)


class MacBridgeServerTests(unittest.TestCase):
    def test_stop_reports_a_server_thread_that_did_not_exit(self) -> None:
        class StuckServer:
            def __init__(self) -> None:
                self.shutdown_called = False

            def shutdown(self) -> None:
                self.shutdown_called = True

        class StuckThread:
            def join(self, timeout: float | None = None) -> None:
                del timeout

            def is_alive(self) -> bool:
                return True

        output = MacOutputTransport()
        input_transport = MacInputTransport(audio_handler=lambda _chunk: None)
        server = MacBridgeServer(
            output=output,
            input_transport=input_transport,
            host="127.0.0.1",
            port=0,
        )
        stuck_server = StuckServer()
        server._server = stuck_server
        server._server_thread = StuckThread()  # type: ignore[assignment]
        server._status = replace(server.snapshot(), running=True)

        with self.assertRaisesRegex(RuntimeError, "did not stop"):
            server.stop()

        self.assertTrue(stuck_server.shutdown_called)
        status = server.snapshot()
        self.assertTrue(status.running)
        self.assertIn("did not stop", status.last_error or "")

    def test_real_websocket_handshake_and_authorize_ack(self) -> None:
        disconnected = []
        lifecycle = []
        opened_event = threading.Event()

        def opened(session_id: str) -> None:
            lifecycle.append(("opened", session_id))
            opened_event.set()

        def closed(session_id: str, reason: str) -> None:
            lifecycle.append(("closed", session_id, reason))

        ids = iter(["control-1"])
        output = MacOutputTransport(
            message_id_factory=ids.__next__,
            disconnected=disconnected.append,
        )
        audio = []
        audio_received = threading.Event()

        def receive_audio(chunk) -> None:
            audio.append(chunk)
            audio_received.set()

        input_transport = MacInputTransport(audio_handler=receive_audio)
        sessions = iter(["session-1"])
        server = MacBridgeServer(
            output=output,
            input_transport=input_transport,
            host="127.0.0.1",
            port=0,
            session_id_factory=sessions.__next__,
            on_session_opened=opened,
            on_session_closed=closed,
        )
        server.start()
        self.addCleanup(server.stop)

        with connect(
            f"ws://127.0.0.1:{server.bound_port}",
            proxy=None,
        ) as websocket:
            websocket.send(encode_message(HelloMessage("windows-test")))
            opened = decode_message(websocket.recv())
            self.assertEqual(opened, SessionOpenMessage("session-1"))
            self.assertTrue(opened_event.wait(timeout=1.0))

            errors = []

            def authorize() -> None:
                try:
                    output.authorize(
                        ResourceLease(
                            resource=ControlResource.FULL_BODY_POSE,
                            token=1,
                            action_id="motion-1",
                        )
                    )
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=authorize)
            thread.start()
            request = decode_message(websocket.recv(timeout=1.0))
            self.assertEqual(
                request,
                AuthorizeMessage(
                    message_id="control-1",
                    session_id="session-1",
                    lease=ResourceLease(
                        resource=ControlResource.FULL_BODY_POSE,
                        token=1,
                        action_id="motion-1",
                    ),
                ),
            )
            websocket.send(
                encode_message(
                    AckMessage(
                        message_id="control-1",
                        accepted=True,
                        reason=None,
                    )
                )
            )
            websocket.send(
                encode_message(
                    AudioChunkMessage(
                        session_id="session-1",
                        sequence=0,
                        captured_monotonic_ns=1,
                        pcm=b"\x00\x00",
                    )
                )
            )
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(audio_received.wait(timeout=1.0))
            self.assertEqual([chunk.sequence for chunk in audio], [0])

        self.assertTrue(server.wait_until_disconnected(timeout=1.0))
        status = server.snapshot()
        self.assertTrue(status.running)
        self.assertFalse(status.connected)
        self.assertEqual(status.connections_opened, 1)
        self.assertGreaterEqual(status.messages_received, 2)
        self.assertEqual(len(disconnected), 1)
        self.assertEqual(lifecycle[0], ("opened", "session-1"))
        self.assertEqual(lifecycle[1][:2], ("closed", "session-1"))

    def test_server_emits_application_heartbeat_while_idle(self) -> None:
        output = MacOutputTransport()
        input_transport = MacInputTransport(audio_handler=lambda _chunk: None)
        server = MacBridgeServer(
            output=output,
            input_transport=input_transport,
            host="127.0.0.1",
            port=0,
            heartbeat_interval_seconds=0.02,
            session_id_factory=lambda: "session-heartbeat",
        )
        server.start()
        self.addCleanup(server.stop)

        with connect(
            f"ws://127.0.0.1:{server.bound_port}",
            proxy=None,
        ) as websocket:
            websocket.send(encode_message(HelloMessage("windows-test")))
            self.assertIsInstance(decode_message(websocket.recv()), SessionOpenMessage)

            heartbeat = decode_message(websocket.recv(timeout=1.0))

        self.assertEqual(
            heartbeat,
            HeartbeatMessage(
                session_id="session-heartbeat",
                sender="mac",
                sequence=0,
            ),
        )
        self.assertGreaterEqual(server.snapshot().messages_sent, 2)


if __name__ == "__main__":
    unittest.main()
