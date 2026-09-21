from __future__ import annotations

import socket
import time
import unittest

from vrc_ardy_agent.action_contracts import ControlResource, ResourceLease
from vrc_ardy_agent.device_gateway import DeviceGateway
from vrc_ardy_agent.mac_bridge_server import MacBridgeServer
from vrc_ardy_agent.mac_input_transport import MacInputTransport
from vrc_ardy_agent.mac_output_transport import MacOutputTransport
from vrc_ardy_agent.windows_bridge_client import WindowsBridgeClient
from vrc_ardy_agent.windows_input_transport import WindowsInputTransport
from vrc_ardy_agent.windows_output_controller import WindowsOutputController


class _Sink:
    def __init__(self) -> None:
        self.neutralized = []

    def apply(self, envelope) -> None:
        pass

    def neutralize(self, resource) -> None:
        self.neutralized.append(resource)


class _CompletionTarget:
    def __init__(self) -> None:
        self.sender = None

    def set_completion_sender(self, sender) -> None:
        self.sender = sender


def _unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until(predicate, *, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class WindowsBridgeClientTests(unittest.TestCase):
    def test_reconnect_opens_fresh_session_without_replaying_old_action(self) -> None:
        port = _unused_local_port()
        output = MacOutputTransport(control_timeout_seconds=1.0)
        input_transport = MacInputTransport(audio_handler=lambda _chunk: None)
        sessions = iter(["session-a", "session-b"])
        gateway = DeviceGateway(sink=_Sink())
        controller = WindowsOutputController(gateway=gateway)
        completions = _CompletionTarget()
        windows_input = WindowsInputTransport(
            screenshot_provider=lambda: b"\xff\xd8fresh\xff\xd9"
        )
        client = WindowsBridgeClient(
            url=f"ws://127.0.0.1:{port}",
            controller=controller,
            completion_target=completions,
            input_transport=windows_input,
            client_instance_id="windows-test",
            reconnect_delay_seconds=0.02,
            receive_poll_seconds=0.02,
            heartbeat_interval_seconds=0.02,
        )
        first_server = MacBridgeServer(
            output=output,
            input_transport=input_transport,
            host="127.0.0.1",
            port=port,
            heartbeat_interval_seconds=0.02,
            session_id_factory=sessions.__next__,
        )
        first_server.start()
        client.start()
        self.addCleanup(client.stop)
        self.addCleanup(first_server.stop)

        self.assertTrue(client.wait_until_connected(timeout=1.0))
        self.assertTrue(
            _wait_until(
                lambda: input_transport.snapshot().remote_heartbeat_sequence
                is not None
            )
        )
        self.assertTrue(
            _wait_until(
                lambda: client.snapshot().remote_heartbeat_sequence is not None
            )
        )
        output.authorize(
            ResourceLease(
                resource=ControlResource.FULL_BODY_POSE,
                token=1,
                action_id="old-motion",
            )
        )
        self.assertEqual(
            gateway.snapshot().resource(ControlResource.FULL_BODY_POSE).action_id,
            "old-motion",
        )
        self.assertEqual(
            input_transport.request_screenshot(),
            b"\xff\xd8fresh\xff\xd9",
        )

        first_server.stop()
        self.assertTrue(_wait_until(lambda: not client.snapshot().connected))
        self.assertIsNone(gateway.snapshot().session_id)
        self.assertIsNone(
            gateway.snapshot().resource(ControlResource.FULL_BODY_POSE).action_id
        )

        second_server = MacBridgeServer(
            output=output,
            input_transport=input_transport,
            host="127.0.0.1",
            port=port,
            heartbeat_interval_seconds=0.02,
            session_id_factory=sessions.__next__,
        )
        second_server.start()
        self.addCleanup(second_server.stop)

        self.assertTrue(client.wait_until_connected(timeout=2.0))
        snapshot = client.snapshot()
        self.assertEqual(snapshot.session_id, "session-b")
        self.assertEqual(snapshot.connections_opened, 2)
        self.assertGreaterEqual(snapshot.connection_attempts, 2)
        self.assertIsNotNone(completions.sender)
        self.assertEqual(gateway.snapshot().session_id, "session-b")
        self.assertIsNone(
            gateway.snapshot().resource(ControlResource.FULL_BODY_POSE).action_id
        )
        self.assertEqual(output.snapshot().messages_sent, 1)


if __name__ == "__main__":
    unittest.main()
