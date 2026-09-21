from __future__ import annotations

import unittest

from vrc_ardy_agent.action_contracts import (
    ControlResource,
    OutputEnvelope,
    ResourceLease,
)
from vrc_ardy_agent.device_gateway import DeviceGateway, GatewayDropReason
from vrc_ardy_agent.stream_protocol import (
    AckMessage,
    AuthorizeMessage,
    OutputDataMessage,
    OutputNeutralizeMessage,
    SessionOpenMessage,
)
from vrc_ardy_agent.windows_output_controller import (
    OutputDataRejectedError,
    WindowsOutputController,
)


class _Sink:
    def __init__(self) -> None:
        self.applied = []
        self.neutralized = []

    def apply(self, envelope) -> None:
        self.applied.append(envelope)

    def neutralize(self, resource) -> None:
        self.neutralized.append(resource)


class WindowsOutputControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sink = _Sink()
        self.gateway = DeviceGateway(sink=self.sink)
        self.controller = WindowsOutputController(gateway=self.gateway)
        self.controller.open_session(SessionOpenMessage(session_id="session-a"))

    def test_authorize_data_and_neutralize_use_the_device_gateway(self) -> None:
        authorize = self.controller.handle(
            AuthorizeMessage(
                message_id="request-1",
                session_id="session-a",
                lease=ResourceLease(
                    resource=ControlResource.FULL_BODY_POSE,
                    token=2,
                    action_id="motion-a",
                ),
            )
        )
        envelope = OutputEnvelope(
            session_id="session-a",
            action_id="motion-a",
            resource=ControlResource.FULL_BODY_POSE,
            lease_token=2,
            sequence=0,
            ttl_ms=250,
            payload={"kind": "pose"},
        )
        data_response = self.controller.handle(OutputDataMessage(envelope=envelope))
        neutralize = self.controller.handle(
            OutputNeutralizeMessage(
                message_id="request-2",
                session_id="session-a",
                resource=ControlResource.FULL_BODY_POSE,
                lease_token=3,
            )
        )

        self.assertEqual(
            authorize,
            AckMessage(message_id="request-1", accepted=True, reason=None),
        )
        self.assertIsNone(data_response)
        self.assertEqual(self.sink.applied, [envelope])
        self.assertEqual(
            neutralize,
            AckMessage(message_id="request-2", accepted=True, reason=None),
        )
        self.assertIn(ControlResource.FULL_BODY_POSE, self.sink.neutralized)

    def test_stale_session_control_is_rejected_with_reason(self) -> None:
        result = self.controller.handle(
            AuthorizeMessage(
                message_id="request-old",
                session_id="session-old",
                lease=ResourceLease(
                    resource=ControlResource.VOICE_OUTPUT,
                    token=1,
                    action_id="speech-old",
                ),
            )
        )

        self.assertEqual(
            result,
            AckMessage(
                message_id="request-old",
                accepted=False,
                reason=GatewayDropReason.SESSION.value,
            ),
        )

    def test_rejected_pose_data_is_not_silently_ignored(self) -> None:
        envelope = OutputEnvelope(
            session_id="session-a",
            action_id="stale-motion",
            resource=ControlResource.FULL_BODY_POSE,
            lease_token=1,
            sequence=0,
            ttl_ms=2000,
            payload={"kind": "pose"},
        )

        with self.assertRaisesRegex(OutputDataRejectedError, "lease"):
            self.controller.handle(OutputDataMessage(envelope=envelope))

    def test_close_session_neutralizes_all_outputs(self) -> None:
        self.assertTrue(self.controller.close_session("session-a"))

        self.assertIsNone(self.gateway.snapshot().session_id)
        self.assertEqual(set(self.sink.neutralized), set(ControlResource))


if __name__ == "__main__":
    unittest.main()
