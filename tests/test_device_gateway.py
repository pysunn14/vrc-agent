from __future__ import annotations

import unittest

from vrc_ardy_agent.action_contracts import (
    ControlResource,
    OutputEnvelope,
    ResourceLease,
)
from vrc_ardy_agent.device_gateway import DeviceGateway, GatewayDropReason


class _RecordingDeviceSink:
    def __init__(self) -> None:
        self.applied: list[OutputEnvelope] = []
        self.neutralized: list[ControlResource] = []

    def apply(self, envelope: OutputEnvelope) -> None:
        self.applied.append(envelope)

    def neutralize(self, resource: ControlResource) -> None:
        self.neutralized.append(resource)


def _envelope(
    *,
    session_id: str = "session-a",
    action_id: str = "motion-a",
    lease_token: int = 1,
    sequence: int = 1,
    ttl_ms: int = 100,
) -> OutputEnvelope:
    return OutputEnvelope(
        session_id=session_id,
        action_id=action_id,
        resource=ControlResource.FULL_BODY_POSE,
        lease_token=lease_token,
        sequence=sequence,
        ttl_ms=ttl_ms,
        payload={"frame": sequence},
    )


class DeviceGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sink = _RecordingDeviceSink()
        self.gateway = DeviceGateway(sink=self.sink, monotonic=lambda: 10.0)
        self.gateway.open_session("session-a")
        self.assertTrue(
            self.gateway.authorize(
                "session-a",
                ResourceLease(
                    resource=ControlResource.FULL_BODY_POSE,
                    token=1,
                    action_id="motion-a",
                ),
            ).accepted
        )

    def test_accepts_only_current_session_lease_and_increasing_sequence(self) -> None:
        accepted = self.gateway.apply_envelope(
            _envelope(sequence=2),
            received_monotonic=10.0,
            now_monotonic=10.01,
        )
        duplicate = self.gateway.apply_envelope(
            _envelope(sequence=2),
            received_monotonic=10.02,
            now_monotonic=10.02,
        )
        old_session = self.gateway.apply_envelope(
            _envelope(session_id="session-old", sequence=3),
            received_monotonic=10.03,
            now_monotonic=10.03,
        )
        old_lease = self.gateway.apply_envelope(
            _envelope(lease_token=0, sequence=3),
            received_monotonic=10.04,
            now_monotonic=10.04,
        )

        self.assertTrue(accepted.accepted)
        self.assertEqual(duplicate.reason, GatewayDropReason.SEQUENCE)
        self.assertEqual(old_session.reason, GatewayDropReason.SESSION)
        self.assertEqual(old_lease.reason, GatewayDropReason.LEASE)
        self.assertEqual(self.sink.applied, [_envelope(sequence=2)])

    def test_expired_queued_message_is_rejected(self) -> None:
        decision = self.gateway.apply_envelope(
            _envelope(ttl_ms=50),
            received_monotonic=10.0,
            now_monotonic=10.051,
        )

        self.assertEqual(decision.reason, GatewayDropReason.EXPIRED)
        self.assertEqual(self.sink.applied, [])

    def test_watchdog_neutralizes_and_revokes_timed_out_lease(self) -> None:
        self.assertTrue(
            self.gateway.apply_envelope(
                _envelope(ttl_ms=50),
                received_monotonic=10.0,
                now_monotonic=10.0,
            ).accepted
        )

        expired = self.gateway.poll_expired(now_monotonic=10.051)
        late = self.gateway.apply_envelope(
            _envelope(sequence=2, ttl_ms=50),
            received_monotonic=10.052,
            now_monotonic=10.052,
        )

        self.assertEqual(expired, (ControlResource.FULL_BODY_POSE,))
        self.assertIn(ControlResource.FULL_BODY_POSE, self.sink.neutralized)
        self.assertEqual(late.reason, GatewayDropReason.LEASE)

    def test_explicit_neutralize_fences_late_frames_before_they_apply(self) -> None:
        decision = self.gateway.neutralize(
            "session-a",
            ControlResource.FULL_BODY_POSE,
            lease_token=2,
        )
        late = self.gateway.apply_envelope(
            _envelope(sequence=1),
            received_monotonic=10.0,
            now_monotonic=10.0,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(late.reason, GatewayDropReason.LEASE)
        self.assertEqual(
            self.gateway.snapshot()
            .resource(ControlResource.FULL_BODY_POSE)
            .lease_token,
            2,
        )

    def test_new_session_neutralizes_old_outputs_and_rejects_old_session(self) -> None:
        self.gateway.apply_envelope(
            _envelope(),
            received_monotonic=10.0,
            now_monotonic=10.0,
        )

        self.gateway.open_session("session-b")
        old = self.gateway.apply_envelope(
            _envelope(sequence=2),
            received_monotonic=10.01,
            now_monotonic=10.01,
        )

        self.assertIn(ControlResource.FULL_BODY_POSE, self.sink.neutralized)
        self.assertEqual(old.reason, GatewayDropReason.SESSION)
        self.assertEqual(self.gateway.snapshot().session_id, "session-b")

    def test_neutralize_failure_is_not_acknowledged_as_success(self) -> None:
        class FailingSink(_RecordingDeviceSink):
            def neutralize(self, resource: ControlResource) -> None:
                raise RuntimeError("device stuck")

        gateway = DeviceGateway(sink=FailingSink(), monotonic=lambda: 10.0)
        gateway.open_session("session-a")

        decision = gateway.neutralize(
            "session-a",
            ControlResource.FULL_BODY_POSE,
            lease_token=1,
        )

        self.assertEqual(decision.reason, GatewayDropReason.SINK_ERROR)
        self.assertIn("device stuck", gateway.snapshot().last_errors[-1])


if __name__ == "__main__":
    unittest.main()
