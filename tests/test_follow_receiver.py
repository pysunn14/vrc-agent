from __future__ import annotations

import socket
import time
import unittest

from vrc_ardy_agent.follow_protocol import (
    TargetObservation,
    TargetSource,
    encode_target_observation,
)
from vrc_ardy_agent.follow_receiver import LatestObservationStore, UdpObservationReceiver


def _observation(*, session: str = "a", sequence: int = 1) -> TargetObservation:
    return TargetObservation(
        session_id=session,
        sequence=sequence,
        captured_at_ns=sequence,
        visible=True,
        source=TargetSource.BODY,
        center_x=0.4,
        proximity=0.8,
        confidence=0.9,
    )


class LatestObservationStoreTests(unittest.TestCase):
    def test_store_rejects_old_packets_and_requires_staleness_for_session_takeover(self):
        store = LatestObservationStore(session_takeover_after_seconds=0.3)

        self.assertTrue(store.accept(_observation(sequence=2), received_monotonic=10.0))
        self.assertFalse(store.accept(_observation(sequence=1), received_monotonic=10.1))
        self.assertFalse(
            store.accept(_observation(session="b", sequence=0), received_monotonic=10.2)
        )
        self.assertTrue(
            store.accept(_observation(session="b", sequence=0), received_monotonic=10.31)
        )
        self.assertEqual(store.snapshot().observation.session_id, "b")  # type: ignore[union-attr]


class UdpObservationReceiverTests(unittest.TestCase):
    def test_receiver_decodes_udp_and_reports_invalid_packets(self):
        receiver = UdpObservationReceiver(bind_host="127.0.0.1", port=0)
        receiver.start()
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(b"invalid", ("127.0.0.1", receiver.bound_port))
            sender.sendto(
                encode_target_observation(_observation(sequence=3)),
                ("127.0.0.1", receiver.bound_port),
            )

            deadline = time.monotonic() + 1.0
            while receiver.store.snapshot() is None and time.monotonic() < deadline:
                time.sleep(0.01)

            latest = receiver.store.snapshot()
            self.assertIsNotNone(latest)
            self.assertEqual(latest.observation.sequence, 3)  # type: ignore[union-attr]
            self.assertEqual(receiver.status.packets_received, 2)
            self.assertEqual(receiver.status.packets_accepted, 1)
            self.assertEqual(receiver.status.invalid_packets, 1)
            self.assertGreater(receiver.status.heartbeat_monotonic, 0.0)
        finally:
            sender.close()
            receiver.stop()

        self.assertFalse(receiver.status.running)


if __name__ == "__main__":
    unittest.main()
