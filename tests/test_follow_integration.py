from __future__ import annotations

import time
import unittest

from vrc_ardy_agent.follow_control import FollowConfig, FollowController, FollowState
from vrc_ardy_agent.follow_protocol import TargetObservation, TargetSource
from vrc_ardy_agent.follow_receiver import UdpObservationReceiver
from vrc_ardy_agent.follow_runtime import FollowDecisionLoop, LatestDecisionStore
from vrc_ardy_agent.windows_perception import UdpObservationSender


class FollowUdpIntegrationTests(unittest.TestCase):
    def test_udp_observation_reaches_controller_and_stale_state_stops_motion(self):
        receiver = UdpObservationReceiver(bind_host="127.0.0.1", port=0)
        receiver.start()
        sender = UdpObservationSender(host="127.0.0.1", port=receiver.bound_port)
        decisions = LatestDecisionStore()
        loop = FollowDecisionLoop(
            observation_store=receiver.store,
            controller=FollowController(
                FollowConfig(stale_after_seconds=0.3, smoothing_alpha=1.0)
            ),
            decision_store=decisions,
        )
        try:
            sender.send(
                TargetObservation(
                    session_id="windows-test",
                    sequence=0,
                    captured_at_ns=time.perf_counter_ns(),
                    visible=True,
                    source=TargetSource.BODY,
                    center_x=0.5,
                    proximity=0.2,
                    confidence=0.9,
                )
            )
            deadline = time.monotonic() + 1.0
            while receiver.store.snapshot() is None and time.monotonic() < deadline:
                time.sleep(0.005)
            received = receiver.store.snapshot()
            self.assertIsNotNone(received)
            assert received is not None

            following = loop.tick(now_monotonic=received.received_monotonic)
            stopped = loop.tick(now_monotonic=received.received_monotonic + 0.31)

            self.assertEqual(following.state, FollowState.FOLLOW)
            self.assertGreater(following.vertical, 0.0)
            self.assertEqual(stopped.state, FollowState.LOST)
            self.assertEqual(stopped.vertical, 0.0)
            self.assertEqual(stopped.look_horizontal, 0.0)
        finally:
            sender.close()
            receiver.stop()


if __name__ == "__main__":
    unittest.main()
