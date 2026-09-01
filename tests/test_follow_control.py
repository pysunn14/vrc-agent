from __future__ import annotations

import unittest

from vrc_ardy_agent.follow_control import FollowConfig, FollowController, FollowState
from vrc_ardy_agent.follow_protocol import TargetObservation, TargetSource
from vrc_ardy_agent.follow_receiver import ReceivedObservation


def _received(
    *,
    sequence: int = 1,
    center_x: float = 0.5,
    height: float = 0.3,
    visible: bool = True,
    received_at: float = 10.0,
) -> ReceivedObservation:
    if visible:
        confidence = 0.9
        source = TargetSource.BODY
        observed_center_x = center_x
        proximity = height
    else:
        confidence = 0.0
        source = None
        observed_center_x = None
        proximity = None
    return ReceivedObservation(
        observation=TargetObservation(
            session_id="session-a",
            sequence=sequence,
            captured_at_ns=sequence,
            visible=visible,
            source=source,
            center_x=observed_center_x,
            proximity=proximity,
            confidence=confidence,
        ),
        received_monotonic=received_at,
    )


class FollowControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = FollowController(
            FollowConfig(
                stale_after_seconds=0.3,
                align_enter_error=0.2,
                align_exit_error=0.1,
                resume_follow_below_height=0.4,
                hold_above_height=0.5,
                turn_gain=1.0,
                max_turn=0.6,
                forward_gain=1.0,
                max_forward=0.5,
                smoothing_alpha=1.0,
            )
        )

    def test_missing_invisible_or_stale_target_is_lost_and_neutral(self):
        cases = (
            None,
            _received(visible=False),
            _received(received_at=9.0),
        )

        for received in cases:
            with self.subTest(received=received):
                decision = self.controller.step(received, now_monotonic=10.0)
                self.assertEqual(decision.state, FollowState.LOST)
                self.assertEqual(decision.vertical, 0.0)
                self.assertEqual(decision.look_horizontal, 0.0)

    def test_prolonged_loss_scans_then_relocates_before_scanning_again(self):
        controller = FollowController(
            FollowConfig(
                smoothing_alpha=1.0,
                search_delay_seconds=0.5,
                search_sweep_seconds=4.0,
                search_turn=0.3,
                relocate_turn_seconds=1.0,
                relocate_forward_seconds=1.0,
                relocate_forward=0.2,
            )
        )

        just_lost = controller.step(None, now_monotonic=10.0)
        grace = controller.step(None, now_monotonic=10.49)
        scan_right = controller.step(None, now_monotonic=10.5)
        keep_scanning_right = controller.step(None, now_monotonic=11.6)
        turn_away = controller.step(None, now_monotonic=14.6)
        step_away = controller.step(None, now_monotonic=15.6)
        next_scan = controller.step(None, now_monotonic=16.6)

        self.assertEqual(just_lost.state, FollowState.LOST)
        self.assertEqual(grace.state, FollowState.LOST)
        self.assertEqual(scan_right.state, FollowState.SEARCH)
        self.assertEqual(scan_right.look_horizontal, 0.3)
        self.assertEqual(scan_right.vertical, 0.0)
        self.assertEqual(keep_scanning_right.state, FollowState.SEARCH)
        self.assertEqual(keep_scanning_right.look_horizontal, 0.3)
        self.assertEqual(turn_away.state, FollowState.RELOCATE)
        self.assertEqual(turn_away.look_horizontal, 0.3)
        self.assertEqual(turn_away.vertical, 0.0)
        self.assertEqual(step_away.state, FollowState.RELOCATE)
        self.assertEqual(step_away.look_horizontal, 0.0)
        self.assertEqual(step_away.vertical, 0.2)
        self.assertEqual(next_scan.state, FollowState.SEARCH)
        self.assertEqual(next_scan.look_horizontal, -0.3)

    def test_reacquired_target_preempts_search_and_resets_loss_timer(self):
        controller = FollowController(
            FollowConfig(
                smoothing_alpha=1.0,
                search_delay_seconds=0.5,
                search_sweep_seconds=4.0,
                search_turn=0.3,
                relocate_turn_seconds=1.0,
                relocate_forward_seconds=1.0,
                relocate_forward=0.2,
            )
        )

        controller.step(None, now_monotonic=10.0)
        searching = controller.step(None, now_monotonic=10.6)
        reacquired = controller.step(
            _received(center_x=0.5, height=0.2, received_at=10.7),
            now_monotonic=10.7,
        )
        lost_again = controller.step(None, now_monotonic=10.8)
        new_grace = controller.step(None, now_monotonic=11.2)

        self.assertEqual(searching.state, FollowState.SEARCH)
        self.assertEqual(reacquired.state, FollowState.FOLLOW)
        self.assertEqual(lost_again.state, FollowState.LOST)
        self.assertEqual(new_grace.state, FollowState.LOST)

    def test_active_search_can_be_disabled_for_waiting_intent(self):
        controller = FollowController(FollowConfig(active_search=False))

        controller.step(None, now_monotonic=10.0)
        decision = controller.step(None, now_monotonic=100.0)

        self.assertEqual(decision.state, FollowState.LOST)
        self.assertEqual(decision.vertical, 0.0)
        self.assertEqual(decision.look_horizontal, 0.0)

    def test_off_center_target_turns_without_moving_forward(self):
        decision = self.controller.step(
            _received(center_x=0.25, height=0.2),
            now_monotonic=10.0,
        )

        self.assertEqual(decision.state, FollowState.ALIGN)
        self.assertLess(decision.look_horizontal, 0.0)
        self.assertEqual(decision.vertical, 0.0)

    def test_centered_far_target_moves_forward(self):
        decision = self.controller.step(
            _received(center_x=0.5, height=0.2),
            now_monotonic=10.0,
        )

        self.assertEqual(decision.state, FollowState.FOLLOW)
        self.assertGreater(decision.vertical, 0.0)
        self.assertEqual(decision.look_horizontal, 0.0)

    def test_centered_near_target_holds(self):
        decision = self.controller.step(
            _received(center_x=0.5, height=0.6),
            now_monotonic=10.0,
        )

        self.assertEqual(decision.state, FollowState.HOLD)
        self.assertEqual(decision.vertical, 0.0)

    def test_alignment_and_distance_use_hysteresis(self):
        first = self.controller.step(
            _received(sequence=1, center_x=0.7, height=0.2),
            now_monotonic=10.0,
        )
        still_aligning = self.controller.step(
            _received(sequence=2, center_x=0.56, height=0.2, received_at=10.1),
            now_monotonic=10.1,
        )
        following = self.controller.step(
            _received(sequence=3, center_x=0.52, height=0.2, received_at=10.2),
            now_monotonic=10.2,
        )
        still_following = self.controller.step(
            _received(sequence=4, center_x=0.5, height=0.45, received_at=10.3),
            now_monotonic=10.3,
        )
        holding = self.controller.step(
            _received(sequence=5, center_x=0.5, height=0.51, received_at=10.4),
            now_monotonic=10.4,
        )
        still_holding = self.controller.step(
            _received(sequence=6, center_x=0.5, height=0.45, received_at=10.5),
            now_monotonic=10.5,
        )

        self.assertEqual(first.state, FollowState.ALIGN)
        self.assertEqual(still_aligning.state, FollowState.ALIGN)
        self.assertEqual(following.state, FollowState.FOLLOW)
        self.assertEqual(still_following.state, FollowState.FOLLOW)
        self.assertEqual(holding.state, FollowState.HOLD)
        self.assertEqual(still_holding.state, FollowState.HOLD)


if __name__ == "__main__":
    unittest.main()
