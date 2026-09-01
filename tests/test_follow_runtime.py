from __future__ import annotations

import unittest

from vrc_ardy_agent.follow_control import FollowConfig, FollowController, FollowState
from vrc_ardy_agent.follow_protocol import TargetObservation
from vrc_ardy_agent.follow_receiver import LatestObservationStore
from vrc_ardy_agent.follow_runtime import (
    FollowDecisionLoop,
    FollowPromptRouter,
    LatestDecisionStore,
)


class FollowDecisionLoopTests(unittest.TestCase):
    def test_tick_publishes_decision_and_emits_only_real_state_transitions(self):
        observations = LatestObservationStore()
        decisions = LatestDecisionStore()
        emitted = []
        transitions: list[tuple[FollowState, FollowState]] = []
        loop = FollowDecisionLoop(
            observation_store=observations,
            controller=FollowController(
                FollowConfig(smoothing_alpha=1.0, stale_after_seconds=0.3)
            ),
            decision_store=decisions,
            on_decision=emitted.append,
            on_state_change=lambda old, new: transitions.append((old, new)),
            tick_hz=20.0,
        )
        observations.accept(
            TargetObservation(
                session_id="a",
                sequence=1,
                captured_at_ns=1,
                visible=True,
                bbox=(0.4, 0.1, 0.6, 0.3),
                confidence=0.9,
            ),
            received_monotonic=10.0,
        )

        following = loop.tick(now_monotonic=10.0)
        loop.tick(now_monotonic=10.1)
        lost = loop.tick(now_monotonic=10.31)

        self.assertEqual(following.state, FollowState.FOLLOW)
        self.assertEqual(lost.state, FollowState.LOST)
        self.assertEqual(decisions.snapshot().state, FollowState.LOST)
        self.assertEqual(
            transitions,
            [
                (FollowState.LOST, FollowState.FOLLOW),
                (FollowState.FOLLOW, FollowState.LOST),
            ],
        )
        self.assertEqual(loop.status.ticks, 3)
        self.assertEqual(
            [decision.state for decision in emitted],
            [FollowState.FOLLOW, FollowState.FOLLOW, FollowState.LOST],
        )
        self.assertIs(emitted[0], following)
        self.assertIs(emitted[-1], lost)

    def test_stop_emits_neutral_even_when_ardy_has_no_frames(self):
        emitted = []
        loop = FollowDecisionLoop(
            observation_store=LatestObservationStore(),
            controller=FollowController(),
            decision_store=LatestDecisionStore(),
            on_decision=emitted.append,
        )

        loop.stop()

        self.assertEqual(emitted[-1].state, FollowState.LOST)
        self.assertEqual(emitted[-1].vertical, 0.0)
        self.assertEqual(emitted[-1].look_horizontal, 0.0)

    def test_prompt_router_changes_only_between_walking_and_idle_modes(self):
        prompts: list[str] = []
        router = FollowPromptRouter(
            set_prompt=prompts.append,
            initial_prompt="idle",
            walking_prompt="walk",
            idle_prompt="idle",
        )

        router(FollowState.LOST, FollowState.SEARCH)
        router(FollowState.SEARCH, FollowState.RELOCATE)
        router(FollowState.RELOCATE, FollowState.ALIGN)
        router(FollowState.ALIGN, FollowState.FOLLOW)
        router(FollowState.FOLLOW, FollowState.HOLD)
        router(FollowState.HOLD, FollowState.LOST)

        self.assertEqual(prompts, ["walk", "idle"])


if __name__ == "__main__":
    unittest.main()
