from __future__ import annotations

import unittest

from vrc_ardy_agent.dialogue_turns import DialogueTurnManager
from vrc_ardy_agent.fast_stop import is_fast_stop, normalize_stop_transcript


class FastStopTests(unittest.TestCase):
    def test_normalizes_only_spacing_and_punctuation(self) -> None:
        self.assertEqual(normalize_stop_transcript("  멈, 춰!!!  "), "멈춰")
        self.assertTrue(is_fast_stop("  멈춰! "))

    def test_does_not_promote_synonyms_or_longer_phrases_to_fast_stop(self) -> None:
        for transcript in ("그만", "서", "stop", "멈춰줘", "이제 멈춰"):
            with self.subTest(transcript=transcript):
                self.assertFalse(is_fast_stop(transcript))


class DialogueTurnManagerTests(unittest.TestCase):
    def test_only_latest_turn_can_apply_a_brain_response(self) -> None:
        manager = DialogueTurnManager(id_factory=iter(("turn-a", "turn-b")).__next__)

        first = manager.begin_turn()
        second = manager.begin_turn()

        self.assertFalse(manager.is_current(first))
        self.assertTrue(manager.is_current(second))
        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)

    def test_invalidate_pending_is_idempotent_when_nothing_new_started(self) -> None:
        manager = DialogueTurnManager(id_factory=lambda: "turn-a")
        turn = manager.begin_turn()

        first_version = manager.invalidate_pending(reason="halt_all")
        second_version = manager.invalidate_pending(reason="halt_all")

        self.assertFalse(manager.is_current(turn))
        self.assertEqual(first_version, 2)
        self.assertEqual(second_version, 2)
        self.assertEqual(manager.snapshot().last_invalidation_reason, "halt_all")


if __name__ == "__main__":
    unittest.main()
