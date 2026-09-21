from __future__ import annotations

import unittest

from vrc_ardy_agent.action_contracts import (
    ActionBundle,
    ArdyMotionAction,
    PlanValidationError,
    SayAction,
    validate_action_bundle,
)


class ActionContractTests(unittest.TestCase):
    def test_accepts_one_speech_and_one_motion_action(self) -> None:
        bundle = validate_action_bundle(
            {
                "actions": [
                    {"type": "say", "text": "응, 잠깐 춰볼게."},
                    {
                        "type": "ardy_motion",
                        "prompt": "A girl performs a short playful dance.",
                        "duration_seconds": 6,
                    },
                ]
            }
        )

        self.assertEqual(
            bundle,
            ActionBundle(
                actions=(
                    SayAction(text="응, 잠깐 춰볼게."),
                    ArdyMotionAction(
                        prompt="A girl performs a short playful dance.",
                        duration_seconds=6.0,
                    ),
                )
            ),
        )

    def test_rejects_duplicate_action_resources(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "duplicate say"):
            validate_action_bundle(
                {
                    "actions": [
                        {"type": "say", "text": "첫 번째."},
                        {"type": "say", "text": "두 번째."},
                    ]
                }
            )

    def test_rejects_unknown_or_extra_fields_instead_of_recovering_them(self) -> None:
        invalid_payloads = (
            {"actions": [{"type": "wave"}]},
            {"actions": [{"type": "say", "text": "안녕."}], "intent": "greet"},
            {"actions": [{"type": "say", "text": "안녕.", "priority": 9}]},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(PlanValidationError):
                    validate_action_bundle(payload)

    def test_motion_duration_is_bounded_and_prompt_must_be_english(self) -> None:
        invalid_actions = (
            {
                "type": "ardy_motion",
                "prompt": "A girl dances.",
                "duration_seconds": 0.9,
            },
            {
                "type": "ardy_motion",
                "prompt": "A girl dances.",
                "duration_seconds": 10.1,
            },
            {
                "type": "ardy_motion",
                "prompt": "소녀가 춤을 춘다.",
                "duration_seconds": 3,
            },
        )

        for action in invalid_actions:
            with self.subTest(action=action):
                with self.assertRaises(PlanValidationError):
                    validate_action_bundle({"actions": [action]})

    def test_speech_is_limited_to_two_short_sentences(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "two sentences"):
            validate_action_bundle(
                {
                    "actions": [
                        {
                            "type": "say",
                            "text": "좋아. 바로 할게. 잠깐 기다려.",
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
