from __future__ import annotations

import unittest

from vrc_ardy_agent.live_control import IDLE_PROMPT, apply_control_line


class _FakeSession:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.pause_calls = 0
        self.stop_requests = 0

    def set_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)

    def pause(self) -> None:
        self.pause_calls += 1

    def request_stop(self) -> None:
        self.stop_requests += 1


class LiveControlTests(unittest.TestCase):
    def test_plain_text_becomes_a_prompt(self):
        session = _FakeSession()

        result = apply_control_line(session, "wave with the right hand")

        self.assertEqual(session.prompts, ["wave with the right hand"])
        self.assertEqual(result.action, "prompt")
        self.assertFalse(result.quit_requested)

    def test_idle_maps_to_natural_standing_prompt(self):
        session = _FakeSession()

        result = apply_control_line(session, "idle")

        self.assertEqual(session.prompts, [IDLE_PROMPT])
        self.assertEqual(result.action, "idle")

    def test_stop_pauses_without_terminating_runtime(self):
        session = _FakeSession()

        result = apply_control_line(session, "stop")

        self.assertEqual(session.pause_calls, 1)
        self.assertEqual(session.stop_requests, 0)
        self.assertEqual(result.action, "stop")
        self.assertFalse(result.quit_requested)

    def test_quit_requests_session_shutdown(self):
        session = _FakeSession()

        result = apply_control_line(session, "quit")

        self.assertEqual(session.stop_requests, 1)
        self.assertTrue(result.quit_requested)


if __name__ == "__main__":
    unittest.main()
