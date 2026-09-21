from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from vrc_ardy_agent.agentctl_companion import run_companion, run_companion_schedule


class _ConcurrentClient:
    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)

    def test_say(self, text: str) -> dict[str, object]:
        self.barrier.wait(timeout=1.0)
        return {"said": text}

    def stop(self, *, reason: str) -> dict[str, object]:
        self.barrier.wait(timeout=1.0)
        return {"stopped": reason}


class AgentctlCompanionScheduleTests(unittest.TestCase):
    def test_screenshot_command_writes_the_current_frame(self) -> None:
        class Client:
            def test_screenshot(self) -> bytes:
                return b"\xff\xd8frame\xff\xd9"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.jpg"
            args = argparse.Namespace(
                companion_command="screenshot",
                url="http://127.0.0.1:8765",
                output=output,
            )
            with patch(
                "vrc_ardy_agent.agentctl_companion.CompanionApiClient",
                return_value=Client(),
            ):
                result = run_companion(args)

            self.assertEqual(output.read_bytes(), b"\xff\xd8frame\xff\xd9")
            self.assertEqual(result["event"], "screenshot_saved")
            self.assertEqual(result["path"], str(output.resolve()))

    def test_same_deadline_operations_are_executed_concurrently(self) -> None:
        client = _ConcurrentClient()

        result = run_companion_schedule(
            [
                {"at_seconds": 0, "command": "test-say", "text": "hello"},
                {"at_seconds": 0, "command": "stop", "reason": "race"},
            ],
            client_factory=lambda: client,
        )

        self.assertEqual(result["scheduled"], 2)
        rows = result["results"]
        self.assertEqual([row["index"] for row in rows], [0, 1])
        self.assertEqual([row["status"] for row in rows], ["completed", "completed"])
        self.assertEqual(rows[0]["result"], {"said": "hello"})
        self.assertEqual(rows[1]["result"], {"stopped": "race"})

    def test_schedule_rejects_unknown_fields_before_execution(self) -> None:
        calls = []

        with self.assertRaisesRegex(ValueError, "fields"):
            run_companion_schedule(
                [{"at_seconds": 0, "command": "stop", "extra": True}],
                client_factory=lambda: calls.append(True),
            )

        self.assertEqual(calls, [])

    def test_schedule_rejects_a_non_string_command_cleanly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown command"):
            run_companion_schedule(
                [{"at_seconds": 0, "command": []}],
                client_factory=lambda: object(),
            )

    def test_one_failed_operation_does_not_hide_other_results(self) -> None:
        class Client:
            def test_bundle(self, payload: object) -> dict[str, object]:
                del payload
                raise RuntimeError("expected failure")

            def stop(self, *, reason: str) -> dict[str, object]:
                return {"stopped": reason}

        client = Client()
        result = run_companion_schedule(
            [
                {"at_seconds": 0, "command": "test-bundle", "bundle": {}},
                {"at_seconds": 0, "command": "stop"},
            ],
            client_factory=lambda: client,
        )

        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertIn("expected failure", result["results"][0]["error"])
        self.assertEqual(result["results"][1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
