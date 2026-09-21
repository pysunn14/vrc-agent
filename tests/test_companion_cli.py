from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

from scripts.run_companion import build_parser, route_audio
from vrc_ardy_agent.hermes_brain import load_hermes_environment
from vrc_ardy_agent.stream_bridge import RetargetingMode


class CompanionCliTests(unittest.TestCase):
    def test_manual_mode_does_not_trigger_audio_reactions(self):
        from unittest.mock import Mock
        pipeline = Mock()
        route_audio("frame", pipeline=pipeline, manual_actions_only=True)
        pipeline.accept_audio.assert_not_called()
        route_audio("frame", pipeline=pipeline, manual_actions_only=False)
        pipeline.accept_audio.assert_called_once_with("frame")

    def test_runtime_specific_defaults_are_resolved_from_profile(self) -> None:
        args = build_parser().parse_args([])

        self.assertIsNone(args.device)
        self.assertEqual(args.profile.name, "agent.json")
        self.assertEqual(args.pose_ttl_ms, 2000)
        self.assertEqual(args.retargeting_mode, RetargetingMode.FULL.value)

    def test_lower_body_position_only_mode_is_selectable(self) -> None:
        args = build_parser().parse_args(
            ["--retargeting-mode", "lower-body-position-only"]
        )

        self.assertEqual(
            args.retargeting_mode,
            RetargetingMode.LOWER_BODY_POSITION_ONLY.value,
        )

    def test_history_frames_can_be_reduced_for_fast_prompt_adaptation(self) -> None:
        args = build_parser().parse_args(["--history-frames", "4"])

        self.assertEqual(args.history_frames, 4)

    def test_existing_environment_key_wins_over_hermes_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("API_SERVER_KEY=file-key\n", encoding="utf-8")

            environment = load_hermes_environment(
                env_file,
                base_environment={"API_SERVER_KEY": "process-key"},
            )

        self.assertEqual(environment["API_SERVER_KEY"], "process-key")

    def test_hermes_key_can_be_loaded_without_importing_other_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "OTHER_SECRET=do-not-copy\nAPI_SERVER_KEY='hermes-key'\n",
                encoding="utf-8",
            )

            environment = load_hermes_environment(
                env_file,
                base_environment={},
            )

        self.assertEqual(environment, {"API_SERVER_KEY": "hermes-key"})


if __name__ == "__main__":
    unittest.main()
