from __future__ import annotations

from contextlib import redirect_stderr
import io
import unittest

from scripts.run_live_ardy import build_parser


class LiveArdyCliTests(unittest.TestCase):
    def test_live_ardy_only_accepts_body_motion_arguments(self):
        args = build_parser().parse_args(["--model", "core", "--device", "mps"])

        self.assertEqual(args.model, "core")
        self.assertEqual(args.device, "mps")
        self.assertEqual(args.avatar_profile, "felis")

    def test_following_is_not_an_ardy_runner_mode(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--follow"])

    def test_history_frames_can_be_reduced_for_fast_prompt_adaptation(self):
        args = build_parser().parse_args(["--history-frames", "4"])

        self.assertEqual(args.history_frames, 4)


if __name__ == "__main__":
    unittest.main()
