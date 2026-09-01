from __future__ import annotations

import unittest

from scripts.run_live_ardy import build_parser


class LiveArdyCliTests(unittest.TestCase):
    def test_follow_defaults_to_vmt_locomotion(self):
        args = build_parser().parse_args(["--follow"])

        self.assertEqual(args.follow_output, "vmt")
        self.assertEqual(args.follow_turn_pulse_seconds, 0.5)

    def test_follow_can_still_use_vrchat_osc(self):
        args = build_parser().parse_args(
            ["--follow", "--follow-output", "vrchat-osc", "--no-follow-osc-turn-buttons"]
        )

        self.assertEqual(args.follow_output, "vrchat-osc")
        self.assertFalse(args.follow_osc_turn_buttons)


if __name__ == "__main__":
    unittest.main()
