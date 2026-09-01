from __future__ import annotations

import unittest

from scripts.agentctl import build_parser


class AgentctlCliTests(unittest.TestCase):
    def test_nameplate_probe_requires_target_name(self):
        args = build_parser().parse_args(
            [
                "windows",
                "nameplate",
                "--hwnd",
                "123",
                "--target-name",
                "TargetUser 28",
            ]
        )

        self.assertEqual(args.target_name, "TargetUser 28")
        self.assertEqual(args.nameplate_input_width, 960)

    def test_follow_injection_accepts_semantic_target_measurement(self):
        args = build_parser().parse_args(
            [
                "follow",
                "inject",
                "--center-x",
                "0.4",
                "--proximity",
                "0.2",
                "--source",
                "nameplate",
            ]
        )

        self.assertEqual(args.center_x, 0.4)
        self.assertEqual(args.proximity, 0.2)
        self.assertEqual(args.source, "nameplate")


if __name__ == "__main__":
    unittest.main()
