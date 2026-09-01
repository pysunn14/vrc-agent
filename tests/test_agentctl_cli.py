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

    def test_follow_injection_can_simulate_an_active_identity_scan(self):
        args = build_parser().parse_args(
            ["follow", "inject", "--identity-scan-active"]
        )

        self.assertTrue(args.identity_scan_active)

    def test_track_accepts_window_process_id(self):
        args = build_parser().parse_args(
            [
                "windows",
                "track",
                "--window-process-id",
                "22044",
                "--mac-host",
                "100.64.0.1",
            ]
        )

        self.assertEqual(args.window_process_id, 22044)


if __name__ == "__main__":
    unittest.main()
