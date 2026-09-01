from __future__ import annotations

import unittest

from scripts.run_windows_perception import build_parser


class WindowsPerceptionCliTests(unittest.TestCase):
    def test_run_accepts_window_title_instead_of_hwnd(self):
        args = build_parser().parse_args(
            ["run", "--window-title", "VRChat", "--mac-host", "127.0.0.1"]
        )

        self.assertIsNone(args.hwnd)
        self.assertEqual(args.window_title, "VRChat")

    def test_run_accepts_nameplate_target_configuration(self):
        args = build_parser().parse_args(
            [
                "run",
                "--hwnd",
                "123",
                "--mac-host",
                "127.0.0.1",
                "--target-name",
                "TargetUser 28",
            ]
        )

        self.assertEqual(args.target_name, "TargetUser 28")
        self.assertEqual(args.nameplate_scan_interval, 5.0)
        self.assertEqual(args.nameplate_anchor_max_age, 120.0)

    def test_run_requires_exactly_one_window_selector(self):
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "--mac-host", "127.0.0.1"])
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run",
                    "--hwnd",
                    "123",
                    "--window-title",
                    "VRChat",
                    "--mac-host",
                    "127.0.0.1",
                ]
            )


if __name__ == "__main__":
    unittest.main()
