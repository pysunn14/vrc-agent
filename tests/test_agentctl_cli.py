from __future__ import annotations
from tests.rig_fixture import RIG_PATH

import unittest

from scripts.agentctl import _build_neutral_pose, build_parser


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

    def test_pose_show_uses_explicit_profile_body_trackers(self):
        args = build_parser().parse_args(["pose", "show", "--avatar-profile", RIG_PATH])

        self.assertEqual(args.pose_command, "show")
        self.assertEqual(args.avatar_profile, RIG_PATH)
        self.assertEqual(args.hands, "none")
        self.assertEqual(args.body_trackers, "profile")
        self.assertEqual(_build_neutral_pose(args).body_enable, 7)

    def test_pose_hold_accepts_one_controller_for_exploration(self):
        args = build_parser().parse_args(
            ["pose", "hold", "--host", "192.0.2.10", "--hands", "left"]
        )

        self.assertEqual(args.pose_command, "hold")
        self.assertEqual(args.host, "192.0.2.10")
        self.assertEqual(args.hands, "left")

    def test_pose_hold_can_explicitly_disable_profile_body_trackers(self):
        args = build_parser().parse_args(
            [
                "pose",
                "hold",
                "--host",
                "192.0.2.10",
                "--body-trackers",
                "off",
                "--avatar-profile", RIG_PATH,
            ]
        )

        self.assertEqual(_build_neutral_pose(args).body_enable, 0)

    def test_companion_commands_target_the_runtime_rest_boundary(self):
        status = build_parser().parse_args(
            ["companion", "status", "--url", "http://127.0.0.1:8765"]
        )
        motion = build_parser().parse_args(
            [
                "companion",
                "test-motion",
                "--prompt",
                "A girl waves.",
                "--duration",
                "3",
            ]
        )

        self.assertEqual(status.companion_command, "status")
        self.assertEqual(status.url, "http://127.0.0.1:8765")
        self.assertEqual(motion.companion_command, "test-motion")
        self.assertEqual(motion.duration, 3.0)

    def test_companion_schedule_accepts_a_json_timeline(self):
        args = build_parser().parse_args(
            [
                "companion",
                "schedule",
                "--payload",
                '[{"at_seconds":0,"command":"stop"}]',
            ]
        )

        self.assertEqual(args.companion_command, "schedule")
        self.assertIn('"command":"stop"', args.payload)


if __name__ == "__main__":
    unittest.main()
