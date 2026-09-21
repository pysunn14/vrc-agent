from __future__ import annotations

import unittest

from scripts.run_follow import build_parser, validate_args


class FollowCliTests(unittest.TestCase):
    def test_defaults_to_head_only_pose_and_vrchat_osc_without_ardy(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.avatar_profile, "felis")
        self.assertEqual(args.hands, "none")
        self.assertEqual(args.follow_output, "vrchat-osc")
        self.assertEqual(args.pose_rate_hz, 20.0)
        self.assertIsNone(args.duration)
        self.assertEqual(args.left_hand_euler, (90.0, 0.0, 0.0))
        self.assertEqual(args.right_hand_euler, (90.0, 0.0, 0.0))

    def test_vmt_locomotion_requires_both_controller_devices(self):
        args = build_parser().parse_args(
            ["--follow-output", "vmt", "--hands", "left"]
        )

        with self.assertRaisesRegex(ValueError, "both hands"):
            validate_args(args)

    def test_vrchat_osc_can_probe_one_hand(self):
        args = build_parser().parse_args(
            ["--hands", "left", "--follow-output", "vrchat-osc"]
        )

        validate_args(args)


if __name__ == "__main__":
    unittest.main()
