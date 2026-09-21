from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr

from scripts.run_windows_companion import build_parser


class WindowsCompanionCliTests(unittest.TestCase):
    def test_microphone_source_requires_an_explicit_input_device(self) -> None:
        args = build_parser().parse_args(
            [
                "run", "--avatar-profile", "shinano",
                "--window-title",
                "VRChat",
                "--mac-host",
                "100.64.0.1",
                "--virtual-mic-device",
                "2",
                "--audio-source",
                "microphone",
                "--microphone-device",
                "12",
            ]
        )

        self.assertEqual(args.audio_source, "microphone")
        self.assertEqual(args.microphone_device, 12)

    def test_numeric_audio_device_is_parsed_as_an_index(self) -> None:
        args = build_parser().parse_args(
            [
                "run", "--avatar-profile", "shinano",
                "--window-title",
                "VRChat",
                "--mac-host",
                "100.64.0.1",
                "--virtual-mic-device",
                "0",
            ]
        )

        self.assertEqual(args.virtual_mic_device, 0)

    def test_process_audio_helper_can_be_selected_explicitly(self) -> None:
        args = build_parser().parse_args(
            [
                "run", "--avatar-profile", "shinano",
                "--window-title",
                "VRChat",
                "--mac-host",
                "100.64.0.1",
                "--virtual-mic-device",
                "0",
                "--process-audio-helper",
                "C:/tools/vrc-ardy-process-audio-probe.dll",
            ]
        )

        self.assertEqual(
            args.process_audio_helper,
            "C:/tools/vrc-ardy-process-audio-probe.dll",
        )

    def test_vrchat_account_can_replace_a_fixed_process_id(self) -> None:
        args = build_parser().parse_args(
            [
                "run", "--avatar-profile", "shinano",
                "--vrchat-user-name",
                "AgentAvatar",
                "--mac-host",
                "100.64.0.1",
                "--virtual-mic-device",
                "0",
            ]
        )

        self.assertEqual(args.vrchat_user_name, "AgentAvatar")
        self.assertIsNone(args.window_process_id)
        self.assertEqual(args.avatar_profile, "shinano")
        self.assertEqual(args.hmd_base, (0.0, 1.0, 0.0))
        self.assertEqual(args.pose_rate_hz, 20.0)
        self.assertEqual(args.body_output, "vrchat-osc")

    def test_test_controls_can_capture_a_separate_observer_account(self) -> None:
        args = build_parser().parse_args(
            [
                "run", "--avatar-profile", "shinano",
                "--vrchat-user-name",
                "AgentAvatar",
                "--mac-host",
                "100.64.0.1",
                "--virtual-mic-device",
                "0",
                "--enable-test-controls",
                "--debug-observer-vrchat-user-name",
                "Observer",
            ]
        )

        self.assertTrue(args.enable_test_controls)
        self.assertEqual(args.debug_observer_vrchat_user_name, "Observer")

    def test_run_requires_bridge_target_window_and_virtual_microphone(self) -> None:
        args = build_parser().parse_args(
            [
                "run", "--avatar-profile", "shinano",
                "--window-title",
                "VRChat",
                "--mac-host",
                "100.64.0.1",
                "--virtual-mic-device",
                "CABLE Input",
            ]
        )

        self.assertEqual(args.window_title, "VRChat")
        self.assertEqual(args.agent_host, "100.64.0.1")
        self.assertEqual(args.bridge_port, 8766)
        self.assertEqual(args.virtual_mic_device, "CABLE Input")

    def test_run_requires_exactly_one_window_selector(self) -> None:
        parser = build_parser()

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run", "--avatar-profile", "shinano",
                    "--mac-host",
                    "100.64.0.1",
                    "--virtual-mic-device",
                    "CABLE Input",
                ]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run", "--avatar-profile", "shinano",
                    "--hwnd",
                    "123",
                    "--window-title",
                    "VRChat",
                    "--mac-host",
                    "100.64.0.1",
                    "--virtual-mic-device",
                    "CABLE Input",
                ]
            )


if __name__ == "__main__":
    unittest.main()
