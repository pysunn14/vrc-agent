from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.test_provider_profiles import profile_data
from vrc_ardy_agent.runner.profile import Profile
from vrc_ardy_agent.runner.launch import runner_config, bridge_command, apply_runtime_profile


class LaunchTests(unittest.TestCase):
    def test_remote_windows_is_observed_without_local_windows_configuration(self):
        profile = Profile.parse(profile_data(), Path("agent.json"))
        config = runner_config(profile)
        self.assertEqual(config.services[0].id, "agent")
        self.assertIn("--profile", config.services[0].command)
        bridge = next(s for s in config.services if s.id == "bridge")
        self.assertEqual(bridge.command, ())
        self.assertIn("game-host", bridge.health_url)
        with self.assertRaisesRegex(ValueError, "Windows bridge computer"):
            bridge_command(profile)
        from vrc_ardy_agent.runner.launch import connection_info
        self.assertEqual(connection_info(profile)["game"]["address"], "game-host")

    def test_same_windows_compiles_local_owned_bridge(self):
        profile = Profile.parse(profile_data("windows", True), Path("agent.json"))
        config = runner_config(profile)
        bridge = next(s for s in config.services if s.id == "bridge")
        self.assertTrue(bridge.command)
        self.assertIn("127.0.0.1", bridge.command)
        self.assertIn("127.0.0.1", bridge.health_url)

    def test_managed_provider_command_is_not_reinterpreted_by_shell(self):
        data = profile_data()
        data["providers"]["chat"]["deployment"] = {
            "kind": "managed", "host": "runner", "cwd": ".", "command": ["server", "a b", "$(literal)"]}
        config = runner_config(Profile.parse(data, Path("agent.json")))
        service = next(s for s in config.services if s.id == "provider-chat")
        self.assertEqual(service.command, ("server", "a b", "$(literal)"))

    def test_runtime_loads_same_profile_bindings_and_explicit_assets(self):
        from scripts.run_companion import build_parser
        data = profile_data()
        with TemporaryDirectory() as directory:
            from vrc_ardy_agent.runner.profile import ProfileStore
            path = Path(directory) / "agent.json"
            import json
            from tests.test_motion_replay import pose
            from vrc_ardy_agent.device_payloads import encode_pose_payload
            from vrc_ardy_agent.avatar_config import attest_calibration
            (path.parent / "idle.json").write_text(json.dumps(encode_pose_payload(pose())))
            data["avatar"]["calibration"] = attest_calibration(data["avatar"], path.parent, tracking_active=True)
            ProfileStore(path).commit(data, expected_revision=None)
            args = build_parser().parse_args(["--profile", str(path)])
            profile = apply_runtime_profile(args)
            self.assertEqual(args.device, "auto")
            self.assertEqual(args.idle_pose, (path.parent / "idle.json").resolve())
            self.assertEqual(profile.resolve("tts")[1]["voice"], "voice-id")


class CustomRigLaunchTests(unittest.TestCase):
    def test_bridge_uses_its_local_rig_independently_of_agent_paths(self):
        from tests.test_host_roles import role_profile
        data = role_profile("bridge")
        data["bridge"]["avatar_rig"] = "assets/custom-rig.json"
        command = bridge_command(Profile.parse(data, Path("agent.json")))
        self.assertEqual(command[command.index("--avatar-profile") + 1], str(Path("assets/custom-rig.json").resolve()))
        from scripts.run_windows_companion import build_parser
        args = build_parser().parse_args(command[2:])
        self.assertEqual(args.avatar_profile, str(Path("assets/custom-rig.json").resolve()))
