from tests.rig_fixture import RIG_PATH
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.test_provider_profiles import profile_data
from vrc_ardy_agent.runner.control import dispatch
from vrc_ardy_agent.runner.launch import bridge_command, runner_config
from vrc_ardy_agent.runner.profile import Profile, ProfileStore


def role_profile(role="agent", os_name="linux"):
    data = profile_data(os_name if role != "combined" else "windows", role == "combined")
    data.update(version=3, role=role, connection={"stream_port": 8766, "agent_port": 8765, "bridge_port": 8767})
    data["bridge"].pop("mode")
    if role == "agent":
        del data["bridge"]
    if role == "bridge":
        for key in ("runtime", "providers", "bindings", "avatar", "behaviors", "autonomy"):
            del data[key]
        data["bridge"].update(avatar_rig=RIG_PATH, hmd_base=[0, 1, 0])
    return data


class HostRoleTests(unittest.TestCase):
    def test_role_owns_only_its_local_services(self):
        for role, expected in (("agent", {"agent"}), ("bridge", {"bridge"}), ("combined", {"agent", "bridge"})):
            with self.subTest(role=role):
                profile = Profile.parse(role_profile(role), Path("profile.json"))
                services = runner_config(profile).services
                self.assertEqual({s.id for s in services if s.command}, expected)
                self.assertEqual(profile.local_host, "game" if role == "bridge" else "runner")
                self.assertEqual(profile.has_agent, role != "bridge")
                self.assertEqual(profile.has_bridge, role != "agent")

    def test_bridge_needs_no_agent_settings_and_uses_its_own_geometry(self):
        data = role_profile("bridge")
        data["connection"]["stream_port"] = 9000
        data["bridge"]["hmd_base"] = [0, 1.2, 0]
        command = bridge_command(Profile.parse(data))
        self.assertEqual(command[command.index("--bridge-port") + 1], "9000")
        self.assertEqual(command[command.index("--hmd-base") + 2], "1.2")
        self.assertEqual(command[command.index("--agent-host") + 1], "agent-host")

    def test_invalid_roles_topologies_and_foreign_sections_fail(self):
        cases = []
        data = role_profile("agent"); data["bridge"] = {}; cases.append(data)
        data = role_profile("bridge"); data["runtime"] = {}; cases.append(data)
        data = role_profile("combined"); data["hosts"]["runner"]["os"] = "linux"; cases.append(data)
        data = role_profile("bridge"); data["game_host"] = data["runner_host"]; cases.append(data)
        data = role_profile("combined"); data["connection"]["bridge_port"] = 8765; cases.append(data)
        data = role_profile("agent"); data["role"] = "unknown"; cases.append(data)
        data = role_profile("agent"); data["hosts"]["runner"]["address"] = "127.0.0.1"; cases.append(data)
        data = role_profile("bridge"); data["hosts"]["game"]["address"] = "0.0.0.0"; cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError): Profile.parse(data)

    def test_legacy_active_profile_and_partial_draft_upgrade_without_writes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            raw = json.dumps(profile_data()).encode()
            path.write_bytes(raw)
            store = ProfileStore(path)
            store.draft_path.write_text(json.dumps({"profile": {"version": 2, "language": "ko", "hosts": {},
                "runner_host": "runner", "game_host": "game", "runtime": {}}, "revision": store.read()["revision"]}))
            state = dispatch("bootstrap", {}, path=path)
            self.assertEqual(state["profile"]["role"], "agent")
            self.assertEqual(state["draft"]["profile"]["version"], 3)
            self.assertEqual(path.read_bytes(), raw)

    def test_bridge_doctor_does_not_probe_ardy_or_providers(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            ProfileStore(path).commit(role_profile("bridge"), expected_revision=None)
            with patch("vrc_ardy_agent.runner.diagnostics.runtime_probe", side_effect=AssertionError("ARDY forbidden")), \
                 patch("vrc_ardy_agent.runner.diagnostics.current_os", return_value="windows"), \
                 patch("subprocess.Popen", side_effect=AssertionError("offline cannot launch")):
                rows = dispatch("doctor", {"offline": True}, path=path)
            self.assertTrue(any(row["check"] == "platform" and row["result"] == "pass" for row in rows))
            self.assertFalse(any(row["service"] in ("avatar", "autonomy", "behaviors", "chat", "audio") for row in rows))

    def test_foreign_actions_fail_explicitly_and_cannot_start_remote_service(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            ProfileStore(path).commit(role_profile("bridge"), expected_revision=None)
            for action in ("providers", "behaviors", "runtime.probe", "runtime.setup"):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, "agent role"):
                    dispatch(action, {}, path=path)
            with patch("vrc_ardy_agent.runner.control.current_os", return_value="windows"), \
                 patch("vrc_ardy_agent.runner.core.Runner.start", side_effect=AssertionError("must not start")), \
                 self.assertRaisesRegex(ValueError, "remote"):
                dispatch("start", {"service": "agent"}, path=path)

    def test_role_change_is_blocked_while_local_service_is_owned(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            saved = ProfileStore(path).commit(role_profile("bridge"), expected_revision=None)
            with patch("vrc_ardy_agent.runner.core.Runner.observe", return_value={"owned": True}), \
                 self.assertRaisesRegex(ValueError, "stop"):
                dispatch("profile.commit", {"profile": role_profile("agent"), "revision": saved["revision"]}, path=path)

    def test_current_role_change_draft_can_prepare_ardy_but_stale_draft_cannot(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            store = ProfileStore(path)
            saved = store.commit(role_profile("bridge"), expected_revision=None)
            store.save_draft(role_profile("agent", "windows"), expected_revision=saved["revision"])
            with patch("vrc_ardy_agent.runner.runtime_setup.api.options", return_value={"ready": True}):
                self.assertEqual(dispatch("runtime.options", {}, path=path), {"ready": True})
                # Concurrent publication invalidates this setup draft.
                store.draft_path.write_text(json.dumps({"profile": role_profile("agent"), "revision": "stale"}))
                with self.assertRaisesRegex(ValueError, "agent role"):
                    dispatch("runtime.options", {}, path=path)

    def test_incomplete_bridge_remains_inspectable_but_cannot_start(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            data = role_profile("bridge")
            data["bridge"].update(avatar_rig="", virtual_mic_device="")
            ProfileStore(path).commit(data, expected_revision=None)
            with patch("vrc_ardy_agent.runner.core.probe", return_value={"ok": False, "detail": "offline"}):
                rows = dispatch("status", {}, path=path)
            self.assertEqual([row["id"] for row in rows if row["local"]], ["bridge"])
            checks = dispatch("doctor", {"offline": True}, path=path)
            self.assertTrue(any(row["check"] == "rig" and row["result"] == "error" for row in checks))
            with patch("vrc_ardy_agent.runner.control.current_os", return_value="windows"), \
                 patch("subprocess.Popen", side_effect=AssertionError("cannot launch")), \
                 self.assertRaisesRegex(ValueError, "preflight"):
                dispatch("start", {"service": "bridge"}, path=path)

    def test_bridge_online_diagnostics_use_device_session_and_do_not_probe_ardy(self):
        from io import BytesIO
        from vrc_ardy_agent.runner.diagnostics import doctor
        data = role_profile("bridge")
        data["bridge"]["audio_source"] = "microphone"
        data["bridge"]["microphone_device"] = "Microphone"
        profile = Profile.parse(data, Path("agent.json"))
        response = {"device_plane": {"registration_verified": False, "components": {"bridge": {"connected": True}}},
                    "sensor_supervisor": {"state": "RUNNING"}}
        runtime = {"platform": "Windows", "packages": {"websockets": "test"}, "audio_devices": [
            {"index": 1, "name": "Virtual input", "max_output_channels": 2},
            {"index": 2, "name": "Microphone", "max_input_channels": 1}]}
        with patch("vrc_ardy_agent.runner.diagnostics.current_os", return_value="windows"), \
             patch("vrc_ardy_agent.runner.diagnostics.runtime_probe", side_effect=AssertionError("ARDY forbidden")), \
             patch("vrc_ardy_agent.runner.bridge_diagnostics.bridge_probe", return_value=runtime), \
             patch("vrc_ardy_agent.runner.core.Runner.status", return_value=[]), \
             patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as opened:
            rows = doctor(profile)
        self.assertEqual(opened.call_args.args[0], "http://game-host:8767/status")
        self.assertTrue(any(row["check"] == "session" and row["result"] == "pass" for row in rows))
        self.assertTrue(any(row["check"] == "tracking" and row["result"] == "pending" for row in rows))

    def test_bridge_snapshot_records_its_rig_and_environment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = role_profile("bridge")
            data["bridge"]["project_dir"] = str(root)
            for name in ("uv.lock", "package-lock.json"): (root / name).write_text("lock")
            saved = ProfileStore(root / "agent.json").commit(data, expected_revision=None)
            with patch("vrc_ardy_agent.runner.reproducibility.runtime_probe", side_effect=AssertionError("ARDY forbidden")), \
                 patch("vrc_ardy_agent.runner.bridge_diagnostics.bridge_probe", return_value={"platform": "Windows"}):
                result = dispatch("snapshot", {}, path=root / "agent.json")
            self.assertTrue(result["complete"])
            self.assertEqual(set(result["assets"]), {"rig"})
            self.assertEqual(result["profile_revision"], saved["revision"])
