from tests.rig_fixture import RIG_PATH
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from vrc_ardy_agent.providers.catalog import catalog, validate_connection
from vrc_ardy_agent.runner.profile import Profile, ProfileStore, RevisionConflict


def profile_data(os_name="macos", together=False):
    return {
        "version": 2, "language": "en",
        "hosts": {"runner": {"os": os_name, "address": "agent-host"},
                  **({} if together else {"game": {"os": "windows", "address": "game-host"}})},
        "runner_host": "runner", "game_host": "runner" if together else "game",
        "providers": {
            "chat": {"provider": "openai-compatible", "label": "Chat server",
                     "config": {"base_url": "http://agent-host:8000/v1"},
                     "deployment": {"kind": "external", "host": "runner"}},
            "audio": {"provider": "openai", "label": "Audio API",
                      "config": {"api_key_env": "TEST_AUDIO_KEY"}, "deployment": {"kind": "api"}},
        },
        "bindings": {"llm": {"instance": "chat", "model": "chat-model", "vision": False},
                     "stt": {"instance": "audio", "model": "transcribe-model", "language": "ko"},
                     "tts": {"instance": "audio", "model": "speech-model", "voice": "voice-id"}},
        "runtime": {"python": "python", "project_dir": ".", "device": "auto"},
        "avatar": {"rig": RIG_PATH, "base_pose": "idle.json", "face_channels": {}, "hmd_base": [0, 1, 0]},
        "behaviors": {}, "autonomy": {"enabled": False, "interval_seconds": 30, "behaviors": []},
        "bridge": {"mode": "managed" if together else "external", "python": "python",
                   "project_dir": "C:/agent", "virtual_mic_device": "Virtual input"},
    }


class ProfileTests(unittest.TestCase):
    def test_three_topologies_keep_provider_choice_independent(self):
        for os_name, together in (("macos", False), ("linux", False), ("windows", True)):
            with self.subTest(os=os_name):
                profile = Profile.parse(profile_data(os_name, together))
                connection, binding = profile.resolve("tts")
                self.assertEqual(connection["provider"], "openai")
                self.assertEqual(binding["voice"], "voice-id")
                self.assertEqual(profile.bridge_address, "127.0.0.1" if together else "agent-host")

    def test_same_definition_has_independent_named_instances(self):
        data = profile_data()
        data["providers"]["second"] = deepcopy(data["providers"]["chat"])
        data["providers"]["second"]["config"]["base_url"] = "http://another-host:8000/v1"
        data["bindings"]["llm"]["instance"] = "second"
        profile = Profile.parse(data)
        self.assertIn("another-host", profile.resolve("llm")[0]["config"]["base_url"])
        self.assertIn("agent-host", profile.data["providers"]["chat"]["config"]["base_url"])

    def test_invalid_references_capabilities_and_topologies_fail(self):
        mutations = [
            lambda d: d["bindings"]["llm"].update(instance="missing"),
            lambda d: d["providers"]["chat"].update(provider="elevenlabs", config={"api_key_env": "KEY"}),
            lambda d: d["hosts"]["game"].update(os="linux"),
            lambda d: d.update(game_host="runner"),
            lambda d: d["bridge"].update(mode="managed"),
            lambda d: d["providers"]["chat"]["deployment"].update(kind="managed", host="game", command=["server"]),
            lambda d: d["providers"]["chat"]["deployment"].update(host="unknown"),
            lambda d: d["runtime"].update(device="mps") if d["hosts"]["runner"].update(os="linux") is None else None,
        ]
        for mutate in mutations:
            data = profile_data(); mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                Profile.parse(data)

    def test_provider_fields_are_shared_and_strict(self):
        self.assertIn("elevenlabs", catalog())
        self.assertEqual(validate_connection("gemini", {"api_key_env": "KEY"})["base_url"],
                         "https://generativelanguage.googleapis.com/v1beta/openai")
        with self.assertRaises(ValueError):
            validate_connection("openai", {"api_key_env": "KEY", "typo": "value"})
        with self.assertRaises(ValueError):
            validate_connection("whisper-local", {"device": "mps"})

    def test_drafts_do_not_change_active_profile_and_stale_commits_fail(self):
        with TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / "agent.json")
            first = store.commit(profile_data(), expected_revision=None)
            other = deepcopy(first["profile"]); other["language"] = "ko"
            store.save_draft(other, expected_revision=first["revision"])
            self.assertEqual(store.read()["profile"]["language"], "en")
            second = store.commit(other, expected_revision=first["revision"])
            with self.assertRaises(RevisionConflict):
                store.commit(first["profile"], expected_revision=first["revision"])
            self.assertEqual(store.read(), second)
            invalid = deepcopy(other); invalid["bindings"]["tts"]["instance"] = "missing"
            with self.assertRaises(ValueError):
                store.commit(invalid, expected_revision=second["revision"])
            self.assertEqual(store.read(), second)


if __name__ == "__main__":
    unittest.main()
