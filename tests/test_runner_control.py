from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.test_provider_profiles import profile_data
from vrc_ardy_agent.runner.control import dispatch


class ControlTests(unittest.TestCase):
    def test_setup_and_core_share_catalog_and_active_revision(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            bootstrap = dispatch("bootstrap", {}, path=path)
            self.assertIsNone(bootstrap["profile"])
            self.assertIn("gemini", bootstrap["catalog"])
            saved = dispatch("profile.commit", {"profile": profile_data(), "revision": None}, path=path)
            bootstrap = dispatch("bootstrap", {}, path=path)
            self.assertEqual(saved["revision"], bootstrap["revision"])
            self.assertEqual(bootstrap["profile"]["bindings"]["tts"]["voice"], "voice-id")

    def test_offline_doctor_does_not_contact_or_launch_services(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            dispatch("profile.commit", {"profile": profile_data(), "revision": None}, path=path)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
                 patch("subprocess.Popen", side_effect=AssertionError("launch forbidden")):
                checks = dispatch("doctor", {"offline": True}, path=path)
            self.assertTrue(any(row["result"] == "pending" for row in checks))
            self.assertTrue(any(row["check"] == "file" and row["result"] == "error" for row in checks))

    def test_settings_preserve_other_profile_fields(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            saved = dispatch("profile.commit", {"profile": profile_data(), "revision": None}, path=path)
            dispatch("settings", {"language": "ko"}, path=path)
            loaded = dispatch("bootstrap", {}, path=path)
            self.assertEqual(loaded["profile"]["language"], "ko")
            self.assertEqual(loaded["profile"]["providers"], saved["profile"]["providers"])

    def test_start_requires_configuration_and_does_not_start_wrong_os(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            with self.assertRaises(ValueError): dispatch("start", {"service": "agent"}, path=path)
            dispatch("profile.commit", {"profile": profile_data("windows", True), "revision": None}, path=path)
            with patch("vrc_ardy_agent.runner.control.current_os", return_value="macos"), self.assertRaisesRegex(ValueError, "host"):
                dispatch("start", {"service": "agent"}, path=path)
