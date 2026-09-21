from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.test_provider_profiles import profile_data
from vrc_ardy_agent.runner.control import dispatch


class RuntimeConfigStateTests(unittest.TestCase):
    def test_running_services_require_stop_before_changing_execution_config(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            first = dispatch("profile.commit", {"profile": profile_data(), "revision": None}, path=path)
            updated = deepcopy(first["profile"])
            updated["bindings"]["llm"]["model"] = "another-model"
            with patch("vrc_ardy_agent.runner.core.Runner.observe", return_value={"owned": True}), self.assertRaisesRegex(ValueError, "stop"):
                dispatch("profile.commit", {"profile": updated, "revision": first["revision"]}, path=path)
            self.assertEqual(dispatch("bootstrap", {}, path=path)["revision"], first["revision"])

    def test_language_change_does_not_require_stopping_runtime(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            dispatch("profile.commit", {"profile": profile_data(), "revision": None}, path=path)
            with patch("vrc_ardy_agent.runner.core.Runner.observe", return_value={"owned": True}):
                result = dispatch("settings", {"language": "ko"}, path=path)
            self.assertEqual(result["profile"]["language"], "ko")
