import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.test_provider_profiles import profile_data
from vrc_ardy_agent.runner.profile import Profile, ProfileStore
from vrc_ardy_agent.runner.reproducibility import snapshot


class SnapshotTests(unittest.TestCase):
    def test_failed_snapshot_preserves_previous_complete_result_and_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ProfileStore(root / "agent.json")
            saved = store.commit(profile_data(), expected_revision=None)
            lock = root / "agent.lock.json"
            lock.write_text('{"previous":true}')
            with patch("vrc_ardy_agent.runner.reproducibility.runtime_probe", return_value={"python": "test"}), self.assertRaises(FileNotFoundError):
                snapshot(Profile.load(store.path), saved["revision"], progress=lambda *_: None)
            self.assertEqual(json.loads(lock.read_text()), {"previous": True})
            pending = json.loads((root / "agent.lock.json.pending").read_text())
            self.assertFalse(pending["complete"])
            self.assertLess(pending["completed"], pending["total"])

    def test_successful_snapshot_records_exact_assets_and_dependency_locks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = profile_data(); data["runtime"]["project_dir"] = str(root)
            for name in ("idle.json", "motion.json", "face.json", "uv.lock", "package-lock.json"):
                (root / name).write_text("example")
            saved = ProfileStore(root / "agent.json").commit(data, expected_revision=None)
            with patch("vrc_ardy_agent.runner.reproducibility.runtime_probe", return_value={"python": "test"}):
                result = snapshot(Profile.load(root / "agent.json"), saved["revision"], progress=lambda *_: None)
            self.assertTrue(result["complete"])
            self.assertEqual(result["assets"]["avatar.base_pose"]["bytes"], 7)
            self.assertEqual(set(result["dependency_locks"]), {"uv.lock", "package-lock.json"})
            self.assertFalse((root / "agent.lock.json.pending").exists())
