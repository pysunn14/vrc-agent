from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from vrc_ardy_agent.runner.settings import SettingsStore


class RunnerSettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "settings.json"
        self.store = SettingsStore(self.path)

    def test_language_survives_reopening(self):
        self.assertEqual(self.store.read().language, "en")
        self.store.update(language="ko")
        reopened = SettingsStore(self.path).read()
        self.assertEqual(reopened.language, "ko")

    def test_concurrent_clients_write_complete_preference_documents(self):
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(self.store.update, language="ko")
            b = pool.submit(SettingsStore(self.path).update, language="en")
            self.assertEqual(a.result().language, "ko")
            self.assertEqual(b.result().language, "en")
        self.assertIn(json.loads(self.path.read_text()), [{"language": "en"}, {"language": "ko"}])

    def test_invalid_or_broken_settings_are_reported_and_preserved(self):
        for raw in ('{"language":"ja"}', 'not json'):
            self.path.write_text(raw)
            with self.assertRaises(ValueError):
                self.store.read()
            with self.assertRaises(ValueError):
                self.store.update(language="en")
            self.assertEqual(self.path.read_text(), raw)

    def test_unsupported_language_cannot_replace_preferences(self):
        self.store.update(language="ko")
        with self.assertRaises(ValueError):
            self.store.update(language="fr")
        self.assertEqual(self.store.read().language, "ko")



if __name__ == "__main__":
    unittest.main()
