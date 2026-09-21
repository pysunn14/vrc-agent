from __future__ import annotations

from pathlib import Path
import unittest


class WindowsRequirementsTests(unittest.TestCase):
    def test_tracking_runtime_dependencies_are_declared_directly(self):
        requirements = Path("requirements-windows.txt").read_text(encoding="utf-8").splitlines()

        self.assertIn("scipy==1.17.1", requirements)
        self.assertIn("lap==0.5.13", requirements)

    def test_companion_runtime_dependencies_are_declared_directly(self):
        requirements = Path("requirements-windows.txt").read_text(encoding="utf-8").splitlines()

        self.assertFalse(any(requirement.startswith("proc-tap") for requirement in requirements))
        self.assertIn("sounddevice==0.5.6", requirements)
        self.assertIn("websockets==17.1", requirements)


if __name__ == "__main__":
    unittest.main()
