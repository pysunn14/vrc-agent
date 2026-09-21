import json
from pathlib import Path
import subprocess
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vrc_ardy_agent.runner.host_discovery import discover_hosts, tailscale_executable


class DiscoveryTests(unittest.TestCase):
    def test_read_only_discovery_keeps_offline_peers_and_prefers_ipv4(self):
        status = {"BackendState": "Running", "Self": {"ID": "self", "OS": "macOS", "HostName": "runner",
            "TailscaleIPs": ["fd7a:115c:a1e0::1", "100.64.0.1"], "Online": True}, "Peer": {
            "key": {"ID": "peer", "OS": "windows", "HostName": "game", "Online": False,
                    "TailscaleIPs": ["100.64.0.2"]}}}
        with patch("vrc_ardy_agent.runner.host_discovery.tailscale_executable", return_value="tailscale"), \
             patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(status), stderr="")) as run:
            result = discover_hosts()
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["self"]["address"], "100.64.0.1")
        self.assertEqual(result["self"]["os"], "macos")
        self.assertFalse(result["peers"][0]["online"])
        self.assertEqual(run.call_args.args[0], ["tailscale", "status", "--json"])
        self.assertLessEqual(run.call_args.kwargs["timeout"], 10)

    def test_missing_stopped_malformed_and_timeout_are_distinct(self):
        with patch("vrc_ardy_agent.runner.host_discovery.tailscale_executable", return_value=None):
            self.assertEqual(discover_hosts()["state"], "unavailable")
        cases = [(SimpleNamespace(returncode=0, stdout='{"BackendState":"NeedsLogin"}', stderr=""), "needs-login"),
                 (SimpleNamespace(returncode=0, stdout='{"BackendState":"Stopped"}', stderr=""), "stopped"),
                 (SimpleNamespace(returncode=0, stdout='not json', stderr=""), "error"),
                 (SimpleNamespace(returncode=1, stdout='', stderr="daemon unavailable"), "error")]
        for completed, expected in cases:
            with patch("vrc_ardy_agent.runner.host_discovery.tailscale_executable", return_value="tailscale"), \
                 patch("subprocess.run", return_value=completed):
                result = discover_hosts()
                self.assertEqual(result["state"], expected)
                self.assertTrue(result["detail"])
        with patch("vrc_ardy_agent.runner.host_discovery.tailscale_executable", return_value="tailscale"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("tailscale", 5)):
            self.assertEqual(discover_hosts()["state"], "error")

    def test_platform_install_locations_without_mac_assumption(self):
        with patch("shutil.which", return_value=None), patch.object(Path, "is_file", return_value=True):
            self.assertEqual(tailscale_executable("macos"), "/Applications/Tailscale.app/Contents/MacOS/Tailscale")
            with patch.dict("os.environ", {"ProgramFiles": "C:/Program Files"}):
                self.assertEqual(tailscale_executable("windows"), str(Path("C:/Program Files/Tailscale/tailscale.exe")))
            self.assertIsNone(tailscale_executable("linux"))

    def test_python_discovery_uses_the_running_platform_layout(self):
        from tempfile import TemporaryDirectory
        from vrc_ardy_agent.runner.runtime_setup.paths import discover_python
        for os_name, relative in (("nt", "Scripts/python.exe"), ("posix", "bin/python")):
            with self.subTest(os=os_name), TemporaryDirectory() as directory:
                root = Path(directory)
                python = root / ".venv" / relative
                python.parent.mkdir(parents=True)
                python.touch()
                with patch("vrc_ardy_agent.runner.runtime_setup.paths.os", SimpleNamespace(name=os_name, path=os.path)):
                    self.assertEqual(discover_python(root), python)
