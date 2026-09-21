from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from vrc_ardy_agent.runner.config import RunnerConfig, Service
from vrc_ardy_agent.runner.core import Runner
from vrc_ardy_agent.runner.settings import atomic_json


def eventually(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise AssertionError("expected state was not observed")


class RunnerCoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def runner(self, service):
        return Runner(RunnerConfig((service,)), self.root / "state")

    def test_external_health_is_not_claimed_or_stopped(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        service = Service("runtime", command=(sys.executable, "-c", "raise AssertionError('duplicate')"),
                          health_url=f"http://127.0.0.1:{server.server_port}", health_status="ok")
        runner = self.runner(service)
        self.assertEqual(runner.observe("runtime")["state"], "external")
        self.assertEqual(runner.start("runtime")["state"], "external")
        with self.assertRaisesRegex(ValueError, "outside"):
            runner.stop("runtime")
        self.assertEqual(runner.observe("runtime")["state"], "external")

    def test_concurrent_start_reconnect_and_stop_use_one_owned_process(self):
        runner = self.runner(Service("worker", command=(sys.executable, "-u", "-c",
                             "import time; print('worker ready', flush=True); time.sleep(60)")))
        self.addCleanup(lambda: runner.stop("worker"))
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: runner.start("worker"), range(2)))
        self.assertEqual(results[0]["supervisor_pid"], results[1]["supervisor_pid"])
        eventually(lambda: runner.observe("worker")["state"] == "running")
        original = runner.observe("worker")
        reconnected = self.runner(runner.config.services[0])
        self.assertEqual(reconnected.observe("worker")["pid"], original["pid"])
        self.assertIsNotNone(original["heartbeat_at"])
        eventually(lambda: "worker ready" in reconnected.logs("worker"))
        stopped = reconnected.stop("worker")
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(reconnected.stop("worker")["state"], "stopped")

    def test_process_failure_keeps_exit_code_and_original_output(self):
        runner = self.runner(Service("failure", command=(sys.executable, "-u", "-c",
                             "print('raw [error] 원문', flush=True); raise SystemExit(7)")))
        runner.start("failure")
        eventually(lambda: runner.observe("failure")["state"] == "failed")
        self.assertEqual(runner.observe("failure")["exit_code"], 7)
        self.assertIn("raw [error] 원문", runner.logs("failure"))

    def test_stale_pid_cannot_be_used_to_signal_another_process(self):
        runner = self.runner(Service("stale", command=(sys.executable, "-c", "pass")))
        atomic_json(runner.record_path("stale"), {
            "ticket": "old", "supervisor_pid": 1, "supervisor_created": 0,
            "pid": 1, "created": 0, "phase": "running", "exit_code": None,
        })
        self.assertEqual(runner.observe("stale")["state"], "failed")
        self.assertEqual(runner.stop("stale")["state"], "failed")

    def test_doctor_checks_files_and_commands_without_starting_processes(self):
        missing = self.root / "missing.bin"
        runner = self.runner(Service("runtime", command=("/missing/executable",), required_files=(missing,)))
        checks = runner.doctor()
        self.assertTrue(any(row["code"] == "command_invalid" for row in checks))
        self.assertTrue(any(row["code"] == "asset_missing" for row in checks))
        self.assertFalse(runner.record_path("runtime").exists())

    def test_config_rejects_unknown_fields_and_invalid_commands(self):
        path = self.root / "runner.toml"
        path.write_text('[services.runtime]\ncommmand=["python"]\n')
        with self.assertRaises(ValueError):
            RunnerConfig.load(path)
        with self.assertRaises(ValueError):
            Service("bad/path", command=("python",))
        with self.assertRaises(ValueError):
            Service("runtime", command="python")


if __name__ == "__main__":
    unittest.main()
