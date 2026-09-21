from __future__ import annotations

import threading
import unittest

from vrc_ardy_agent.companion_http_runtime import CompanionHttpRuntime


class _Server:
    server_address = ("127.0.0.1", 8765)

    def __init__(self) -> None:
        self.release = threading.Event()
        self.closed = False
        self.shutdown_calls = 0

    def serve_forever(self) -> None:
        self.release.wait(timeout=1.0)

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.release.set()

    def server_close(self) -> None:
        self.closed = True


class CompanionHttpRuntimeTests(unittest.TestCase):
    def test_start_and_stop_observe_the_server_thread(self) -> None:
        server = _Server()
        runtime = CompanionHttpRuntime(server)

        runtime.start()
        self.assertTrue(runtime.snapshot().running)
        runtime.stop()

        self.assertFalse(runtime.snapshot().running)
        self.assertTrue(server.closed)

    def test_stop_before_start_only_closes_the_bound_socket(self) -> None:
        server = _Server()
        runtime = CompanionHttpRuntime(server)

        runtime.stop()

        self.assertTrue(server.closed)
        self.assertEqual(server.shutdown_calls, 0)
        self.assertFalse(runtime.snapshot().running)


if __name__ == "__main__":
    unittest.main()
