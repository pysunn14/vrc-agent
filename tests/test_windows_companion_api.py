from __future__ import annotations

from dataclasses import dataclass
import threading
import unittest

from vrc_ardy_agent.companion_api import CompanionApiClient, create_companion_server
from vrc_ardy_agent.companion_http_runtime import CompanionHttpRuntime
from vrc_ardy_agent.windows_companion_api import WindowsCompanionControlService


@dataclass(frozen=True)
class _RuntimeState:
    running: bool
    heartbeat_monotonic: float


class _Runtime:
    def snapshot(self):
        return _RuntimeState(running=True, heartbeat_monotonic=12.0)


class _ScreenshotProvider:
    def __call__(self) -> bytes:
        return b"\xff\xd8observer\xff\xd9"

    @staticmethod
    def snapshot():
        return {"requests": 3, "captures": 2, "last_error": None}


class WindowsCompanionApiTests(unittest.TestCase):
    def test_health_status_and_remote_shutdown_request(self) -> None:
        shutdown = threading.Event()
        service = WindowsCompanionControlService(
            _Runtime(),
            shutdown_requested=shutdown,
        )
        control = CompanionHttpRuntime(
            create_companion_server(("127.0.0.1", 0), service)
        )
        control.start()
        self.addCleanup(control.stop)
        address = control.snapshot()
        client = CompanionApiClient(
            f"http://{address.bound_host}:{address.bound_port}"
        )

        self.assertEqual(client.health()["status"], "ok")
        self.assertTrue(client.status()["running"])
        result = client.stop(reason="exploratory_test")

        self.assertEqual(result["reason"], "exploratory_test")
        self.assertTrue(result["shutdown_requested"])
        self.assertTrue(shutdown.is_set())

    def test_debug_observer_screenshot_requires_test_controls(self) -> None:
        shutdown = threading.Event()
        service = WindowsCompanionControlService(
            _Runtime(),
            shutdown_requested=shutdown,
            enable_test_controls=True,
            screenshot_provider=lambda: b"\xff\xd8observer\xff\xd9",
        )
        control = CompanionHttpRuntime(
            create_companion_server(("127.0.0.1", 0), service)
        )
        control.start()
        self.addCleanup(control.stop)
        address = control.snapshot()
        client = CompanionApiClient(
            f"http://{address.bound_host}:{address.bound_port}"
        )

        self.assertEqual(
            client.test_screenshot(),
            b"\xff\xd8observer\xff\xd9",
        )

    def test_status_exposes_debug_observer_state(self) -> None:
        service = WindowsCompanionControlService(
            _Runtime(),
            shutdown_requested=threading.Event(),
            enable_test_controls=True,
            screenshot_provider=_ScreenshotProvider(),
        )

        status = service.status()

        self.assertEqual(status["debug_observer"]["requests"], 3)



if __name__ == "__main__":
    unittest.main()
