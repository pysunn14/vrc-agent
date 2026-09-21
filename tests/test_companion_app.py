from __future__ import annotations

import unittest

from vrc_ardy_agent.companion_app import MacCompanionApp


class _Component:
    def __init__(self, name: str, events: list[str], *, fail_start: bool = False) -> None:
        self.name = name
        self.events = events
        self.fail_start = fail_start

    def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} failed")

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")

    def close(self, **_kwargs) -> None:
        self.events.append(f"close:{self.name}")

    def snapshot(self):
        return {"name": self.name}


class _Interaction:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def halt_all(self, *, reason: str):
        self.events.append(f"halt:{reason}")


class _Supervisor:
    def __init__(self, events: list[str], *, idle: bool = True) -> None:
        self.events = events
        self.idle = idle

    def wait_until_idle(self, *, timeout: float):
        self.events.append(f"wait:{timeout}")
        return self.idle


class MacCompanionAppTests(unittest.TestCase):
    def test_shutdown_closes_ingress_before_invalidating_actions(self) -> None:
        events = []
        bridge = _Component("bridge", events)
        control = _Component("control", events)
        pipeline = _Component("pipeline", events)
        motion = _Component("motion", events)
        app = MacCompanionApp(
            bridge=bridge,
            control=control,
            pipeline=pipeline,
            interaction=_Interaction(events),
            supervisor=_Supervisor(events),
            motion_director=motion,
            idle_timeout_seconds=3.0,
        )

        app.start()
        app.stop()

        self.assertEqual(
            events,
            [
                "start:motion",
                "start:bridge",
                "start:control",
                "stop:control",
                "close:pipeline",
                "halt:companion_shutdown",
                "wait:3.0",
                "stop:motion",
                "stop:bridge",
            ],
        )
        self.assertFalse(app.snapshot().running)

    def test_partial_start_is_torn_down(self) -> None:
        events = []
        app = MacCompanionApp(
            bridge=_Component("bridge", events),
            control=_Component("control", events, fail_start=True),
            pipeline=_Component("pipeline", events),
            interaction=_Interaction(events),
            supervisor=_Supervisor(events),
            motion_director=_Component("motion", events),
        )

        with self.assertRaisesRegex(RuntimeError, "control failed"):
            app.start()

        self.assertIn("stop:bridge", events)
        self.assertIn("stop:motion", events)
        self.assertIn("halt:companion_shutdown", events)
        self.assertFalse(app.snapshot().running)


if __name__ == "__main__":
    unittest.main()
