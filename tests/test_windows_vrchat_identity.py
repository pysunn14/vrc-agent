from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from vrc_ardy_agent.windows_capture import WindowInfo
from vrc_ardy_agent.windows_vrchat_identity import (
    AmbiguousVrchatIdentityError,
    VrchatIdentityResolver,
)


def _local_timestamp(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%d_%H-%M-%S").timestamp()


class VrchatIdentityResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.log_dir = Path(self._temporary.name)
        self.windows = [
            WindowInfo(101, "VRChat", 10484, 1280, 720, False),
            WindowInfo(202, "VRChat", 6296, 1280, 720, False),
        ]
        self.started_at = {
            10484: _local_timestamp("2026-09-03_13-45-21"),
            6296: _local_timestamp("2026-09-03_15-35-25"),
        }
        self._write_log(
            "2026-09-03_13-45-21",
            "ObserverAvatar",
            "usr_human",
        )
        self._write_log(
            "2026-09-03_15-35-25",
            "AgentAvatar",
            "usr_agent",
        )

    def test_resolves_only_the_requested_account_among_two_vrchat_windows(self) -> None:
        resolver = self._resolver(user_name="AgentAvatar")

        target = resolver.resolve()

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.window.hwnd, 202)
        self.assertEqual(target.window.process_id, 6296)
        self.assertEqual(target.identity.display_name, "AgentAvatar")
        self.assertEqual(target.identity.user_id, "usr_agent")
        self.assertEqual(target.generation_key, (6296, self.started_at[6296]))

    def test_user_id_is_the_strong_identity_selector(self) -> None:
        target = self._resolver(user_id="usr_agent").resolve()

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.identity.display_name, "AgentAvatar")

    def test_returns_none_while_the_requested_account_is_not_running(self) -> None:
        self.assertIsNone(self._resolver(user_name="SomeoneElse").resolve())

    def test_diagnostics_explain_the_nearest_log_for_each_live_process(self) -> None:
        diagnostics = self._resolver(user_name="ObserverAvatar").diagnostics()

        self.assertEqual(diagnostics["selector"], "ObserverAvatar")
        self.assertEqual(diagnostics["window_count"], 2)
        self.assertEqual(diagnostics["session_count"], 2)
        candidates = {
            candidate["process_id"]: candidate
            for candidate in diagnostics["processes"]
        }
        self.assertTrue(candidates[10484]["identity_matches"])
        self.assertEqual(
            candidates[10484]["nearest_display_name"],
            "ObserverAvatar",
        )
        self.assertTrue(candidates[10484]["within_start_tolerance"])
        self.assertFalse(candidates[6296]["identity_matches"])

    def test_rejects_multiple_live_processes_for_the_same_account(self) -> None:
        duplicate_started = _local_timestamp("2026-09-03_15-36-25")
        self.windows.append(WindowInfo(303, "VRChat", 7777, 1280, 720, False))
        self.started_at[7777] = duplicate_started
        self._write_log("2026-09-03_15-36-25", "AgentAvatar", "usr_agent")

        with self.assertRaises(AmbiguousVrchatIdentityError):
            self._resolver(user_name="AgentAvatar").resolve()

    def test_target_liveness_rejects_pid_reuse(self) -> None:
        resolver = self._resolver(user_name="AgentAvatar")
        target = resolver.resolve()
        assert target is not None

        self.assertTrue(resolver.is_current(target))
        self.started_at[6296] += 10.0
        self.assertFalse(resolver.is_current(target))

    def _resolver(
        self,
        *,
        user_name: str | None = None,
        user_id: str | None = None,
    ) -> VrchatIdentityResolver:
        return VrchatIdentityResolver(
            user_name=user_name,
            user_id=user_id,
            log_dir=self.log_dir,
            enumerate_windows=lambda **_kwargs: list(self.windows),
            process_started_at=lambda pid: self.started_at[pid],
        )

    def _write_log(self, stamp: str, display_name: str, user_id: str) -> None:
        path = self.log_dir / f"output_log_{stamp}.txt"
        path.write_text(
            f"2026.09.03 15:35:31 Debug - User Authenticated: "
            f"{display_name} ({user_id})\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
