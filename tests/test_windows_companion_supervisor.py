from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import unittest

from vrc_ardy_agent.windows_capture import WindowInfo
from vrc_ardy_agent.windows_companion_supervisor import (
    WindowsCompanionState,
    WindowsCompanionSupervisor,
)
from vrc_ardy_agent.windows_vrchat_identity import (
    AmbiguousVrchatIdentityError,
    VrchatIdentity,
    VrchatTarget,
)


@dataclass(frozen=True)
class _RuntimeSnapshot:
    running: bool
    components: dict[str, object]
    heartbeat_monotonic: float
    last_errors: tuple[str, ...] = ()


class _Runtime:
    def __init__(
        self,
        target: VrchatTarget,
        *,
        fail_stop: bool = False,
    ) -> None:
        self.target = target
        self.fail_stop = fail_stop
        self.started = 0
        self.stopped = 0
        self.running = False

    def start(self) -> None:
        self.started += 1
        self.running = True

    def stop(self) -> None:
        self.stopped += 1
        self.running = False
        if self.fail_stop:
            raise RuntimeError("release failed")

    def snapshot(self) -> _RuntimeSnapshot:
        return _RuntimeSnapshot(
            running=self.running,
            components={"audio": {"running": self.running}},
            heartbeat_monotonic=time.monotonic(),
        )


def _target(pid: int, started_at: float) -> VrchatTarget:
    return VrchatTarget(
        window=WindowInfo(pid + 100, "VRChat", pid, 1280, 720, False),
        process_started_at=started_at,
        log_path=f"output_log_{pid}.txt",
        identity=VrchatIdentity("AgentAvatar", "usr_agent"),
    )


class WindowsCompanionSupervisorTests(unittest.TestCase):
    def test_process_exit_stops_old_epoch_and_attaches_new_pid(self) -> None:
        first = _target(1001, 10.0)
        second = _target(2002, 20.0)
        selected = [first]
        alive = {first.generation_key: True, second.generation_key: True}
        runtimes: list[_Runtime] = []

        def factory(target: VrchatTarget, _epoch: int) -> _Runtime:
            runtime = _Runtime(target)
            runtimes.append(runtime)
            return runtime

        supervisor = WindowsCompanionSupervisor(
            resolve_target=lambda: selected[0],
            target_is_current=lambda target: alive[target.generation_key],
            runtime_factory=factory,
            poll_interval_seconds=0.01,
            retry_delay_seconds=0.01,
        )
        self.addCleanup(supervisor.stop)

        supervisor.start()
        self._wait_until(lambda: len(runtimes) == 1 and runtimes[0].running)
        selected[0] = second
        alive[first.generation_key] = False
        self._wait_until(lambda: len(runtimes) == 2 and runtimes[1].running)

        snapshot = supervisor.snapshot()
        self.assertEqual(runtimes[0].stopped, 1)
        self.assertEqual(snapshot.state, WindowsCompanionState.RUNNING)
        self.assertEqual(snapshot.epoch, 2)
        self.assertEqual(snapshot.target["pid"], 2002)
        self.assertEqual(snapshot.successful_attaches, 2)
        self.assertEqual(snapshot.detachments, 1)
        self.assertIn("target process changed", snapshot.last_detach_reason)

    def test_ambiguity_is_observable_and_does_not_start_a_runtime(self) -> None:
        attempted = threading.Event()

        def resolve():
            attempted.set()
            raise AmbiguousVrchatIdentityError("two AgentAvatar processes")

        supervisor = WindowsCompanionSupervisor(
            resolve_target=resolve,
            target_is_current=lambda _target: True,
            runtime_factory=lambda _target, _epoch: self.fail("must not start"),
            poll_interval_seconds=0.01,
            retry_delay_seconds=0.01,
        )
        self.addCleanup(supervisor.stop)

        supervisor.start()
        self.assertTrue(attempted.wait(1.0))
        self._wait_until(
            lambda: supervisor.snapshot().state
            == WindowsCompanionState.AMBIGUOUS
        )

        snapshot = supervisor.snapshot()
        self.assertTrue(snapshot.running)
        self.assertIn("two AgentAvatar processes", snapshot.last_error)
        self.assertGreaterEqual(snapshot.resolution_failures, 1)

    def test_stop_is_idempotent_while_waiting_for_target(self) -> None:
        supervisor = WindowsCompanionSupervisor(
            resolve_target=lambda: None,
            target_is_current=lambda _target: False,
            runtime_factory=lambda _target, _epoch: self.fail("must not start"),
            poll_interval_seconds=0.01,
            retry_delay_seconds=0.01,
        )

        supervisor.start()
        self._wait_until(
            lambda: supervisor.snapshot().state == WindowsCompanionState.WAITING
        )
        supervisor.stop()
        supervisor.stop()

        snapshot = supervisor.snapshot()
        self.assertFalse(snapshot.running)
        self.assertEqual(snapshot.state, WindowsCompanionState.STOPPED)

    def test_runtime_factory_failure_is_observable_and_retried(self) -> None:
        target = _target(1001, 10.0)
        attempts = 0
        runtime = _Runtime(target)

        def factory(_target: VrchatTarget, _epoch: int) -> _Runtime:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("device unavailable")
            return runtime

        supervisor = WindowsCompanionSupervisor(
            resolve_target=lambda: target,
            target_is_current=lambda _target: True,
            runtime_factory=factory,
            poll_interval_seconds=0.01,
            retry_delay_seconds=0.1,
        )
        self.addCleanup(supervisor.stop)

        supervisor.start()
        self._wait_until(
            lambda: supervisor.snapshot().state
            == WindowsCompanionState.RETRYING
        )
        failed = supervisor.snapshot()
        self.assertTrue(failed.running)
        self.assertIn("device unavailable", failed.last_error)

        self._wait_until(lambda: runtime.running)
        recovered = supervisor.snapshot()
        self.assertEqual(attempts, 2)
        self.assertEqual(recovered.epoch, 1)
        self.assertEqual(recovered.successful_attaches, 1)

    def test_target_inspection_failure_detaches_without_killing_supervisor(self) -> None:
        target = _target(1001, 10.0)
        runtimes: list[_Runtime] = []
        inspection_calls = 0

        def inspect(_target: VrchatTarget) -> bool:
            nonlocal inspection_calls
            inspection_calls += 1
            if inspection_calls == 1:
                raise OSError("inspection denied")
            return True

        def factory(selected: VrchatTarget, _epoch: int) -> _Runtime:
            runtime = _Runtime(selected)
            runtimes.append(runtime)
            return runtime

        supervisor = WindowsCompanionSupervisor(
            resolve_target=lambda: target,
            target_is_current=inspect,
            runtime_factory=factory,
            poll_interval_seconds=0.01,
            retry_delay_seconds=0.01,
        )
        self.addCleanup(supervisor.stop)

        supervisor.start()
        self._wait_until(lambda: len(runtimes) == 2 and runtimes[1].running)

        snapshot = supervisor.snapshot()
        self.assertTrue(snapshot.running)
        self.assertEqual(snapshot.epoch, 2)
        self.assertEqual(snapshot.detachments, 1)
        self.assertIn("inspection denied", snapshot.last_detach_reason)

    def test_runtime_stop_failure_remains_observable_after_detach(self) -> None:
        target = _target(1001, 10.0)
        alive = True
        runtime = _Runtime(target, fail_stop=True)

        supervisor = WindowsCompanionSupervisor(
            resolve_target=lambda: target,
            target_is_current=lambda _target: alive,
            runtime_factory=lambda _target, _epoch: runtime,
            poll_interval_seconds=0.01,
            retry_delay_seconds=1.0,
        )

        supervisor.start()
        self._wait_until(lambda: runtime.running)
        alive = False
        self._wait_until(lambda: supervisor.snapshot().detachments == 1)

        snapshot = supervisor.snapshot()
        self.assertIn("epoch stop failed", snapshot.last_error)
        self.assertIn("release failed", snapshot.last_error)
        supervisor.stop()

    def test_reading_snapshot_does_not_advance_supervisor_heartbeat(self) -> None:
        monotonic_calls = 0

        def monotonic() -> float:
            nonlocal monotonic_calls
            monotonic_calls += 1
            return float(monotonic_calls)

        supervisor = WindowsCompanionSupervisor(
            resolve_target=lambda: None,
            target_is_current=lambda _target: False,
            runtime_factory=lambda _target, _epoch: self.fail("must not start"),
            monotonic=monotonic,
        )

        first = supervisor.snapshot()
        second = supervisor.snapshot()

        self.assertEqual(first.heartbeat_monotonic, second.heartbeat_monotonic)
        self.assertEqual(monotonic_calls, 1)

    @staticmethod
    def _wait_until(predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not predicate():
            raise AssertionError("condition was not reached before timeout")


if __name__ == "__main__":
    unittest.main()
