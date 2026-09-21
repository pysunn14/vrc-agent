from __future__ import annotations

import unittest

from vrc_ardy_agent.pose_runtime import PoseStreamRunner
from vrc_ardy_agent.tracking_rig import (
    NeutralPoseConfig,
    build_neutral_tracking_frame,
    get_avatar_profile,
)

from tests.test_avatar_rig_profile import _profile_document
from vrc_ardy_agent.avatar_rig_profile import AvatarRigProfile

TEST_PROFILE = AvatarRigProfile.from_mapping(_profile_document())


class _FakeSink:
    def __init__(self) -> None:
        self.frames = []
        self.closed = False

    def send(self, frame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


class _KeyboardInterruptSink(_FakeSink):
    def send(self, frame) -> None:
        raise KeyboardInterrupt


class PoseRuntimeTests(unittest.TestCase):
    def test_runner_streams_a_bounded_pose_and_closes_the_sink(self):
        frame = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(hmd_base=(0.0, 1.0, 0.0)),
        )
        sink = _FakeSink()
        runner = PoseStreamRunner(frame=frame, sink=sink, rate_hz=20.0)

        status = runner.run(max_frames=3, realtime=False)

        self.assertEqual(len(sink.frames), 3)
        self.assertEqual(status.frames_sent, 3)
        self.assertFalse(status.running)
        self.assertTrue(sink.closed)

    def test_runner_emits_observable_heartbeat(self):
        frame = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(hmd_base=(0.0, 1.0, 0.0)),
        )
        sink = _FakeSink()
        runner = PoseStreamRunner(frame=frame, sink=sink, rate_hz=20.0)
        heartbeats = []

        runner.run(
            max_frames=2,
            realtime=False,
            heartbeat_interval_seconds=0.001,
            heartbeat=heartbeats.append,
        )

        self.assertGreaterEqual(len(heartbeats), 1)
        self.assertGreaterEqual(heartbeats[-1].frames_sent, 1)

    def test_keyboard_interrupt_is_not_reported_as_a_runtime_failure(self):
        frame = build_neutral_tracking_frame(
            TEST_PROFILE,
            NeutralPoseConfig(hmd_base=(0.0, 1.0, 0.0)),
        )
        sink = _KeyboardInterruptSink()
        runner = PoseStreamRunner(frame=frame, sink=sink, rate_hz=20.0)

        with self.assertRaises(KeyboardInterrupt):
            runner.run(max_frames=1, realtime=False)

        self.assertIsNone(runner.status.last_error)
        self.assertTrue(sink.closed)


if __name__ == "__main__":
    unittest.main()
