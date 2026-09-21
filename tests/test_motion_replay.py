from dataclasses import replace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from vrc_ardy_agent.opentrack_bridge import OpenTrackFrame
from vrc_ardy_agent.six_point_bridge import SixPointFrame
from vrc_ardy_agent.vmt_bridge import VmtFrame
from vrc_ardy_agent.pose_transition import PoseTransition
from vrc_ardy_agent.motion_replay import MotionReplayRunner


def pose(x=0., yaw=0.):
    tracker = VmtFrame((x, 1., 0.), (0., 0., 0., 1.), 20.)
    return SixPointFrame(OpenTrackFrame((x, 0., 0.), (yaw, 0., 0.), 20.),
                         tracker, tracker, tracker, tracker, tracker,
                         20., 1., 0., 0., 0.)


class Sink:
    def __init__(self):
        self.frames = []
        self.closed = 0
        self.callback = lambda: None

    def send(self, frame):
        self.frames.append(frame)
        self.callback()

    def close(self):
        self.closed += 1


class TransitionTests(unittest.TestCase):
    def test_exact_endpoints_and_eased_position(self):
        a, b = pose(), pose(1.)
        blend = PoseTransition(a, b)
        self.assertIs(blend.at(0), a)
        self.assertIs(blend.at(1), b)
        self.assertAlmostEqual(blend.at(.25).left.position[0], .15625)
        self.assertAlmostEqual(blend.at(.5).left.position[0], .5)

    def test_quaternion_sign_and_shortest_arc(self):
        a = pose()
        b = replace(a, left=replace(a.left, quaternion_xyzw=(0, 0, 0, -1)))
        middle = PoseTransition(a, b).at(.5)
        np.testing.assert_allclose(Rotation.from_quat(middle.left.quaternion_xyzw).as_matrix(), np.eye(3))
        a = replace(a, left=replace(a.left, quaternion_xyzw=tuple(Rotation.from_euler('y', 170, degrees=True).as_quat())))
        b = replace(b, left=replace(b.left, quaternion_xyzw=tuple(Rotation.from_euler('y', -170, degrees=True).as_quat())))
        angle = Rotation.from_quat(PoseTransition(a, b).at(.5).left.quaternion_xyzw).magnitude()
        self.assertAlmostEqual(angle, np.pi)

    def test_head_wrap_uses_rotation_not_euler_lerp(self):
        middle = PoseTransition(pose(yaw=170), pose(yaw=-170)).at(.5)
        self.assertAlmostEqual(abs(middle.head.ypr_deg[0]), 180.)

    def test_invalid_pose_and_progress_fail(self):
        a = pose()
        for b in [replace(a, left=replace(a.left, quaternion_xyzw=(0, 0, 0, 0))),
                  replace(a, head=replace(a.head, xyz_cm=(float('nan'), 0, 0))),
                  replace(a, tracker_activation=replace(a.tracker_activation, left=False))]:
            with self.assertRaises(ValueError):
                PoseTransition(a, b)
        for progress in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                PoseTransition(a, a).at(progress)


class ReplayTests(unittest.TestCase):
    def test_complete_blend_then_hold_without_closing_between_states(self):
        sink = Sink()
        clip, idle = [pose(0), pose(1)], pose(2)
        runner = MotionReplayRunner(frames=clip, idle=idle, sink=sink, transition_seconds=1)
        states = []
        def observe(status):
            self.assertEqual(sink.closed, 0)
            states.append(status.state)
            if status.idle_frames_sent == 3:
                runner.request_stop()
        result = runner.run(realtime=False, heartbeat=observe)
        self.assertEqual(sink.frames[:2], clip)
        self.assertEqual(len(sink.frames), 25)
        self.assertEqual(sink.frames[-4:], [idle] * 4)
        self.assertGreater(sink.frames[2].left.position[0], 1)
        self.assertLess(sink.frames[2].left.position[0], 1.02)
        self.assertIn('returning_idle', states)
        self.assertIn('idle', states)
        self.assertEqual(result.state, 'stopped')
        self.assertEqual(result.motion_frames_sent, 2)
        self.assertEqual(sink.closed, 1)
        with self.assertRaises(RuntimeError):
            runner.run(realtime=False)

    def test_stop_during_return_does_not_claim_idle(self):
        sink = Sink()
        runner = MotionReplayRunner(frames=[pose()], idle=pose(1), sink=sink)
        def observe(status):
            if status.transition_frames_sent == 3:
                runner.request_stop()
                runner.request_stop()
        result = runner.run(realtime=False, heartbeat=observe)
        self.assertEqual(result.idle_frames_sent, 0)
        self.assertEqual(len(sink.frames), 4)
        self.assertEqual(sink.closed, 1)

    def test_send_failure_is_observable_and_closes(self):
        sink = Sink()
        def fail():
            raise OSError('send failed')
        sink.callback = fail
        runner = MotionReplayRunner(frames=[pose()], idle=pose(), sink=sink)
        with self.assertRaisesRegex(OSError, 'send failed'):
            runner.run(realtime=False)
        self.assertEqual(runner.status.state, 'failed')
        self.assertEqual(runner.status.motion_frames_sent, 0)
        self.assertEqual(runner.status.last_error, 'send failed')
        self.assertEqual(sink.closed, 1)

    def test_rejects_empty_clip_and_invalid_duration(self):
        with self.assertRaises(ValueError):
            MotionReplayRunner(frames=[], idle=pose(), sink=Sink())
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                MotionReplayRunner(frames=[pose()], idle=pose(), sink=Sink(), transition_seconds=value)

class ReplayBoundaryTests(unittest.TestCase):
    def test_explicit_stop_before_first_frame_sends_nothing(self):
        sink = Sink()
        runner = MotionReplayRunner(frames=[pose()], idle=pose(), sink=sink)
        runner.request_stop()
        result = runner.run(realtime=False)
        self.assertEqual(sink.frames, [])
        self.assertEqual(result.state, 'stopped')
        self.assertEqual(sink.closed, 1)

    def test_real_udp_encoder_stays_open_until_idle_shutdown(self):
        from vrc_ardy_agent.live_sink import SixPointUdpSink
        from vrc_ardy_agent.opentrack_bridge import encode_opentrack_packet
        class Socket:
            def __init__(self):
                self.packets = []
                self.closed = False
            def sendto(self, packet, destination):
                self.packets.append((packet, destination))
            def close(self):
                self.closed = True
        sock = Socket()
        sink = SixPointUdpSink(host='127.0.0.1', send_locomotion=False,
                               disable_trackers_on_close=True, socket_factory=lambda: sock)
        idle = pose(2)
        runner = MotionReplayRunner(frames=[pose(1)], idle=idle, sink=sink)
        def observe(status):
            if status.idle_frames_sent == 2:
                self.assertFalse(sock.closed)
                self.assertEqual(len(sock.packets), 23 * 6)
                self.assertEqual(sock.packets[-6][0], encode_opentrack_packet(2, 0, 0, 0, 0, 0))
                runner.request_stop()
        runner.run(realtime=False, heartbeat=observe)
        self.assertTrue(sock.closed)
        self.assertEqual(len(sock.packets), 24 * 6)

    def test_cli_dry_run_writes_final_checkpoint_without_opening_socket(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from scripts.agentctl import build_parser
        from vrc_ardy_agent.agentctl_replay import run_replay
        from vrc_ardy_agent.device_payloads import encode_pose_payload
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            idle = root / 'idle.json'
            checkpoint = root / 'status.json'
            idle.write_text(json.dumps(encode_pose_payload(pose())))
            args = build_parser().parse_args([
                'replay', '--motion', 'unused.npz', '--idle-pose', str(idle),
                '--avatar-profile', 'shinano', '--host', '127.0.0.1',
                '--checkpoint', str(checkpoint), '--dry-run',
            ])
            with patch('vrc_ardy_agent.agentctl_replay.load_motion_frames', return_value=[pose(1)]), \
                 patch('vrc_ardy_agent.live_sink.socket.socket') as socket, \
                 patch('builtins.print'):
                run_replay(args)
                socket.assert_not_called()
            status = json.loads(checkpoint.read_text())
            self.assertEqual(status['state'], 'stopped')
            self.assertEqual(status['transition_frames_sent'], 20)
            self.assertEqual(status['idle_frames_sent'], 3)
            self.assertTrue(status['dry_run'])
