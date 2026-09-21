import unittest
from vrc_ardy_agent.face_cue import FaceCueSink


class FaceCueTests(unittest.TestCase):
    def test_one_shot_then_neutral_and_idempotent_close(self):
        class Pose:
            frames = []
            closed = 0

            def send(self, f):
                self.frames.append(f)

            def close(self):
                self.closed += 1

        pose = Pose()
        values = []
        sink = FaceCueSink(pose, [0, 0.8, 0], values.append)
        for f in range(5):
            sink.send(f)
        sink.close()
        sink.close()
        self.assertEqual(values, [0, 0.8, 0, 0, 0, 0])
        self.assertEqual(pose.closed, 1)

    def test_face_reset_failure_still_closes_pose(self):
        class Pose:
            closed = False

            def close(self):
                self.closed = True

        pose = Pose()

        def fail(_):
            raise OSError("face output unavailable")

        sink = FaceCueSink(pose, [0], fail)
        with self.assertRaises(OSError):
            sink.close()
        self.assertTrue(pose.closed)

    def test_rejects_non_neutral_endpoints_and_invalid_values(self):
        for track in ([], [0.5, 0], [0, 0.5], [0, float("nan"), 0], [0, 2, 0]):
            with self.assertRaises(ValueError):
                FaceCueSink(None, track, lambda _: None)
