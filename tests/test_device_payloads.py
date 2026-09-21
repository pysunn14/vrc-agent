from __future__ import annotations

import unittest

from vrc_ardy_agent.device_payloads import (
    DevicePayloadError,
    decode_pose_payload,
    decode_wav_payload,
    encode_pose_payload,
    encode_wav_payload,
)
from vrc_ardy_agent.opentrack_bridge import OpenTrackFrame
from vrc_ardy_agent.six_point_bridge import SixPointFrame, TrackerActivation
from vrc_ardy_agent.vmt_bridge import VmtFrame


def _frame() -> SixPointFrame:
    tracker = VmtFrame(
        position=(0.1, 1.2, -0.3),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        fps=20.0,
    )
    return SixPointFrame(
        head=OpenTrackFrame(
            xyz_cm=(1.0, 2.0, 3.0),
            ypr_deg=(4.0, 5.0, 6.0),
            fps=20.0,
        ),
        left=tracker,
        right=tracker,
        hips=tracker,
        left_foot=tracker,
        right_foot=tracker,
        fps=20.0,
        scale=0.75,
        locomotion_x=-0.25,
        locomotion_y=0.5,
        locomotion_turn=0.1,
    )


class DevicePayloadTests(unittest.TestCase):
    def test_six_point_pose_round_trip(self) -> None:
        frame = _frame()

        self.assertEqual(decode_pose_payload(encode_pose_payload(frame)), frame)

    def test_tracker_activation_round_trip_preserves_disabled_hands(self) -> None:
        frame = _frame()
        frame = SixPointFrame(
            head=frame.head,
            left=frame.left,
            right=frame.right,
            hips=frame.hips,
            left_foot=frame.left_foot,
            right_foot=frame.right_foot,
            fps=frame.fps,
            scale=frame.scale,
            locomotion_x=frame.locomotion_x,
            locomotion_y=frame.locomotion_y,
            locomotion_turn=frame.locomotion_turn,
            tracker_activation=TrackerActivation(left=False, right=False),
        )

        decoded = decode_pose_payload(encode_pose_payload(frame))

        self.assertEqual(decoded.tracker_activation, frame.tracker_activation)

    def test_wav_round_trip(self) -> None:
        wav = b"RIFF-example-wave"

        self.assertEqual(decode_wav_payload(encode_wav_payload(wav)), wav)

    def test_pose_rejects_non_finite_numbers_and_extra_fields(self) -> None:
        payload = encode_pose_payload(_frame())
        payload["fps"] = float("nan")
        with self.assertRaisesRegex(DevicePayloadError, "finite"):
            decode_pose_payload(payload)

        payload = encode_pose_payload(_frame())
        payload["extra"] = True
        with self.assertRaisesRegex(DevicePayloadError, "fields"):
            decode_pose_payload(payload)

    def test_wav_rejects_invalid_base64(self) -> None:
        with self.assertRaisesRegex(DevicePayloadError, "base64"):
            decode_wav_payload({"kind": "wav", "data": "%%%"})


if __name__ == "__main__":
    unittest.main()
