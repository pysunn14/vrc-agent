from __future__ import annotations

import unittest

from vrc_ardy_agent.follow_protocol import (
    FollowProtocolError,
    TargetObservation,
    decode_target_observation,
    encode_target_observation,
)


class FollowProtocolTests(unittest.TestCase):
    def test_visible_observation_round_trips_as_normalized_full_state(self):
        observation = TargetObservation(
            session_id="session-a",
            sequence=12,
            captured_at_ns=123_456_789,
            visible=True,
            bbox=(0.2, 0.1, 0.6, 0.9),
            confidence=0.93,
        )

        decoded = decode_target_observation(encode_target_observation(observation))

        self.assertEqual(decoded, observation)
        self.assertAlmostEqual(decoded.center_x, 0.4)
        self.assertAlmostEqual(decoded.height, 0.8)

    def test_invisible_observation_has_no_stale_target_geometry(self):
        observation = TargetObservation(
            session_id="session-a",
            sequence=13,
            captured_at_ns=123_456_999,
            visible=False,
            bbox=None,
            confidence=0.0,
        )

        decoded = decode_target_observation(encode_target_observation(observation))

        self.assertFalse(decoded.visible)
        self.assertIsNone(decoded.bbox)

    def test_invalid_or_unsupported_packets_are_rejected(self):
        invalid_packets = (
            b"not-json",
            b'{"version":2}',
            (
                b'{"version":1,"session":"s","seq":0,"captured_at_ns":0,'
                b'"visible":true,"bbox":[0.8,0.1,0.2,0.9],"confidence":0.9}'
            ),
            (
                b'{"version":1,"session":"s","seq":0,"captured_at_ns":0,'
                b'"visible":false,"bbox":[0.1,0.1,0.2,0.2],"confidence":0.0}'
            ),
        )

        for packet in invalid_packets:
            with self.subTest(packet=packet):
                with self.assertRaises(FollowProtocolError):
                    decode_target_observation(packet)


if __name__ == "__main__":
    unittest.main()
