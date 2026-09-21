import math
import struct
import unittest

from vrc_ardy_agent.vrchat_osc import (
    encode_vrchat_axis,
    encode_vrchat_button,
    encode_vrchat_head_tracker_position,
    encode_vrchat_tracker_position,
    encode_vrchat_tracker_rotation,
)


def _decode_vector3(packet: bytes) -> tuple[str, str, tuple[float, float, float]]:
    address_end = packet.index(b"\x00")
    address = packet[:address_end].decode("utf-8")
    type_offset = (address_end + 4) & ~3
    type_end = packet.index(b"\x00", type_offset)
    type_tag = packet[type_offset:type_end].decode("ascii")
    values_offset = (type_end + 4) & ~3
    return address, type_tag, struct.unpack(">fff", packet[values_offset:values_offset + 12])


class VrchatOscTests(unittest.TestCase):
    def test_axis_packet(self):
        packet = encode_vrchat_axis("Vertical", 0.5)
        self.assertTrue(packet.startswith(b"/input/Vertical\x00"))
        self.assertIn(b",f\x00\x00", packet)
        self.assertTrue(packet.endswith(struct.pack(">f", 0.5)))

    def test_button_packet(self):
        packet = encode_vrchat_button("UseRight", True)
        self.assertTrue(packet.startswith(b"/input/UseRight\x00"))
        self.assertIn(b",i\x00\x00", packet)
        self.assertTrue(packet.endswith(struct.pack(">i", 1)))

    def test_axis_range_is_validated(self):
        with self.assertRaises(ValueError):
            encode_vrchat_axis("Vertical", 1.01)

    def test_body_tracker_position_packet(self):
        packet = encode_vrchat_tracker_position(3, (1.25, -2.5, 3.75))

        address, type_tag, values = _decode_vector3(packet)

        self.assertEqual(address, "/tracking/trackers/3/position")
        self.assertEqual(type_tag, ",fff")
        self.assertEqual(values, (1.25, -2.5, 3.75))

    def test_body_tracker_rotation_packet(self):
        packet = encode_vrchat_tracker_rotation(8, (10.0, 20.0, -30.0))

        address, type_tag, values = _decode_vector3(packet)

        self.assertEqual(address, "/tracking/trackers/8/rotation")
        self.assertEqual(type_tag, ",fff")
        self.assertEqual(values, (10.0, 20.0, -30.0))

    def test_head_tracker_position_packet(self):
        packet = encode_vrchat_head_tracker_position((0.0, 1.0, 0.25))

        address, type_tag, values = _decode_vector3(packet)

        self.assertEqual(address, "/tracking/trackers/head/position")
        self.assertEqual(type_tag, ",fff")
        self.assertEqual(values, (0.0, 1.0, 0.25))

    def test_tracker_index_and_vector_are_validated(self):
        with self.assertRaises(ValueError):
            encode_vrchat_tracker_position(0, (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            encode_vrchat_tracker_rotation(9, (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            encode_vrchat_head_tracker_position((0.0, math.nan, 0.0))


if __name__ == "__main__":
    unittest.main()
