import struct
import unittest

from vrc_ardy_agent.vrchat_osc import encode_vrchat_axis, encode_vrchat_button


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


if __name__ == "__main__":
    unittest.main()