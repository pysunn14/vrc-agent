from __future__ import annotations

import struct
import unittest

from vrc_ardy_agent.vmt_input import (
    encode_vmt_button,
    encode_vmt_joystick,
    encode_vmt_trigger,
)


def read_osc_string(packet: bytes, offset: int = 0):
    end = packet.index(b"\x00", offset)
    value = packet[offset:end].decode("utf-8")
    end += 1
    while end % 4:
        end += 1
    return value, end


class VmtInputTests(unittest.TestCase):
    def test_joystick_packet(self):
        packet = encode_vmt_joystick(index=1, joystick_index=0, timeoffset=0.0, x=-0.5, y=1.0)
        address, off = read_osc_string(packet)
        tags, off = read_osc_string(packet, off)
        self.assertEqual(address, "/VMT/Input/Joystick")
        self.assertEqual(tags, ",iifff")
        self.assertEqual(struct.unpack(">ii", packet[off : off + 8]), (1, 0))
        values = struct.unpack(">fff", packet[off + 8 : off + 20])
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], -0.5)
        self.assertAlmostEqual(values[2], 1.0)

    def test_trigger_packet(self):
        packet = encode_vmt_trigger(index=2, trigger_index=0, timeoffset=0.0, value=0.75)
        address, off = read_osc_string(packet)
        tags, off = read_osc_string(packet, off)
        self.assertEqual(address, "/VMT/Input/Trigger")
        self.assertEqual(tags, ",iiff")
        self.assertEqual(struct.unpack(">ii", packet[off : off + 8]), (2, 0))
        timeoffset, value = struct.unpack(">ff", packet[off + 8 : off + 16])
        self.assertAlmostEqual(timeoffset, 0.0)
        self.assertAlmostEqual(value, 0.75)

    def test_button_packet(self):
        packet = encode_vmt_button(index=2, button_index=1, timeoffset=0.0, pressed=True)
        address, off = read_osc_string(packet)
        tags, off = read_osc_string(packet, off)
        self.assertEqual(address, "/VMT/Input/Button")
        self.assertEqual(tags, ",iifi")
        self.assertEqual(struct.unpack(">ii", packet[off : off + 8]), (2, 1))
        timeoffset = struct.unpack(">f", packet[off + 8 : off + 12])[0]
        value = struct.unpack(">i", packet[off + 12 : off + 16])[0]
        self.assertAlmostEqual(timeoffset, 0.0)
        self.assertEqual(value, 1)

    def test_input_ranges_are_validated(self):
        with self.assertRaises(ValueError):
            encode_vmt_joystick(index=1, joystick_index=0, timeoffset=0.0, x=2.0, y=0.0)
        with self.assertRaises(ValueError):
            encode_vmt_trigger(index=2, trigger_index=0, timeoffset=0.0, value=-0.1)
        with self.assertRaises(ValueError):
            encode_vmt_button(index=2, button_index=8, timeoffset=0.0, pressed=True)


if __name__ == "__main__":
    unittest.main()