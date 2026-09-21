from __future__ import annotations

from array import array
import unittest

from vrc_ardy_agent.energy_vad import EnergyVadSegmenter


def _frame(amplitude: int) -> bytes:
    return array("h", [amplitude] * 320).tobytes()


class EnergyVadSegmenterTests(unittest.TestCase):
    def test_emits_one_utterance_after_confirmed_speech_and_silence(self) -> None:
        vad = EnergyVadSegmenter(
            rms_threshold=500,
            start_trigger_ms=40,
            end_silence_ms=60,
            pre_roll_ms=20,
            minimum_speech_ms=40,
            maximum_utterance_ms=1000,
        )
        emitted = []

        for frame in [
            _frame(0),
            _frame(1000),
            _frame(1000),
            _frame(1000),
            _frame(0),
            _frame(0),
            _frame(0),
        ]:
            emitted.extend(vad.push(frame))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0]), 7 * len(_frame(0)))
        snapshot = vad.snapshot()
        self.assertFalse(snapshot.active)
        self.assertEqual(snapshot.utterances_emitted, 1)

    def test_short_noise_does_not_create_an_utterance(self) -> None:
        vad = EnergyVadSegmenter(
            rms_threshold=500,
            start_trigger_ms=40,
            end_silence_ms=60,
            pre_roll_ms=20,
            minimum_speech_ms=40,
            maximum_utterance_ms=1000,
        )

        emitted = []
        for frame in [_frame(1000), _frame(0), _frame(0), _frame(0)]:
            emitted.extend(vad.push(frame))

        self.assertEqual(emitted, [])
        self.assertEqual(vad.snapshot().utterances_discarded, 0)

    def test_reset_discards_incomplete_audio_on_reconnect(self) -> None:
        vad = EnergyVadSegmenter(
            rms_threshold=500,
            start_trigger_ms=20,
            end_silence_ms=60,
            pre_roll_ms=0,
            minimum_speech_ms=20,
            maximum_utterance_ms=1000,
        )
        vad.push(_frame(1000))

        vad.reset(reason="session changed")

        self.assertFalse(vad.snapshot().active)
        self.assertEqual(vad.snapshot().resets, 1)
        self.assertEqual(vad.snapshot().last_reset_reason, "session changed")

    def test_maximum_duration_forces_a_bounded_utterance(self) -> None:
        vad = EnergyVadSegmenter(
            rms_threshold=500,
            start_trigger_ms=20,
            end_silence_ms=60,
            pre_roll_ms=0,
            minimum_speech_ms=20,
            maximum_utterance_ms=60,
        )
        emitted = []

        for _ in range(3):
            emitted.extend(vad.push(_frame(1000)))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0]), 3 * len(_frame(0)))


if __name__ == "__main__":
    unittest.main()
