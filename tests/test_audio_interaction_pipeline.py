from __future__ import annotations

from array import array
import threading
import unittest

from vrc_ardy_agent.audio_interaction_pipeline import AudioInteractionPipeline
from vrc_ardy_agent.dialogue_turns import TurnToken
from vrc_ardy_agent.energy_vad import EnergyVadSegmenter
from vrc_ardy_agent.interaction_runtime import TurnOutcome, TurnOutcomeStatus
from vrc_ardy_agent.stream_protocol import AudioChunkMessage


def _frame(amplitude: int) -> bytes:
    return array("h", [amplitude] * 320).tobytes()


class _Stt:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls = []

    def transcribe_pcm(self, pcm: bytes) -> str:
        self.calls.append(pcm)
        return self.transcript


class _Runtime:
    def __init__(self) -> None:
        self.begun = []
        self.handled = []
        self.failed = []
        self.event = threading.Event()

    def begin_utterance(self) -> TurnToken:
        token = TurnToken(f"turn-{len(self.begun) + 1}", len(self.begun) + 1)
        self.begun.append(token)
        return token

    def handle_reserved_utterance(self, token, *, transcript, screenshot):
        self.handled.append((token, transcript, screenshot))
        self.event.set()
        return TurnOutcome(
            turn_id=token.turn_id,
            turn_version=token.version,
            status=(
                TurnOutcomeStatus.FAST_STOP_REQUESTED
                if transcript == "멈춰"
                else TurnOutcomeStatus.APPLIED
            ),
        )

    def fail_reserved_utterance(self, token, *, error):
        self.failed.append((token, error))
        self.event.set()
        return TurnOutcome(
            turn_id=token.turn_id,
            turn_version=token.version,
            status=TurnOutcomeStatus.FAILED,
            error=error,
        )


def _vad() -> EnergyVadSegmenter:
    return EnergyVadSegmenter(
        rms_threshold=500,
        start_trigger_ms=20,
        end_silence_ms=40,
        pre_roll_ms=0,
        minimum_speech_ms=20,
        maximum_utterance_ms=1000,
    )


def _message(sequence: int, amplitude: int, *, session: str = "session-a"):
    return AudioChunkMessage(
        session_id=session,
        sequence=sequence,
        captured_monotonic_ns=sequence,
        pcm=_frame(amplitude),
    )


class AudioInteractionPipelineTests(unittest.TestCase):
    def test_completed_utterance_passes_text_without_eager_capture(self) -> None:
        runtime = _Runtime()
        stt = _Stt("춤춰 줘")
        pipeline = AudioInteractionPipeline(
            vad=_vad(),
            stt=stt,
            runtime=runtime,
        )
        self.addCleanup(pipeline.close)

        for sequence, amplitude in enumerate((1000, 0, 0)):
            pipeline.accept_audio(_message(sequence, amplitude))

        self.assertTrue(runtime.event.wait(timeout=1.0))
        self.assertTrue(pipeline.wait_until_idle(timeout=1.0))
        self.assertEqual(len(runtime.begun), 1)
        self.assertEqual(runtime.handled[0][1:], ("춤춰 줘", None))
        self.assertEqual(len(stt.calls), 1)
        snapshot = pipeline.snapshot()
        self.assertEqual(snapshot.last_utterance_audio_ms, 60)
        self.assertGreaterEqual(snapshot.last_stt_latency_ms, 0)
        self.assertGreaterEqual(snapshot.last_turn_handling_latency_ms, 0)
        self.assertGreaterEqual(snapshot.last_pipeline_latency_ms, 0)

    def test_fast_stop_does_not_wait_for_screenshot(self) -> None:
        runtime = _Runtime()
        pipeline = AudioInteractionPipeline(
            vad=_vad(),
            stt=_Stt("멈춰"),
            runtime=runtime,
        )
        self.addCleanup(pipeline.close)

        for sequence, amplitude in enumerate((1000, 0, 0)):
            pipeline.accept_audio(_message(sequence, amplitude))

        self.assertTrue(runtime.event.wait(timeout=0.2))
        self.assertEqual(runtime.handled[0][1:], ("멈춰", None))
        self.assertTrue(pipeline.wait_until_idle(timeout=1.0))

    def test_audio_sequence_gap_discards_the_incomplete_vad_segment(self) -> None:
        runtime = _Runtime()
        vad = _vad()
        pipeline = AudioInteractionPipeline(
            vad=vad,
            stt=_Stt("unused"),
            runtime=runtime,
        )
        self.addCleanup(pipeline.close)

        pipeline.accept_audio(_message(0, 1000))
        pipeline.accept_audio(_message(2, 0))
        pipeline.accept_audio(_message(3, 0))

        self.assertEqual(runtime.begun, [])
        self.assertEqual(pipeline.snapshot().audio_discontinuities, 1)
        self.assertEqual(vad.snapshot().last_reset_reason, "audio discontinuity")

    def test_stt_failure_is_attributed_to_stt(self) -> None:
        class FailingStt:
            def transcribe_pcm(self, _pcm):
                raise RuntimeError("decoder unavailable")

        runtime = _Runtime()
        pipeline = AudioInteractionPipeline(
            vad=_vad(),
            stt=FailingStt(),
            runtime=runtime,
        )
        self.addCleanup(pipeline.close)

        for sequence, amplitude in enumerate((1000, 0, 0)):
            pipeline.accept_audio(_message(sequence, amplitude))

        self.assertTrue(pipeline.wait_until_idle(timeout=1.0))
        self.assertEqual(len(runtime.begun), 1)
        self.assertEqual(len(runtime.failed), 1)
        snapshot = pipeline.snapshot()
        self.assertEqual(snapshot.utterances_failed, 1)
        self.assertIn("STT failed", snapshot.last_error)
        self.assertIn("decoder unavailable", snapshot.last_error)

    def test_older_slow_transcript_cannot_overtake_a_newer_transcript(self) -> None:
        class OrderedStt:
            def __init__(self) -> None:
                self.calls = 0
                self.first_started = threading.Event()
                self.release_first = threading.Event()

            def transcribe_pcm(self, _pcm):
                self.calls += 1
                if self.calls == 1:
                    self.first_started.set()
                    self.release_first.wait(timeout=1.0)
                    return "오래된 요청"
                return "새 요청"

        runtime = _Runtime()
        stt = OrderedStt()
        pipeline = AudioInteractionPipeline(
            vad=_vad(),
            stt=stt,
            runtime=runtime,
        )
        self.addCleanup(pipeline.close)

        for sequence, amplitude in enumerate((1000, 0, 0)):
            pipeline.accept_audio(_message(sequence, amplitude))
        self.assertTrue(stt.first_started.wait(timeout=1.0))
        for sequence, amplitude in enumerate((1000, 0, 0), start=3):
            pipeline.accept_audio(_message(sequence, amplitude))

        self.assertTrue(runtime.event.wait(timeout=1.0))
        stt.release_first.set()
        self.assertTrue(pipeline.wait_until_idle(timeout=1.0))

        self.assertEqual(len(runtime.begun), 2)
        self.assertEqual([item[1] for item in runtime.handled], ["새 요청"])
        snapshot = pipeline.snapshot()
        self.assertEqual(snapshot.utterances_superseded, 1)
        self.assertEqual(snapshot.last_transcript, "새 요청")
        self.assertEqual(snapshot.last_outcome, "APPLIED")

    def test_runtime_failure_is_not_mislabeled_as_stt_failure(self) -> None:
        class FailingRuntime(_Runtime):
            def handle_reserved_utterance(self, token, *, transcript, screenshot):
                raise RuntimeError("dispatcher unavailable")

        runtime = FailingRuntime()
        pipeline = AudioInteractionPipeline(
            vad=_vad(),
            stt=_Stt("춤춰 줘"),
            runtime=runtime,
        )
        self.addCleanup(pipeline.close)

        for sequence, amplitude in enumerate((1000, 0, 0)):
            pipeline.accept_audio(_message(sequence, amplitude))

        self.assertTrue(runtime.event.wait(timeout=1.0))
        self.assertTrue(pipeline.wait_until_idle(timeout=1.0))
        self.assertIn("turn handling failed", runtime.failed[0][1])
        self.assertNotIn("STT failed", runtime.failed[0][1])


if __name__ == "__main__":
    unittest.main()
