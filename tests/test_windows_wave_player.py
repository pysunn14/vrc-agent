from __future__ import annotations

from io import BytesIO
import threading
import unittest
import wave

from vrc_ardy_agent.windows_wave_player import WaveAudioPlayer


def _wav(frame_count: int = 1600) -> bytes:
    result = BytesIO()
    with wave.open(result, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * frame_count)
    return result.getvalue()


class _Stream:
    def __init__(self, **options) -> None:
        self.options = options
        self.writes = []
        self.entered = False
        self.abort_calls = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def abort(self) -> None:
        self.abort_calls += 1


class WaveAudioPlayerTests(unittest.TestCase):
    def test_plays_wav_on_the_selected_output_and_reports_completion(self) -> None:
        streams = []

        def factory(**options):
            stream = _Stream(**options)
            streams.append(stream)
            return stream

        player = WaveAudioPlayer(
            device="CABLE Input",
            stream_factory=factory,
            chunk_frames=400,
        )
        completed = []

        player.play(_wav(), lambda state, error: completed.append((state, error)))
        self.assertTrue(player.wait_until_idle(timeout=1.0))

        self.assertEqual(completed, [("completed", None)])
        self.assertEqual(streams[0].options["device"], "CABLE Input")
        self.assertEqual(streams[0].options["samplerate"], 16000)
        self.assertEqual(streams[0].options["channels"], 1)
        self.assertEqual(streams[0].options["dtype"], "int16")
        self.assertEqual(sum(len(chunk) for chunk in streams[0].writes), 3200)

    def test_stop_reports_cancelled_and_waits_for_the_worker(self) -> None:
        entered_write = threading.Event()
        release_write = threading.Event()

        class BlockingStream(_Stream):
            def write(self, data: bytes) -> None:
                entered_write.set()
                release_write.wait(timeout=1.0)
                super().write(data)

            def abort(self) -> None:
                super().abort()
                release_write.set()

        stream = BlockingStream()
        player = WaveAudioPlayer(stream_factory=lambda **_options: stream)
        completed = []
        player.play(_wav(3200), lambda state, error: completed.append((state, error)))
        self.assertTrue(entered_write.wait(timeout=1.0))

        player.stop()

        self.assertTrue(player.wait_until_idle(timeout=1.0))
        self.assertEqual(completed, [("cancelled", None)])
        self.assertEqual(stream.abort_calls, 1)


if __name__ == "__main__":
    unittest.main()
