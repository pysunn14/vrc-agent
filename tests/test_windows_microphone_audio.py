from __future__ import annotations

import time
import unittest

import numpy as np

from vrc_ardy_agent.windows_microphone_audio import (
    Pcm48kMonoTo16kMono,
    WasapiMicrophoneAudioSource,
)


def _mono_tone(*, duration_ms: int, frequency_hz: float = 1000.0) -> bytes:
    sample_count = 48_000 * duration_ms // 1000
    timeline = np.arange(sample_count, dtype=np.float64) / 48_000.0
    samples = np.rint(
        0.5 * np.sin(2.0 * np.pi * frequency_hz * timeline) * 32_767.0
    ).astype("<i2")
    return samples.tobytes()


class Pcm48kMonoTo16kMonoTests(unittest.TestCase):
    def test_emits_exact_twenty_millisecond_frames_across_chunks(self) -> None:
        converter = Pcm48kMonoTo16kMono()
        pcm = _mono_tone(duration_ms=40)

        frames = []
        frames.extend(converter.push(pcm[: 317 * 2]))
        frames.extend(converter.push(pcm[317 * 2 : 1001 * 2]))
        frames.extend(converter.push(pcm[1001 * 2 :]))

        self.assertEqual([len(frame) for frame in frames], [640, 640])
        samples = np.frombuffer(b"".join(frames), dtype="<i2")
        self.assertGreater(
            float(np.sqrt(np.mean(samples.astype(float) ** 2))),
            5_000,
        )

    def test_rejects_partial_source_samples(self) -> None:
        converter = Pcm48kMonoTo16kMono()

        with self.assertRaisesRegex(ValueError, "complete mono"):
            converter.push(b"\x00")


class _FakeInputStream:
    def __init__(self, *, device, callback) -> None:
        self.device = device
        self.callback = callback
        self.active = False
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.active = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.active = False

    def close(self) -> None:
        self.close_calls += 1

    def emit(self, pcm: bytes, *, status="") -> None:
        self.callback(pcm, len(pcm) // 2, object(), status)


class WasapiMicrophoneAudioSourceTests(unittest.TestCase):
    def test_publishes_normalized_frames_and_exposes_signal_level(self) -> None:
        streams = []
        published = []

        def factory(device, callback):
            stream = _FakeInputStream(device=device, callback=callback)
            streams.append(stream)
            return stream

        source = WasapiMicrophoneAudioSource(
            device=12,
            frame_handler=lambda frame, timestamp: published.append((frame, timestamp)),
            stream_factory=factory,
            monotonic_ns=lambda: 99,
            monitor_interval_seconds=0.01,
        )

        source.start()
        streams[0].emit(_mono_tone(duration_ms=40))
        source.close()

        self.assertEqual(streams[0].device, 12)
        self.assertEqual([len(frame) for frame, _ in published], [640, 640])
        self.assertEqual([timestamp for _, timestamp in published], [99, 99])
        snapshot = source.snapshot()
        self.assertFalse(snapshot.running)
        self.assertEqual(snapshot.chunks_received, 1)
        self.assertEqual(snapshot.frames_published, 2)
        self.assertGreater(snapshot.peak_sample, 10_000)
        self.assertGreater(snapshot.rms_sample, 5_000)
        self.assertEqual(streams[0].stop_calls, 1)
        self.assertEqual(streams[0].close_calls, 1)

    def test_callback_and_overflow_failures_are_observable(self) -> None:
        streams = []

        def factory(device, callback):
            stream = _FakeInputStream(device=device, callback=callback)
            streams.append(stream)
            return stream

        source = WasapiMicrophoneAudioSource(
            device=12,
            frame_handler=lambda _frame, _timestamp: (_ for _ in ()).throw(
                RuntimeError("network down")
            ),
            stream_factory=factory,
        )
        self.addCleanup(source.close)
        source.start()

        streams[0].emit(_mono_tone(duration_ms=20), status="input overflow")

        snapshot = source.snapshot()
        self.assertEqual(snapshot.overflow_events, 1)
        self.assertEqual(snapshot.publish_failures, 1)
        self.assertIn("network down", snapshot.last_error)

    def test_monitor_exposes_an_unexpected_stream_stop(self) -> None:
        streams = []

        def factory(device, callback):
            stream = _FakeInputStream(device=device, callback=callback)
            streams.append(stream)
            return stream

        source = WasapiMicrophoneAudioSource(
            device=12,
            frame_handler=lambda _frame, _timestamp: None,
            stream_factory=factory,
            monitor_interval_seconds=0.01,
        )
        self.addCleanup(source.close)
        source.start()
        streams[0].active = False

        deadline = time.monotonic() + 1.0
        while source.snapshot().running and time.monotonic() < deadline:
            time.sleep(0.01)

        snapshot = source.snapshot()
        self.assertFalse(snapshot.running)
        self.assertIn("stopped unexpectedly", snapshot.last_error)


if __name__ == "__main__":
    unittest.main()
