from __future__ import annotations

import json
import sys
import textwrap
import time
import unittest

import numpy as np

from vrc_ardy_agent.windows_process_audio import (
    NativeProcessAudioSource,
    StreamingFloatPcmTo16kMono,
)


def _tone(
    *,
    sample_rate: int,
    channels: int,
    duration_ms: int,
    frequency_hz: float = 1000.0,
) -> bytes:
    sample_count = sample_rate * duration_ms // 1000
    timeline = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    mono = 0.5 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    interleaved = np.repeat(mono[:, None], channels, axis=1).astype("<f4")
    return interleaved.tobytes()


class StreamingFloatPcmTo16kMonoTests(unittest.TestCase):
    def test_converts_real_vrchat_44100_stereo_across_awkward_chunks(self) -> None:
        converter = StreamingFloatPcmTo16kMono(sample_rate=44_100, channels=2)
        pcm = _tone(sample_rate=44_100, channels=2, duration_ms=40)

        frames = []
        frames.extend(converter.push(pcm[:317]))
        frames.extend(converter.push(pcm[317:8_013]))
        frames.extend(converter.push(pcm[8_013:]))

        self.assertEqual([len(frame) for frame in frames], [640, 640])
        samples = np.frombuffer(b"".join(frames), dtype="<i2")
        self.assertEqual(samples.size, 640)
        self.assertGreater(float(np.sqrt(np.mean(samples.astype(float) ** 2))), 5_000)
        self.assertEqual(converter.source_frames_seen, 1_764)

    def test_converts_48000_mono_without_sample_drift(self) -> None:
        converter = StreamingFloatPcmTo16kMono(sample_rate=48_000, channels=1)

        frames = converter.push(
            _tone(sample_rate=48_000, channels=1, duration_ms=100)
        )

        self.assertEqual(len(frames), 5)
        self.assertEqual(sum(len(frame) for frame in frames), 3_200)

    def test_rejects_invalid_format_and_non_finite_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            StreamingFloatPcmTo16kMono(sample_rate=8_000, channels=2)
        with self.assertRaisesRegex(ValueError, "channels"):
            StreamingFloatPcmTo16kMono(sample_rate=44_100, channels=0)

        converter = StreamingFloatPcmTo16kMono(sample_rate=44_100, channels=2)
        invalid = np.array([[np.nan, 0.0]], dtype="<f4").tobytes()
        with self.assertRaisesRegex(ValueError, "finite"):
            converter.push(invalid)


class NativeProcessAudioSourceTests(unittest.TestCase):
    def test_streams_target_process_audio_and_exposes_native_heartbeat(self) -> None:
        pcm = _tone(sample_rate=44_100, channels=2, duration_ms=40)
        helper = self._fake_helper_command(
            events=[
                {
                    "event": "started",
                    "pid": 1234,
                    "format": {
                        "sample_rate": 44_100,
                        "channels": 2,
                        "bits_per_sample": 32,
                        "sample_format": "float32",
                    },
                },
                {"event": "heartbeat", "packets": 2, "bytes_streamed": len(pcm)},
            ],
            pcm=pcm,
        )
        published: list[tuple[bytes, int]] = []
        source = NativeProcessAudioSource(
            pid=1234,
            frame_handler=lambda frame, timestamp: published.append((frame, timestamp)),
            helper_command=helper,
            startup_timeout_seconds=2.0,
        )
        self.addCleanup(source.close)

        source.start()
        self._wait_until(
            lambda: len(published) == 2
            and source.snapshot().native_heartbeats == 1
        )

        snapshot = source.snapshot()
        self.assertTrue(snapshot.running)
        self.assertEqual(snapshot.pid, 1234)
        self.assertEqual(snapshot.sample_rate, 44_100)
        self.assertEqual(snapshot.channels, 2)
        self.assertEqual(snapshot.frames_published, 2)
        self.assertEqual(snapshot.native_heartbeats, 1)
        self.assertEqual([len(frame) for frame, _ in published], [640, 640])
        self.assertTrue(all(timestamp > 0 for _, timestamp in published))

    def test_rejects_an_unexpected_native_format_before_streaming(self) -> None:
        helper = self._fake_helper_command(
            events=[
                {
                    "event": "started",
                    "pid": 1234,
                    "format": {
                        "sample_rate": 44_100,
                        "channels": 2,
                        "bits_per_sample": 16,
                        "sample_format": "int16",
                    },
                }
            ],
            pcm=b"",
        )
        source = NativeProcessAudioSource(
            pid=1234,
            frame_handler=lambda _frame, _timestamp: None,
            helper_command=helper,
            startup_timeout_seconds=2.0,
        )
        self.addCleanup(source.close)

        with self.assertRaisesRegex(RuntimeError, "float32"):
            source.start()

        self.assertFalse(source.snapshot().running)
        self.assertIn("float32", source.snapshot().last_error)

    def test_native_error_is_reported_instead_of_falling_back(self) -> None:
        helper = self._fake_helper_command(
            events=[
                {
                    "event": "error",
                    "error_type": "InvalidOperationException",
                    "message": "target session vanished",
                }
            ],
            pcm=b"",
        )
        source = NativeProcessAudioSource(
            pid=1234,
            frame_handler=lambda _frame, _timestamp: None,
            helper_command=helper,
            startup_timeout_seconds=2.0,
        )
        self.addCleanup(source.close)

        with self.assertRaisesRegex(RuntimeError, "target session vanished"):
            source.start()

        self.assertIn("target session vanished", source.snapshot().last_error)

    @staticmethod
    def _fake_helper_command(*, events: list[dict[str, object]], pcm: bytes) -> tuple[str, ...]:
        script = textwrap.dedent(
            f"""
            import json
            import sys
            import time

            events = json.loads({json.dumps(json.dumps(events))})
            for event in events:
                sys.stderr.write(json.dumps(event) + "\\n")
                sys.stderr.flush()
            sys.stdout.buffer.write(bytes.fromhex({pcm.hex()!r}))
            sys.stdout.buffer.flush()
            time.sleep(10)
            """
        )
        return (sys.executable, "-u", "-c", script)

    @staticmethod
    def _wait_until(predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not predicate():
            raise AssertionError("condition was not reached before timeout")


if __name__ == "__main__":
    unittest.main()
