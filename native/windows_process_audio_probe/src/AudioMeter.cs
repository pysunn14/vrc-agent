using System.Runtime.InteropServices;

namespace VrcArdy.ProcessAudioProbe;

public sealed record AudioMeterSnapshot(
    long Packets,
    long SilentPackets,
    long NonSilentPackets,
    long Samples,
    double Peak,
    double Rms,
    string SignalState
);

public sealed class AudioMeter
{
    private const double SignalThreshold = 0.00001;
    private readonly object gate = new();
    private long packets;
    private long silentPackets;
    private long nonSilentPackets;
    private long samples;
    private double peak;
    private double sumOfSquares;

    public void Observe(ReadOnlySpan<byte> payload, bool silent)
    {
        if (payload.Length % sizeof(float) != 0)
        {
            throw new ArgumentException("The PCM payload must contain complete float32 samples.", nameof(payload));
        }

        int sampleCount = payload.Length / sizeof(float);
        if (silent)
        {
            lock (gate)
            {
                packets++;
                silentPackets++;
                samples += sampleCount;
            }
            return;
        }

        ReadOnlySpan<float> values = MemoryMarshal.Cast<byte, float>(payload);
        double packetPeak = 0;
        double packetSumOfSquares = 0;
        foreach (float value in values)
        {
            if (!float.IsFinite(value))
            {
                throw new ArgumentException("The PCM payload contains a non-finite sample.", nameof(payload));
            }
            double absolute = Math.Abs(value);
            packetPeak = Math.Max(packetPeak, absolute);
            packetSumOfSquares += value * (double)value;
        }

        lock (gate)
        {
            packets++;
            nonSilentPackets++;
            samples += sampleCount;
            peak = Math.Max(peak, packetPeak);
            sumOfSquares += packetSumOfSquares;
        }
    }

    public AudioMeterSnapshot Snapshot()
    {
        lock (gate)
        {
            double rms = samples == 0 ? 0 : Math.Sqrt(sumOfSquares / samples);
            string signalState = packets == 0
                ? "no_packets"
                : peak >= SignalThreshold
                    ? "signal"
                    : "silence";
            return new AudioMeterSnapshot(
                packets,
                silentPackets,
                nonSilentPackets,
                samples,
                peak,
                rms,
                signalState
            );
        }
    }
}
