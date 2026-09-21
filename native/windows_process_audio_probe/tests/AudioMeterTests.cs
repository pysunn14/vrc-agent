namespace VrcArdy.ProcessAudioProbe.Tests;

public sealed class AudioMeterTests
{
    [Fact]
    public void Observe_AccumulatesPeakRmsAndPacketCounts()
    {
        var meter = new AudioMeter();

        meter.Observe(FloatBytes(0.5f, -0.25f, 0.0f, 1.0f), silent: false);

        AudioMeterSnapshot snapshot = meter.Snapshot();
        Assert.Equal(1, snapshot.Packets);
        Assert.Equal(1, snapshot.NonSilentPackets);
        Assert.Equal(4, snapshot.Samples);
        Assert.Equal(1.0, snapshot.Peak, precision: 6);
        Assert.Equal(Math.Sqrt(1.3125 / 4.0), snapshot.Rms, precision: 6);
        Assert.Equal("signal", snapshot.SignalState);
    }

    [Fact]
    public void Observe_TreatsWasapiSilentPacketAsSilenceWithoutReadingPayload()
    {
        var meter = new AudioMeter();

        meter.Observe(FloatBytes(1.0f, -1.0f), silent: true);

        AudioMeterSnapshot snapshot = meter.Snapshot();
        Assert.Equal(1, snapshot.Packets);
        Assert.Equal(1, snapshot.SilentPackets);
        Assert.Equal(0, snapshot.NonSilentPackets);
        Assert.Equal(2, snapshot.Samples);
        Assert.Equal(0.0, snapshot.Peak);
        Assert.Equal(0.0, snapshot.Rms);
        Assert.Equal("silence", snapshot.SignalState);
    }

    [Fact]
    public void Snapshot_ReportsNoPacketsBeforeCaptureProducesData()
    {
        var snapshot = new AudioMeter().Snapshot();

        Assert.Equal("no_packets", snapshot.SignalState);
    }

    [Fact]
    public void Observe_RejectsMalformedFloat32Payload()
    {
        var meter = new AudioMeter();

        Assert.Throws<ArgumentException>(() => meter.Observe(new byte[3], silent: false));
    }

    [Fact]
    public void Observe_RejectsNonFiniteSamples()
    {
        var meter = new AudioMeter();

        Assert.Throws<ArgumentException>(() => meter.Observe(FloatBytes(float.NaN), silent: false));
    }

    private static byte[] FloatBytes(params float[] samples)
    {
        var bytes = new byte[samples.Length * sizeof(float)];
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
