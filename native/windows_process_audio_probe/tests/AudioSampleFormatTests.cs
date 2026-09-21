using NAudio.Wave;

namespace VrcArdy.ProcessAudioProbe.Tests;

public sealed class AudioSampleFormatTests
{
    [Fact]
    public void RequireFloat32_AcceptsStandardIeeeFloat()
    {
        AudioSampleFormat.RequireFloat32(WaveFormat.CreateIeeeFloatWaveFormat(48_000, 2));
    }

    [Fact]
    public void RequireFloat32_AcceptsExtensibleIeeeFloat()
    {
        var format = new WaveFormatExtensible(
            48_000,
            32,
            2,
            useIeeeFloat: true,
            validBitsPerSample: 32,
            channelMask: 3
        );

        AudioSampleFormat.RequireFloat32(format);
    }

    [Fact]
    public void RequireFloat32_RejectsIntegerPcm()
    {
        Assert.Throws<InvalidOperationException>(
            () => AudioSampleFormat.RequireFloat32(new WaveFormat(48_000, 16, 2))
        );
    }
}
