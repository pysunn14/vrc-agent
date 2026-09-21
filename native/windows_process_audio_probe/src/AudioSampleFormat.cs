using NAudio.Wave;

namespace VrcArdy.ProcessAudioProbe;

public static class AudioSampleFormat
{
    public static void RequireFloat32(WaveFormat waveFormat)
    {
        bool isStandardFloat = waveFormat.Encoding == WaveFormatEncoding.IeeeFloat;
        bool isExtensibleFloat = waveFormat is WaveFormatExtensible extensible
            && extensible.SubFormat == AudioMediaSubtypes.MEDIASUBTYPE_IEEE_FLOAT;
        if (waveFormat.BitsPerSample != 32 || (!isStandardFloat && !isExtensibleFloat))
        {
            throw new InvalidOperationException(
                $"Expected float32 PCM, received {waveFormat.Encoding} "
                + $"with {waveFormat.BitsPerSample} bits per sample."
            );
        }
    }
}
