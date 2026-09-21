using System.Globalization;

namespace VrcArdy.ProcessAudioProbe;

public enum AudioCaptureSource
{
    Process,
    System,
    SelfTest,
}

public sealed record ProbeOptions(
    AudioCaptureSource Source,
    uint? ProcessId,
    TimeSpan Duration,
    string Label
)
{
    private static readonly TimeSpan MaximumDuration = TimeSpan.FromSeconds(120);

    public static ProbeOptions Parse(IReadOnlyList<string> args)
    {
        uint? processId = null;
        AudioCaptureSource source = AudioCaptureSource.Process;
        TimeSpan duration = TimeSpan.FromSeconds(10);
        string label = "capture";

        for (int index = 0; index < args.Count; index += 2)
        {
            if (index + 1 >= args.Count)
            {
                throw new ArgumentException($"Missing value for {args[index]}.");
            }

            string value = args[index + 1];
            switch (args[index])
            {
                case "--source":
                    source = value switch
                    {
                        "process" => AudioCaptureSource.Process,
                        "system" => AudioCaptureSource.System,
                        "self-test" => AudioCaptureSource.SelfTest,
                        _ => throw new ArgumentException(
                            "--source must be process, system, or self-test."
                        ),
                    };
                    break;

                case "--pid":
                    if (!uint.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out uint parsedPid)
                        || parsedPid == 0
                        || parsedPid > int.MaxValue)
                    {
                        throw new ArgumentException("--pid must be a positive Windows process identifier.");
                    }
                    processId = parsedPid;
                    break;

                case "--duration-seconds":
                    if (!double.TryParse(
                            value,
                            NumberStyles.AllowDecimalPoint,
                            CultureInfo.InvariantCulture,
                            out double seconds
                        )
                        || !double.IsFinite(seconds)
                        || seconds <= 0)
                    {
                        throw new ArgumentException("--duration-seconds must be positive.");
                    }
                    duration = TimeSpan.FromSeconds(seconds);
                    if (duration > MaximumDuration)
                    {
                        throw new ArgumentException("--duration-seconds must not exceed 120.");
                    }
                    break;

                case "--label":
                    label = value.Trim();
                    if (label.Length is 0 or > 64)
                    {
                        throw new ArgumentException("--label must contain 1 to 64 characters.");
                    }
                    break;

                default:
                    throw new ArgumentException($"Unknown argument: {args[index]}.");
            }
        }

        if (source == AudioCaptureSource.Process && processId is null)
        {
            throw new ArgumentException("--pid is required when --source=process.");
        }
        if (source != AudioCaptureSource.Process && processId is not null)
        {
            throw new ArgumentException("--pid can only be used when --source=process.");
        }

        return new ProbeOptions(source, processId, duration, label);
    }
}
