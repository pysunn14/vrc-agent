using System.Globalization;

namespace VrcArdy.ProcessAudioProbe;

public sealed record ProcessAudioStreamOptions(uint ProcessId)
{
    public static ProcessAudioStreamOptions Parse(IReadOnlyList<string> args)
    {
        uint? processId = null;
        for (int index = 0; index < args.Count; index += 2)
        {
            if (index + 1 >= args.Count)
            {
                throw new ArgumentException($"Missing value for {args[index]}.");
            }
            if (args[index] != "--pid")
            {
                throw new ArgumentException($"Unknown argument: {args[index]}.");
            }
            if (processId is not null)
            {
                throw new ArgumentException("--pid may only be specified once.");
            }
            if (!uint.TryParse(
                    args[index + 1],
                    NumberStyles.None,
                    CultureInfo.InvariantCulture,
                    out uint parsedPid
                )
                || parsedPid == 0
                || parsedPid > int.MaxValue)
            {
                throw new ArgumentException("--pid must be a positive Windows process identifier.");
            }
            processId = parsedPid;
        }

        return new ProcessAudioStreamOptions(
            processId ?? throw new ArgumentException("--pid is required.")
        );
    }
}
