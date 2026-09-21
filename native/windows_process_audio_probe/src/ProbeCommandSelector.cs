namespace VrcArdy.ProcessAudioProbe;

public enum ProbeCommand
{
    Capture,
    ListSessions,
    Stream,
}

public static class ProbeCommandSelector
{
    public static ProbeCommand Select(IReadOnlyList<string> args)
    {
        if (args.Count > 0 && args[0] == "list-sessions")
        {
            if (args.Count != 1)
            {
                throw new ArgumentException("list-sessions does not accept arguments.");
            }
            return ProbeCommand.ListSessions;
        }

        if (args.Count > 0 && args[0] == "stream")
        {
            return ProbeCommand.Stream;
        }

        return ProbeCommand.Capture;
    }
}
