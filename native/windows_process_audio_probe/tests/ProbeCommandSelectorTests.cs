namespace VrcArdy.ProcessAudioProbe.Tests;

public sealed class ProbeCommandSelectorTests
{
    [Fact]
    public void Select_RecognizesSessionInventoryCommand()
    {
        Assert.Equal(
            ProbeCommand.ListSessions,
            ProbeCommandSelector.Select(["list-sessions"])
        );
    }

    [Fact]
    public void Select_TreatsCaptureArgumentsAsCaptureCommand()
    {
        Assert.Equal(
            ProbeCommand.Capture,
            ProbeCommandSelector.Select(["--pid", "42"])
        );
    }

    [Fact]
    public void Select_RecognizesContinuousStreamCommand()
    {
        Assert.Equal(
            ProbeCommand.Stream,
            ProbeCommandSelector.Select(["stream", "--pid", "42"])
        );
    }

    [Fact]
    public void Select_RejectsArgumentsAfterSessionInventoryCommand()
    {
        Assert.Throws<ArgumentException>(
            () => ProbeCommandSelector.Select(["list-sessions", "extra"])
        );
    }
}
