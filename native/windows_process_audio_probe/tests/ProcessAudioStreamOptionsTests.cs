namespace VrcArdy.ProcessAudioProbe.Tests;

public sealed class ProcessAudioStreamOptionsTests
{
    [Fact]
    public void Parse_RequiresAndReturnsTargetPid()
    {
        ProcessAudioStreamOptions options = ProcessAudioStreamOptions.Parse(["--pid", "2860"]);

        Assert.Equal(2860u, options.ProcessId);
    }

    [Theory]
    [InlineData()]
    [InlineData("--pid", "0")]
    [InlineData("--pid", "nope")]
    [InlineData("--pid", "42", "--unknown", "value")]
    public void Parse_RejectsInvalidArguments(params string[] args)
    {
        Assert.Throws<ArgumentException>(() => ProcessAudioStreamOptions.Parse(args));
    }
}
