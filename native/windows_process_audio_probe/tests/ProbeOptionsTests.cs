namespace VrcArdy.ProcessAudioProbe.Tests;

public sealed class ProbeOptionsTests
{
    [Fact]
    public void Parse_RequiresOnlyPidAndUsesSafeDefaults()
    {
        ProbeOptions options = ProbeOptions.Parse(["--pid", "16544"]);

        Assert.Equal(AudioCaptureSource.Process, options.Source);
        Assert.Equal(16544u, options.ProcessId);
        Assert.Equal(TimeSpan.FromSeconds(10), options.Duration);
        Assert.Equal("capture", options.Label);
    }

    [Fact]
    public void Parse_AcceptsSystemLoopbackWithoutPid()
    {
        ProbeOptions options = ProbeOptions.Parse(
            ["--source", "system", "--duration-seconds", "3", "--label", "system-output"]
        );

        Assert.Equal(AudioCaptureSource.System, options.Source);
        Assert.Null(options.ProcessId);
        Assert.Equal(TimeSpan.FromSeconds(3), options.Duration);
        Assert.Equal("system-output", options.Label);
    }

    [Fact]
    public void Parse_AcceptsSelfTestWithoutPid()
    {
        ProbeOptions options = ProbeOptions.Parse(
            ["--source", "self-test", "--duration-seconds", "2", "--label", "known-tone"]
        );

        Assert.Equal(AudioCaptureSource.SelfTest, options.Source);
        Assert.Null(options.ProcessId);
    }

    [Fact]
    public void Parse_RejectsPidForSystemLoopback()
    {
        Assert.Throws<ArgumentException>(
            () => ProbeOptions.Parse(["--source", "system", "--pid", "42"])
        );
    }

    [Fact]
    public void Parse_RejectsPidForSelfTest()
    {
        Assert.Throws<ArgumentException>(
            () => ProbeOptions.Parse(["--source", "self-test", "--pid", "42"])
        );
    }

    [Fact]
    public void Parse_AcceptsDurationAndLabel()
    {
        ProbeOptions options = ProbeOptions.Parse(
            ["--pid", "42", "--duration-seconds", "12.5", "--label", "v-on"]
        );

        Assert.Equal(TimeSpan.FromSeconds(12.5), options.Duration);
        Assert.Equal("v-on", options.Label);
    }

    [Theory]
    [InlineData()]
    [InlineData("--pid", "0")]
    [InlineData("--pid", "nope")]
    [InlineData("--pid", "42", "--duration-seconds", "0")]
    [InlineData("--pid", "42", "--duration-seconds", "121")]
    [InlineData("--source", "process")]
    [InlineData("--source", "unknown", "--pid", "42")]
    [InlineData("--pid", "42", "--unknown", "value")]
    public void Parse_RejectsInvalidArguments(params string[] args)
    {
        Assert.Throws<ArgumentException>(() => ProbeOptions.Parse(args));
    }
}
