using System.Diagnostics;
using System.Text.Json;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace VrcArdy.ProcessAudioProbe;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        bool binaryStandardOutput = args.Length > 0 && args[0] == "stream";
        try
        {
            ProbeCommand command = ProbeCommandSelector.Select(args);
            if (command == ProbeCommand.ListSessions)
            {
                WriteJson(new
                {
                    @event = "audio_sessions",
                    inventory = AudioSessionInventory.Read(),
                });
                return 0;
            }
            if (command == ProbeCommand.Stream)
            {
                ProcessAudioStreamOptions streamOptions = ProcessAudioStreamOptions.Parse(args[1..]);
                return await ProcessAudioStreamer.RunAsync(streamOptions);
            }

            ProbeOptions options = ProbeOptions.Parse(args);
            return await RunAsync(options);
        }
        catch (Exception exception)
        {
            WriteJson(new
            {
                @event = "error",
                error_type = exception.GetType().Name,
                message = exception.Message,
            }, useStandardError: binaryStandardOutput);
            return 2;
        }
    }

    private static async Task<int> RunAsync(ProbeOptions options)
    {
        return options.Source switch
        {
            AudioCaptureSource.Process => await RunProcessCaptureAsync(options),
            AudioCaptureSource.System => await RunSystemCaptureAsync(options),
            AudioCaptureSource.SelfTest => await RunSelfTestAsync(options),
            _ => throw new ArgumentOutOfRangeException(nameof(options)),
        };
    }

    private static async Task<int> RunProcessCaptureAsync(ProbeOptions options)
    {
        uint processId = options.ProcessId
            ?? throw new InvalidOperationException("Process capture requires a process identifier.");
        using Process target = Process.GetProcessById(checked((int)processId));
        if (target.HasExited)
        {
            throw new InvalidOperationException($"Process {processId} has already exited.");
        }

        await using WasapiRecorder recorder = await new WasapiRecorderBuilder()
            .WithProcessLoopback(
                processId,
                ProcessLoopbackMode.IncludeTargetProcessTree
            )
            .BuildAsync();

        AudioSampleFormat.RequireFloat32(recorder.WaveFormat);

        var meter = new AudioMeter();
        var stopped = new TaskCompletionSource<Exception?>(
            TaskCreationOptions.RunContinuationsAsynchronously
        );
        recorder.DataAvailable += (buffer, flags, _, _) =>
        {
            meter.Observe(buffer, (flags & AudioClientBufferFlags.Silent) != 0);
        };
        recorder.RecordingStopped += (_, eventArgs) => stopped.TrySetResult(eventArgs.Exception);

        WriteJson(new
        {
            @event = "started",
            label = options.Label,
            source = "process",
            pid = processId,
            duration_seconds = options.Duration.TotalSeconds,
            format = new
            {
                sample_rate = recorder.WaveFormat.SampleRate,
                channels = recorder.WaveFormat.Channels,
                bits_per_sample = recorder.WaveFormat.BitsPerSample,
                encoding = recorder.WaveFormat.Encoding.ToString(),
            },
        });

        var stopwatch = Stopwatch.StartNew();
        recorder.StartRecording();
        while (stopwatch.Elapsed < options.Duration && !stopped.Task.IsCompleted)
        {
            TimeSpan remaining = options.Duration - stopwatch.Elapsed;
            await Task.Delay(remaining < TimeSpan.FromSeconds(1) ? remaining : TimeSpan.FromSeconds(1));
            if (stopwatch.Elapsed >= options.Duration || stopped.Task.IsCompleted)
            {
                break;
            }
            WriteJson(new
            {
                @event = "heartbeat",
                label = options.Label,
                elapsed_seconds = Math.Round(stopwatch.Elapsed.TotalSeconds, 3),
                process_alive = !target.HasExited,
                audio = meter.Snapshot(),
            });
        }

        if (!stopped.Task.IsCompleted)
        {
            recorder.StopRecording();
        }
        Exception? captureError = await stopped.Task.WaitAsync(TimeSpan.FromSeconds(5));
        if (captureError is not null)
        {
            throw new InvalidOperationException("WASAPI capture stopped with an error.", captureError);
        }

        AudioMeterSnapshot result = meter.Snapshot();
        WriteJson(new
        {
            @event = "result",
            label = options.Label,
            source = "process",
            pid = processId,
            elapsed_seconds = Math.Round(stopwatch.Elapsed.TotalSeconds, 3),
            process_alive = !target.HasExited,
            audio = result,
        });
        return 0;
    }

    private static async Task<int> RunSystemCaptureAsync(ProbeOptions options)
    {
        await using WasapiRecorder recorder = new WasapiRecorderBuilder()
            .WithLoopbackCapture()
            .Build();
        AudioSampleFormat.RequireFloat32(recorder.WaveFormat);

        var meter = new AudioMeter();
        var stopped = new TaskCompletionSource<Exception?>(
            TaskCreationOptions.RunContinuationsAsynchronously
        );
        recorder.DataAvailable += (buffer, flags, _, _) =>
        {
            meter.Observe(buffer, (flags & AudioClientBufferFlags.Silent) != 0);
        };
        recorder.RecordingStopped += (_, eventArgs) => stopped.TrySetResult(eventArgs.Exception);

        WriteJson(new
        {
            @event = "started",
            label = options.Label,
            source = "system",
            duration_seconds = options.Duration.TotalSeconds,
            format = new
            {
                sample_rate = recorder.WaveFormat.SampleRate,
                channels = recorder.WaveFormat.Channels,
                bits_per_sample = recorder.WaveFormat.BitsPerSample,
                encoding = recorder.WaveFormat.Encoding.ToString(),
            },
        });

        var stopwatch = Stopwatch.StartNew();
        recorder.StartRecording();
        while (stopwatch.Elapsed < options.Duration && !stopped.Task.IsCompleted)
        {
            TimeSpan remaining = options.Duration - stopwatch.Elapsed;
            await Task.Delay(remaining < TimeSpan.FromSeconds(1) ? remaining : TimeSpan.FromSeconds(1));
            if (stopwatch.Elapsed >= options.Duration || stopped.Task.IsCompleted)
            {
                break;
            }
            WriteJson(new
            {
                @event = "heartbeat",
                label = options.Label,
                elapsed_seconds = Math.Round(stopwatch.Elapsed.TotalSeconds, 3),
                audio = meter.Snapshot(),
            });
        }

        if (!stopped.Task.IsCompleted)
        {
            recorder.StopRecording();
        }
        Exception? captureError = await stopped.Task.WaitAsync(TimeSpan.FromSeconds(5));
        if (captureError is not null)
        {
            throw new InvalidOperationException("WASAPI capture stopped with an error.", captureError);
        }

        WriteJson(new
        {
            @event = "result",
            label = options.Label,
            source = "system",
            elapsed_seconds = Math.Round(stopwatch.Elapsed.TotalSeconds, 3),
            audio = meter.Snapshot(),
        });
        return 0;
    }

    private static async Task<int> RunSelfTestAsync(ProbeOptions options)
    {
        uint processId = checked((uint)Environment.ProcessId);
        await using WasapiRecorder recorder = await new WasapiRecorderBuilder()
            .WithProcessLoopback(processId, ProcessLoopbackMode.IncludeTargetProcessTree)
            .BuildAsync();
        AudioSampleFormat.RequireFloat32(recorder.WaveFormat);

        await using WasapiPlayer player = new WasapiPlayerBuilder().Build();
        var tone = new SignalGenerator(44_100, 2)
        {
            Frequency = 440,
            Gain = 0.05,
            Type = SignalGeneratorType.Sin,
        };
        player.Init(tone.ToWaveProvider());

        var meter = new AudioMeter();
        var stopped = new TaskCompletionSource<Exception?>(
            TaskCreationOptions.RunContinuationsAsynchronously
        );
        recorder.DataAvailable += (buffer, flags, _, _) =>
        {
            meter.Observe(buffer, (flags & AudioClientBufferFlags.Silent) != 0);
        };
        recorder.RecordingStopped += (_, eventArgs) => stopped.TrySetResult(eventArgs.Exception);

        WriteJson(new
        {
            @event = "started",
            label = options.Label,
            source = "self-test",
            pid = processId,
            duration_seconds = options.Duration.TotalSeconds,
            format = new
            {
                sample_rate = recorder.WaveFormat.SampleRate,
                channels = recorder.WaveFormat.Channels,
                bits_per_sample = recorder.WaveFormat.BitsPerSample,
                encoding = recorder.WaveFormat.Encoding.ToString(),
            },
        });

        var stopwatch = Stopwatch.StartNew();
        recorder.StartRecording();
        player.Play();
        while (stopwatch.Elapsed < options.Duration && !stopped.Task.IsCompleted)
        {
            TimeSpan remaining = options.Duration - stopwatch.Elapsed;
            await Task.Delay(remaining < TimeSpan.FromSeconds(1) ? remaining : TimeSpan.FromSeconds(1));
            if (stopwatch.Elapsed >= options.Duration || stopped.Task.IsCompleted)
            {
                break;
            }
            WriteJson(new
            {
                @event = "heartbeat",
                label = options.Label,
                elapsed_seconds = Math.Round(stopwatch.Elapsed.TotalSeconds, 3),
                audio = meter.Snapshot(),
            });
        }

        player.Stop();
        if (!stopped.Task.IsCompleted)
        {
            recorder.StopRecording();
        }
        Exception? captureError = await stopped.Task.WaitAsync(TimeSpan.FromSeconds(5));
        if (captureError is not null)
        {
            throw new InvalidOperationException("WASAPI capture stopped with an error.", captureError);
        }

        WriteJson(new
        {
            @event = "result",
            label = options.Label,
            source = "self-test",
            pid = processId,
            elapsed_seconds = Math.Round(stopwatch.Elapsed.TotalSeconds, 3),
            audio = meter.Snapshot(),
        });
        return 0;
    }

    private static void WriteJson(object value, bool useStandardError = false)
    {
        TextWriter writer = useStandardError ? Console.Error : Console.Out;
        writer.WriteLine(JsonSerializer.Serialize(value));
        writer.Flush();
    }
}
