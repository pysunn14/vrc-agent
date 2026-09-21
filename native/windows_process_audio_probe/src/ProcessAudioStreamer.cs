using System.Diagnostics;
using System.Text.Json;
using System.Threading.Channels;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace VrcArdy.ProcessAudioProbe;

public static class ProcessAudioStreamer
{
    private const int QueueCapacity = 128;

    public static async Task<int> RunAsync(ProcessAudioStreamOptions options)
    {
        using Process target = Process.GetProcessById(checked((int)options.ProcessId));
        if (target.HasExited)
        {
            throw new InvalidOperationException(
                $"Process {options.ProcessId} has already exited."
            );
        }

        await using WasapiRecorder recorder = await new WasapiRecorderBuilder()
            .WithProcessLoopback(
                options.ProcessId,
                ProcessLoopbackMode.IncludeTargetProcessTree
            )
            .BuildAsync();
        AudioSampleFormat.RequireFloat32(recorder.WaveFormat);

        var queue = Channel.CreateBounded<byte[]>(
            new BoundedChannelOptions(QueueCapacity)
            {
                SingleReader = true,
                SingleWriter = true,
                FullMode = BoundedChannelFullMode.Wait,
            }
        );
        var meter = new AudioMeter();
        var stopped = new TaskCompletionSource<Exception?>(
            TaskCreationOptions.RunContinuationsAsynchronously
        );
        var failed = new TaskCompletionSource<Exception>(
            TaskCreationOptions.RunContinuationsAsynchronously
        );
        long packetsEnqueued = 0;
        long bytesStreamed = 0;

        recorder.DataAvailable += (buffer, flags, _, _) =>
        {
            bool silent = (flags & AudioClientBufferFlags.Silent) != 0;
            meter.Observe(buffer, silent);
            byte[] payload = silent ? new byte[buffer.Length] : buffer.ToArray();
            if (!queue.Writer.TryWrite(payload))
            {
                failed.TrySetResult(
                    new InvalidOperationException(
                        $"Audio stream queue exceeded its {QueueCapacity}-packet capacity."
                    )
                );
                return;
            }
            Interlocked.Increment(ref packetsEnqueued);
        };
        recorder.RecordingStopped += (_, eventArgs) => stopped.TrySetResult(eventArgs.Exception);

        Stream stdout = Console.OpenStandardOutput();
        Task writer = Task.Run(async () =>
        {
            try
            {
                await foreach (byte[] payload in queue.Reader.ReadAllAsync())
                {
                    await stdout.WriteAsync(payload);
                    await stdout.FlushAsync();
                    Interlocked.Add(ref bytesStreamed, payload.Length);
                }
            }
            catch (Exception exception)
            {
                // A broken stdout pipe means the owning companion is gone. Stop
                // capture instead of leaving an orphan process in the session.
                failed.TrySetResult(
                    new IOException("Audio stream output pipe failed.", exception)
                );
            }
        });

        WriteStatus(new
        {
            @event = "started",
            source = "process",
            pid = options.ProcessId,
            format = new
            {
                sample_rate = recorder.WaveFormat.SampleRate,
                channels = recorder.WaveFormat.Channels,
                bits_per_sample = recorder.WaveFormat.BitsPerSample,
                sample_format = "float32",
            },
        });

        recorder.StartRecording();
        try
        {
            while (!stopped.Task.IsCompleted && !failed.Task.IsCompleted)
            {
                Task delay = Task.Delay(TimeSpan.FromSeconds(1));
                Task completed = await Task.WhenAny(delay, stopped.Task, failed.Task);
                if (completed != delay)
                {
                    break;
                }
                if (target.HasExited)
                {
                    failed.TrySetResult(
                        new InvalidOperationException(
                            $"Target process {options.ProcessId} exited during capture."
                        )
                    );
                    break;
                }
                WriteStatus(new
                {
                    @event = "heartbeat",
                    pid = options.ProcessId,
                    process_alive = true,
                    packets = Interlocked.Read(ref packetsEnqueued),
                    bytes_streamed = Interlocked.Read(ref bytesStreamed),
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
                throw new InvalidOperationException(
                    "WASAPI process capture stopped with an error.",
                    captureError
                );
            }
            if (failed.Task.IsCompleted)
            {
                throw await failed.Task;
            }
        }
        finally
        {
            queue.Writer.TryComplete();
            await writer;
        }

        return 0;
    }

    private static void WriteStatus(object value)
    {
        Console.Error.WriteLine(JsonSerializer.Serialize(value));
        Console.Error.Flush();
    }
}
