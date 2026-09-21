using System.Diagnostics;
using NAudio.CoreAudioApi;

namespace VrcArdy.ProcessAudioProbe;

public sealed record RenderDefaults(string Console, string Multimedia, string Communications);

public sealed record AudioSessionSnapshot(
    string DeviceId,
    string DeviceName,
    uint ProcessId,
    string? ProcessName,
    string State,
    float Peak,
    float Volume,
    bool Muted,
    bool IsSystemSounds
);

public sealed record AudioSessionInventorySnapshot(
    RenderDefaults Defaults,
    IReadOnlyList<AudioSessionSnapshot> Sessions
);

public static class AudioSessionInventory
{
    public static AudioSessionInventorySnapshot Read()
    {
        using var enumerator = new MMDeviceEnumerator();
        var defaults = new RenderDefaults(
            enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console).ID,
            enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia).ID,
            enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Communications).ID
        );
        var snapshots = new List<AudioSessionSnapshot>();

        using MMDeviceCollection devices = enumerator.EnumerateAudioEndPoints(
            DataFlow.Render,
            DeviceState.Active
        );
        foreach (MMDevice device in devices)
        {
            using (device)
            {
                AudioSessionManager manager = device.AudioSessionManager;
                manager.RefreshSessions();
                for (int index = 0; index < manager.Sessions.Count; index++)
                {
                    using AudioSessionControl session = manager.Sessions[index];
                    uint processId = session.GetProcessID;
                    snapshots.Add(new AudioSessionSnapshot(
                        device.ID,
                        device.FriendlyName,
                        processId,
                        TryGetProcessName(processId),
                        session.State.ToString(),
                        session.AudioMeterInformation.MasterPeakValue,
                        session.SimpleAudioVolume.Volume,
                        session.SimpleAudioVolume.Mute,
                        session.IsSystemSoundsSession
                    ));
                }
            }
        }

        return new AudioSessionInventorySnapshot(defaults, snapshots);
    }

    private static string? TryGetProcessName(uint processId)
    {
        if (processId == 0)
        {
            return null;
        }

        try
        {
            using Process process = Process.GetProcessById(checked((int)processId));
            return process.ProcessName;
        }
        catch (ArgumentException)
        {
            return null;
        }
    }
}
