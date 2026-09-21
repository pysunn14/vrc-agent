using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDKBase;
using VRC.SDK3.Data;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class CalibrationProbe : UdonSharpBehaviour
{
    public Text statusText;
    public float sampleInterval = 0.5f;
    public float captureSeconds = 60f;
    private VRCPlayerApi local;
    private bool capturing;
    private float deadline;
    private float nextSample;
    private int sequence;
    private int session;
    private readonly HumanBodyBones[] bones = {
        HumanBodyBones.Head, HumanBodyBones.Neck, HumanBodyBones.Hips,
        HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,
        HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,
        HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot
    };
    private readonly string[] boneNames = {
        "head", "neck", "hips", "leftUpperArm", "leftLowerArm", "leftHand",
        "rightUpperArm", "rightLowerArm", "rightHand", "leftFoot", "rightFoot"
    };

    public override void OnPlayerJoined(VRCPlayerApi player)
    {
        if (!player.isLocal) return;
        local = player;
        BeginCapture();
    }

    public override void Interact()
    {
        if (capturing) StopCapture(); else BeginCapture();
    }

    public void BeginCapture()
    {
        local = Networking.LocalPlayer;
        if (!Utilities.IsValid(local)) return;
        session++;
        sequence = 0;
        capturing = true;
        deadline = Time.realtimeSinceStartup + captureSeconds;
        nextSample = 0;
    }

    public void StopCapture()
    {
        capturing = false;
        if (statusText != null) statusText.text = "VRC AGENT / CALIBRATION\n\nCapture stopped.\nUse the green button to record again.\n\nNo avatar poses are changed by this world.";
        Debug.Log("VRC_AGENT_CALIBRATION_END session=" + session + " samples=" + sequence);
    }

    public override void PostLateUpdate()
    {
        if (!capturing || !Utilities.IsValid(local)) return;
        float now = Time.realtimeSinceStartup;
        if (now >= deadline) { StopCapture(); return; }
        if (now < nextSample) return;
        nextSample = now + Mathf.Max(0.1f, sampleInterval);
        sequence++;
        DataDictionary record = new DataDictionary();
        record["schema"] = 1;
        record["session"] = session;
        record["sequence"] = sequence;
        record["time_seconds"] = now;
        record["frame"] = Time.frameCount;
        record["is_vr"] = local.IsUserInVR();
        record["eye_height_m"] = local.GetAvatarEyeHeightAsMeters();
        record["player"] = Pose(local.GetPosition(), local.GetRotation());
        DataDictionary tracking = new DataDictionary();
        tracking["head"] = Tracking(VRCPlayerApi.TrackingDataType.Head);
        tracking["leftHand"] = Tracking(VRCPlayerApi.TrackingDataType.LeftHand);
        tracking["rightHand"] = Tracking(VRCPlayerApi.TrackingDataType.RightHand);
        tracking["origin"] = Tracking(VRCPlayerApi.TrackingDataType.Origin);
        tracking["avatarRoot"] = Tracking(VRCPlayerApi.TrackingDataType.AvatarRoot);
        record["tracking"] = tracking;
        DataDictionary skeleton = new DataDictionary();
        for (int i = 0; i < bones.Length; i++)
            skeleton[boneNames[i]] = Pose(local.GetBonePosition(bones[i]), local.GetBoneRotation(bones[i]));
        record["bones"] = skeleton;
        DataToken json;
        if (VRCJson.TrySerializeToJson(record, JsonExportType.Minify, out json))
            Debug.Log("VRC_AGENT_CALIBRATION " + json.String);
        else { Debug.LogError("VRC_AGENT_CALIBRATION_ERROR serialization failed"); StopCapture(); return; }
        if (statusText != null)
            statusText.text = "VRC AGENT / CALIBRATION\n\nRECORDING   #" + sequence
                + "\nRemaining: " + Mathf.CeilToInt(deadline - now) + " s"
                + "\nVR input: " + local.IsUserInVR()
                + "\nEye height: " + local.GetAvatarEyeHeightAsMeters().ToString("F3") + " m"
                + "\n\nTracking head: " + local.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position.ToString("F3")
                + "\nLeft wrist bone: " + local.GetBonePosition(HumanBodyBones.LeftHand).ToString("F3")
                + "\nRight wrist bone: " + local.GetBonePosition(HumanBodyBones.RightHand).ToString("F3")
                + "\n\nGreen button: stop / restart\nData is recorded to the local VRChat log.";
    }

    private DataDictionary Tracking(VRCPlayerApi.TrackingDataType type)
    {
        VRCPlayerApi.TrackingData data = local.GetTrackingData(type);
        return Pose(data.position, data.rotation);
    }

    private DataDictionary Pose(Vector3 p, Quaternion q)
    {
        DataDictionary pose = new DataDictionary();
        DataList position = new DataList(); position.Add(p.x); position.Add(p.y); position.Add(p.z);
        DataList rotation = new DataList(); rotation.Add(q.x); rotation.Add(q.y); rotation.Add(q.z); rotation.Add(q.w);
        pose["position"] = position;
        pose["quaternion_xyzw"] = rotation;
        return pose;
    }
}
