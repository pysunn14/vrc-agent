using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace VrcArdy.Editor
{
    /// <summary>
    /// Exports a Humanoid avatar's current editor pose in a stable, head-relative
    /// coordinate frame. Runtime retargeting must apply tracker mount offsets;
    /// exported bone rotations are source calibration data, not VMT commands.
    /// </summary>
    internal static class AvatarRigProfileExporter
    {
        private const string MenuPath = "Tools/VRC Ardy/Export Selected Avatar Rig Profile";
        private const string OutputDirectory = "Assets/VrcArdy/Generated";

        private static readonly BoneSpec[] BoneSpecs =
        {
            new BoneSpec("hips", HumanBodyBones.Hips, true),
            new BoneSpec("spine", HumanBodyBones.Spine, false),
            new BoneSpec("chest", HumanBodyBones.Chest, false),
            new BoneSpec("upperChest", HumanBodyBones.UpperChest, false),
            new BoneSpec("neck", HumanBodyBones.Neck, false),
            new BoneSpec("head", HumanBodyBones.Head, true),
            new BoneSpec("leftShoulder", HumanBodyBones.LeftShoulder, false),
            new BoneSpec("leftUpperArm", HumanBodyBones.LeftUpperArm, true),
            new BoneSpec("leftLowerArm", HumanBodyBones.LeftLowerArm, true),
            new BoneSpec("leftHand", HumanBodyBones.LeftHand, true),
            // Palm landmarks are optional for general body-only Humanoids.
            // Hand calibration must check their presence before deriving axes;
            // a missing finger must never be replaced with a guessed direction.
            new BoneSpec("leftThumbProximal", HumanBodyBones.LeftThumbProximal, false),
            new BoneSpec("leftIndexProximal", HumanBodyBones.LeftIndexProximal, false),
            new BoneSpec("leftMiddleProximal", HumanBodyBones.LeftMiddleProximal, false),
            new BoneSpec("leftLittleProximal", HumanBodyBones.LeftLittleProximal, false),
            new BoneSpec("rightShoulder", HumanBodyBones.RightShoulder, false),
            new BoneSpec("rightUpperArm", HumanBodyBones.RightUpperArm, true),
            new BoneSpec("rightLowerArm", HumanBodyBones.RightLowerArm, true),
            new BoneSpec("rightHand", HumanBodyBones.RightHand, true),
            new BoneSpec("rightThumbProximal", HumanBodyBones.RightThumbProximal, false),
            new BoneSpec("rightIndexProximal", HumanBodyBones.RightIndexProximal, false),
            new BoneSpec("rightMiddleProximal", HumanBodyBones.RightMiddleProximal, false),
            new BoneSpec("rightLittleProximal", HumanBodyBones.RightLittleProximal, false),
            new BoneSpec("leftUpperLeg", HumanBodyBones.LeftUpperLeg, true),
            new BoneSpec("leftLowerLeg", HumanBodyBones.LeftLowerLeg, true),
            new BoneSpec("leftFoot", HumanBodyBones.LeftFoot, true),
            new BoneSpec("leftToes", HumanBodyBones.LeftToes, false),
            new BoneSpec("rightUpperLeg", HumanBodyBones.RightUpperLeg, true),
            new BoneSpec("rightLowerLeg", HumanBodyBones.RightLowerLeg, true),
            new BoneSpec("rightFoot", HumanBodyBones.RightFoot, true),
            new BoneSpec("rightToes", HumanBodyBones.RightToes, false),
        };

        [MenuItem(MenuPath, true)]
        private static bool ValidateExport()
        {
            return ResolveSelectedAnimator() != null && !EditorApplication.isPlayingOrWillChangePlaymode;
        }

        [MenuItem(MenuPath)]
        private static void Export()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                ShowError("Stop Play Mode before exporting a reference pose.");
                return;
            }

            Animator animator = ResolveSelectedAnimator();
            if (animator == null)
            {
                ShowError("Select a GameObject with a Humanoid Animator, or one of its children.");
                return;
            }

            try
            {
                AvatarRigProfileDocument document = BuildDocument(animator);
                string fileName = SanitizeFileName(animator.gameObject.name) + ".avatar-rig.json";
                string assetPath = OutputDirectory + "/" + fileName;
                string absolutePath = Path.Combine(
                    Directory.GetParent(Application.dataPath).FullName,
                    assetPath.Replace('/', Path.DirectorySeparatorChar));

                if (File.Exists(absolutePath) && !EditorUtility.DisplayDialog(
                        "Replace avatar rig profile?",
                        assetPath + " already exists.",
                        "Replace",
                        "Cancel"))
                {
                    return;
                }

                Directory.CreateDirectory(Path.GetDirectoryName(absolutePath));
                File.WriteAllText(
                    absolutePath,
                    JsonUtility.ToJson(document, true) + Environment.NewLine,
                    new UTF8Encoding(false));
                AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);

                UnityEngine.Object exportedAsset = AssetDatabase.LoadMainAssetAtPath(assetPath);
                Selection.activeObject = exportedAsset;
                EditorGUIUtility.PingObject(exportedAsset);
                Debug.Log(
                    "[VRC Ardy] Exported avatar rig profile: " + assetPath
                    + " (floor=" + document.floorReference
                    + ", height=" + document.referenceHeightMeters.ToString("F4") + " m)",
                    exportedAsset);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                ShowError(exception.Message);
            }
        }

        private static AvatarRigProfileDocument BuildDocument(Animator animator)
        {
            if (animator.avatar == null || !animator.avatar.isValid || !animator.avatar.isHuman)
            {
                throw new InvalidOperationException(
                    "The selected Animator must have a valid Humanoid avatar.");
            }

            Transform root = animator.transform;
            Transform head = RequireBone(animator, HumanBodyBones.Head, "head");
            Quaternion inverseRootRotation = Quaternion.Inverse(root.rotation);
            var exportedBones = new List<BonePoseDocument>(BoneSpecs.Length);
            var positions = new Dictionary<string, Vector3>(BoneSpecs.Length);

            foreach (BoneSpec spec in BoneSpecs)
            {
                Transform bone = animator.GetBoneTransform(spec.humanBone);
                if (bone == null)
                {
                    if (spec.required)
                    {
                        throw new InvalidOperationException(
                            "Required Humanoid bone is missing: " + spec.profileName);
                    }
                    continue;
                }

                // World distances preserve the avatar's imported scale. Rotating the
                // delta into root axes removes scene placement without losing meters.
                Vector3 position = inverseRootRotation * (bone.position - head.position);
                Quaternion rotation = NormalizeCanonical(inverseRootRotation * bone.rotation);
                positions.Add(spec.profileName, position);
                exportedBones.Add(new BonePoseDocument
                {
                    name = spec.profileName,
                    positionMeters = new[] { position.x, position.y, position.z },
                    rotationXyzw = new[] { rotation.x, rotation.y, rotation.z, rotation.w },
                });
            }

            bool hasBothToes = positions.ContainsKey("leftToes") && positions.ContainsKey("rightToes");
            string floorReference = hasBothToes ? "toes" : "feet";
            string leftFloorBone = hasBothToes ? "leftToes" : "leftFoot";
            string rightFloorBone = hasBothToes ? "rightToes" : "rightFoot";
            float referenceHeight = -(positions[leftFloorBone].y + positions[rightFloorBone].y) * 0.5f;
            if (!IsFinitePositive(referenceHeight))
            {
                throw new InvalidOperationException(
                    "The current pose does not produce a positive head-to-floor height. "
                    + "Reset the avatar to its neutral editor pose and try again.");
            }

            if (!hasBothToes)
            {
                Debug.LogWarning(
                    "[VRC Ardy] The avatar does not provide both Humanoid toe bones. "
                    + "The exported profile explicitly uses foot bones as its floor reference.",
                    animator);
            }

            return new AvatarRigProfileDocument
            {
                schemaVersion = 1,
                profileName = animator.gameObject.name,
                units = "meters",
                coordinateSpace = "avatar_root_axes_head_relative",
                referencePose = "current_editor_pose",
                floorReference = floorReference,
                referenceHeightMeters = referenceHeight,
                bones = exportedBones.ToArray(),
            };
        }

        private static Animator ResolveSelectedAnimator()
        {
            GameObject selected = Selection.activeGameObject;
            return selected == null ? null : selected.GetComponentInParent<Animator>();
        }

        private static Transform RequireBone(
            Animator animator,
            HumanBodyBones humanBone,
            string profileName)
        {
            Transform bone = animator.GetBoneTransform(humanBone);
            if (bone == null)
            {
                throw new InvalidOperationException(
                    "Required Humanoid bone is missing: " + profileName);
            }
            return bone;
        }

        private static Quaternion NormalizeCanonical(Quaternion value)
        {
            float magnitude = Mathf.Sqrt(
                value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);
            if (!IsFinitePositive(magnitude))
            {
                throw new InvalidOperationException("A Humanoid bone has an invalid rotation.");
            }

            float sign = value.w < 0f ? -1f : 1f;
            float scale = sign / magnitude;
            return new Quaternion(
                value.x * scale,
                value.y * scale,
                value.z * scale,
                value.w * scale);
        }

        private static bool IsFinitePositive(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value) && value > 0f;
        }

        private static string SanitizeFileName(string value)
        {
            var invalid = new HashSet<char>(Path.GetInvalidFileNameChars());
            var builder = new StringBuilder(value.Length);
            foreach (char character in value)
            {
                builder.Append(invalid.Contains(character) ? '_' : character);
            }
            string result = builder.ToString().Trim();
            return string.IsNullOrEmpty(result) ? "avatar" : result;
        }

        private static void ShowError(string message)
        {
            EditorUtility.DisplayDialog("VRC Ardy avatar rig export failed", message, "OK");
        }

        private readonly struct BoneSpec
        {
            public readonly string profileName;
            public readonly HumanBodyBones humanBone;
            public readonly bool required;

            public BoneSpec(string profileName, HumanBodyBones humanBone, bool required)
            {
                this.profileName = profileName;
                this.humanBone = humanBone;
                this.required = required;
            }
        }

        [Serializable]
        private sealed class AvatarRigProfileDocument
        {
            public int schemaVersion;
            public string profileName;
            public string units;
            public string coordinateSpace;
            public string referencePose;
            public string floorReference;
            public float referenceHeightMeters;
            public BonePoseDocument[] bones;
        }

        [Serializable]
        private sealed class BonePoseDocument
        {
            public string name;
            public float[] positionMeters;
            public float[] rotationXyzw;
        }
    }
}
