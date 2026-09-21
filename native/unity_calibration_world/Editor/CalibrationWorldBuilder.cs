using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;
using UdonSharp;
using UdonSharpEditor;
using VRC.SDK3.Components;
using VRC.Core;

public static class CalibrationWorldBuilder
{
    private const string ScenePath = "Assets/VrcAgentCalibration/Scenes/Calibration.unity";

    [MenuItem("VRC Agent/Create Calibration Scene")]
    public static void CreateAndVerify()
    {
        // Interactive use must not discard another scene's unsaved work.
        if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
        floor.name = "Calibration Floor";
        floor.transform.position = new Vector3(0, -.1f, 0);
        floor.transform.localScale = new Vector3(12, .2f, 12);
        var spawn = new GameObject("Spawn");
        spawn.transform.position = Vector3.zero;
        var descriptor = new GameObject("World").AddComponent<VRCSceneDescriptor>();
        descriptor.spawns = new[] { spawn.transform };
        descriptor.gameObject.AddComponent<PipelineManager>();
        var light = new GameObject("Light").AddComponent<Light>();
        light.type = LightType.Directional;
        light.transform.rotation = Quaternion.Euler(50, -30, 0);
        RenderSettings.ambientLight = new Color(.65f, .65f, .65f);
        var canvas = new GameObject("Measurement Panel", typeof(Canvas));
        canvas.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
        canvas.transform.position = new Vector3(0, 1.8f, 3);
        canvas.transform.localScale = Vector3.one * .003f;
        var canvasRect = canvas.GetComponent<RectTransform>();
        canvasRect.sizeDelta = new Vector2(1000, 800);
        var panel = new GameObject("Status", typeof(RectTransform), typeof(Text));
        panel.transform.SetParent(canvas.transform, false);
        var text = panel.GetComponent<Text>();
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        text.fontSize = 34;
        text.color = Color.black;
        text.alignment = TextAnchor.MiddleCenter;
        text.text = "VRC AGENT / CALIBRATION\nWaiting for local player";
        text.rectTransform.sizeDelta = canvasRect.sizeDelta;
        var button = GameObject.CreatePrimitive(PrimitiveType.Cube);
        button.name = "Record - Stop or Restart";
        button.transform.position = new Vector3(0, .8f, 2);
        button.transform.localScale = new Vector3(.5f, .25f, .25f);
        Directory.CreateDirectory("Assets/VrcAgentCalibration/Generated");
        var material = new Material(Shader.Find("Standard"));
        material.color = new Color(.2f, .7f, .3f);
        const string materialPath = "Assets/VrcAgentCalibration/Generated/Record.mat";
        var existing = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
        if (existing == null) AssetDatabase.CreateAsset(material, materialPath);
        else { EditorUtility.CopySerialized(material, existing); UnityEngine.Object.DestroyImmediate(material); material = existing; }
        button.GetComponent<Renderer>().sharedMaterial = material;
        var probe = button.AddUdonSharpComponent<CalibrationProbe>();
        probe.statusText = text;
        probe.sampleInterval = .5f;
        probe.captureSeconds = 60f;
        UdonSharpEditorUtility.CopyProxyToUdon(probe);
        UdonSharpProgramAsset.CompileAllCsPrograms(true);
        var backing = UdonSharpEditorUtility.GetBackingUdonBehaviour(probe);
        if (backing == null || backing.programSource == null) throw new Exception("Calibration Udon program missing");
        backing.interactText = "Stop / restart measurement";
        backing.proximity = 5f;
        Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
        if (!EditorSceneManager.SaveScene(scene, ScenePath)) throw new Exception("Scene save failed");
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();
        Debug.Log("VRC_AGENT_CALIBRATION_SCENE_READY " + ScenePath);
    }
}
