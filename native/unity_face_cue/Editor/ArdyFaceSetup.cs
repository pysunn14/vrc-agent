using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;
using VRC.SDK3.Avatars.Components;
using VRC.SDK3.Avatars.ScriptableObjects;

namespace VrcArdy.Editor {
// Explicit command files make editor automation inspectable and one-shot.
[InitializeOnLoad]
public static class ArdyFaceSetup {
    const string Root = "Assets/VrcArdy/FaceCue";
    const string Command = "Temp/ardy-face-command.txt";
    const string Result = "Temp/ardy-face-result.json";
    const string Parameter = "ArdyYawn";
    static bool busy;
    static double nextHeartbeat;
    static string phase = "ready";
    [Serializable] class Status { public string state; public string detail; public double heartbeat; }
    static ArdyFaceSetup() { EditorApplication.update += Tick; }
    static void Report(string state, string detail = "") {
        phase = state;
        File.WriteAllText(Result, JsonUtility.ToJson(new Status { state=state, detail=detail, heartbeat=EditorApplication.timeSinceStartup }));
    }
    static async void Tick() {
        if (busy) { if (EditorApplication.timeSinceStartup > nextHeartbeat) { Report(phase); nextHeartbeat=EditorApplication.timeSinceStartup+1; } return; }
        if (EditorApplication.isCompiling || EditorApplication.isUpdating || !File.Exists(Command)) return;
        string operation=File.ReadAllText(Command).Trim(); File.Delete(Command); busy=true;
        try {
            if (operation == "setup") { Report("configuring"); Configure(); Report("configured", "Dedicated facial layer and synced float installed; validation passed"); }
            else if(operation == "upload") {
                Report("uploading");
                var avatar=Avatar();
                var pipeline=avatar.GetComponent<VRC.Core.PipelineManager>();
                if(pipeline == null || string.IsNullOrEmpty(pipeline.blueprintId)) throw new Exception("Existing avatar ID required");
                if(!VRCSdkControlPanel.TryGetBuilder<VRC.SDK3A.Editor.IVRCSdkAvatarBuilderApi>(out var builder)) throw new Exception("Open SDK Builder panel first");
                var data=await VRC.SDKBase.Editor.Api.VRCApi.GetAvatar(pipeline.blueprintId, true);
                int previous=data.Version;
                await builder.BuildAndUpload(avatar.gameObject,data);
                var verified=await VRC.SDKBase.Editor.Api.VRCApi.GetAvatar(pipeline.blueprintId,true);
                if(verified.Version <= previous) throw new Exception("Remote avatar version did not advance");
                Report("uploaded", "Remote version advanced to "+verified.Version);
            } else throw new Exception("Unknown command");
        } catch(Exception e) { Debug.LogException(e); Report("failed",e.ToString()); }
        finally { busy=false; }
    }
    static VRCAvatarDescriptor Avatar() {
        var all=UnityEngine.Object.FindObjectsOfType<VRCAvatarDescriptor>();
        if(all.Length != 1) throw new Exception("Expected exactly one active avatar in the scene");
        return all[0];
    }
    static void Configure() {
        var avatar=Avatar();
        if(!avatar.gameObject.name.ToLowerInvariant().Contains("shinano")) throw new Exception("Expected Shinano avatar");
        if(EditorApplication.isPlayingOrWillChangePlaymode) throw new Exception("Stop Play Mode first");
        Directory.CreateDirectory(Root);AssetDatabase.Refresh();
        var layers=avatar.baseAnimationLayers;
        int index=Array.FindIndex(layers,x=>x.type==VRCAvatarDescriptor.AnimLayerType.FX);
        if(index<0 || !(layers[index].animatorController is AnimatorController original)) throw new Exception("Custom FX controller required");
        string controllerPath=Root+"/ShinanoFaceFX.controller";
        if(AssetDatabase.GetAssetPath(original)!=controllerPath && !AssetDatabase.CopyAsset(AssetDatabase.GetAssetPath(original),controllerPath)) throw new Exception("Cannot copy FX controller");
        var controller=AssetDatabase.LoadAssetAtPath<AnimatorController>(controllerPath);
        if(controller.parameters.Any(x=>x.name==Parameter)) controller.RemoveParameter(controller.parameters.First(x=>x.name==Parameter));
        controller.AddParameter(Parameter,AnimatorControllerParameterType.Float);
        for(int i=controller.layers.Length-1;i>=0;i--) if(controller.layers[i].name=="Ardy Face Cue") controller.RemoveLayer(i);
        string baselinePath=AssetDatabase.FindAssets("Default_face t:AnimationClip",new[]{"Assets/Shinano"}).Select(AssetDatabase.GUIDToAssetPath).Single();
        var originalClip=AssetDatabase.LoadAssetAtPath<AnimationClip>(baselinePath);
        var bindings=AnimationUtility.GetCurveBindings(originalClip).Where(b=>b.propertyName.StartsWith("blendShape.")).ToArray();
        if(!bindings.Any(b=>b.propertyName=="blendShape.eye_close") || !bindings.Any(b=>b.propertyName=="blendShape.mouth_a1")) throw new Exception("Required eye/mouth blendshapes absent");
        var neutral=MakeClip("Neutral",originalClip,bindings,false);
        var yawn=MakeClip("Yawn",originalClip,bindings,true);
        var sm=new AnimatorStateMachine {name="Ardy Face Cue"};AssetDatabase.AddObjectToAsset(sm,controller);
        var layer=new AnimatorControllerLayer {name="Ardy Face Cue",stateMachine=sm,defaultWeight=1};controller.AddLayer(layer);
        var off=sm.AddState("Original Face");off.writeDefaultValues=false;sm.defaultState=off;
        var on=sm.AddState("Yawn");on.writeDefaultValues=false;
        var tree=new BlendTree {name="Yawn Intensity",blendType=BlendTreeType.Simple1D,blendParameter=Parameter,useAutomaticThresholds=false};AssetDatabase.AddObjectToAsset(tree,controller);
        tree.AddChild(neutral,0);tree.AddChild(yawn,1);on.motion=tree;
        var enter=off.AddTransition(on);enter.hasExitTime=false;enter.duration=.08f;enter.AddCondition(AnimatorConditionMode.Greater,.001f,Parameter);
        var leave=on.AddTransition(off);leave.hasExitTime=false;leave.duration=.15f;leave.AddCondition(AnimatorConditionMode.Less,.001f,Parameter);
        var controlled=on.AddStateMachineBehaviour<VRCAnimatorTrackingControl>();controlled.trackingEyes=VRC.SDKBase.VRC_AnimatorTrackingControl.TrackingType.Animation;controlled.trackingMouth=VRC.SDKBase.VRC_AnimatorTrackingControl.TrackingType.Animation;
        var normal=off.AddStateMachineBehaviour<VRCAnimatorTrackingControl>();normal.trackingEyes=VRC.SDKBase.VRC_AnimatorTrackingControl.TrackingType.Tracking;normal.trackingMouth=VRC.SDKBase.VRC_AnimatorTrackingControl.TrackingType.Tracking;
        layers[index].isDefault=false;layers[index].animatorController=controller;avatar.baseAnimationLayers=layers;
        string paramsPath=Root+"/FaceParameters.asset";
        if(avatar.expressionParameters==null) throw new Exception("Expression parameters required");
        if(AssetDatabase.GetAssetPath(avatar.expressionParameters)!=paramsPath && !AssetDatabase.CopyAsset(AssetDatabase.GetAssetPath(avatar.expressionParameters),paramsPath)) throw new Exception("Cannot copy expression parameters");
        var parameters=AssetDatabase.LoadAssetAtPath<VRCExpressionParameters>(paramsPath);
        parameters.parameters=parameters.parameters.Where(p=>p.name!=Parameter).Concat(new[]{new VRCExpressionParameters.Parameter {name=Parameter,valueType=VRCExpressionParameters.ValueType.Float,defaultValue=0,saved=false,networkSynced=true}}).ToArray();
        if(parameters.CalcTotalCost()>VRCExpressionParameters.MAX_PARAMETER_COST) throw new Exception("Expression parameter budget exceeded");
        avatar.expressionParameters=parameters;
        EditorUtility.SetDirty(controller);EditorUtility.SetDirty(parameters);EditorUtility.SetDirty(avatar);
        AssetDatabase.SaveAssets();EditorSceneManager.MarkSceneDirty(avatar.gameObject.scene);EditorSceneManager.SaveScene(avatar.gameObject.scene);
        Selection.activeGameObject=avatar.gameObject;
    }
    static AnimationClip MakeClip(string name,AnimationClip source,EditorCurveBinding[] bindings,bool yawn) {
        string path=Root+"/"+name+".anim";
        var clip=AssetDatabase.LoadAssetAtPath<AnimationClip>(path);
        if(clip==null){clip=new AnimationClip {name=name,frameRate=20};AssetDatabase.CreateAsset(clip,path);}else clip.ClearCurves();
        foreach(var binding in bindings){
            float value=AnimationUtility.GetEditorCurve(source,binding).Evaluate(0);
            if(yawn && binding.propertyName=="blendShape.eye_close") value=90;
            if(yawn && binding.propertyName=="blendShape.mouth_a1") value=75;
            AnimationUtility.SetEditorCurve(clip,binding,AnimationCurve.Constant(0,1,value));
        }
        EditorUtility.SetDirty(clip);return clip;
    }
}
}
