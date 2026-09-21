using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;
using VRC.SDK3.Avatars.Components;
using Tracking = VRC.SDKBase.VRC_AnimatorTrackingControl.TrackingType;

namespace VrcArdy.Editor {
// Editor-only experiment. Runtime uses SDK states, never a custom avatar script.
[InitializeOnLoad]
public static class ArdyWalkArms {
    const string Root = "Assets/VrcArdy/WalkArms";
    const string Output = Root + "/ShinanoWalkArms.controller";
    const string Source = "Assets/Shinano/Animation/Animator/Shinano_Locomotion.controller";
    const string Walk = "Ardy Walking Arms";
    const string Command = "Temp/ardy-walk-command.txt";
    const string Result = "Temp/ardy-walk-result.json";
    [Serializable] class Status { public string state; public string detail; }
    static ArdyWalkArms() { EditorApplication.update += Tick; }
    static void Tick() {
        if(EditorApplication.isCompiling || EditorApplication.isUpdating || !File.Exists(Command)) return;
        string command=File.ReadAllText(Command).Trim(); File.Delete(Command);
        try {
            if(EditorApplication.isPlayingOrWillChangePlaymode) throw new Exception("Edit mode required");
            if(command=="setup") Configure();
            else if(command!="validate") throw new Exception("Unknown command");
            Validate();
            File.WriteAllText(Result,JsonUtility.ToJson(new Status {state="passed",detail=command+": graph, ownership, hysteresis and preserved exits verified"}));
        } catch(Exception e) {
            Debug.LogException(e);
            File.WriteAllText(Result,JsonUtility.ToJson(new Status {state="failed",detail=e.ToString()}));
        }
    }
    static VRCAvatarDescriptor Avatar() {
        var avatars=UnityEngine.Object.FindObjectsOfType<VRCAvatarDescriptor>();
        if(avatars.Length!=1 || !avatars[0].name.ToLowerInvariant().Contains("shinano")) throw new Exception("Exactly one active Shinano required");
        return avatars[0];
    }
    static IEnumerable<AnimatorState> States(AnimatorStateMachine sm) {
        foreach(var s in sm.states) yield return s.state;
        foreach(var child in sm.stateMachines) foreach(var s in States(child.stateMachine)) yield return s;
    }
    static AnimatorStateTransition Transition(AnimatorState from, AnimatorState to, string parameter, AnimatorConditionMode mode, float threshold) {
        var t=from.AddTransition(to); t.hasExitTime=false; t.hasFixedDuration=true; t.duration=.15f;
        t.AddCondition(mode,threshold,parameter); return t;
    }
    static void Parameter(AnimatorController controller,string name,AnimatorControllerParameterType type) {
        var p=controller.parameters.FirstOrDefault(x=>x.name==name);
        if(p==null) controller.AddParameter(name,type);
        else if(p.type!=type) throw new Exception("Wrong parameter type: "+name);
    }
    static void Configure() {
        var avatar=Avatar(); var layers=avatar.baseAnimationLayers;
        int index=Array.FindIndex(layers,x=>x.type==VRCAvatarDescriptor.AnimLayerType.Base);
        if(index<0) throw new Exception("Base layer absent");
        string current=AssetDatabase.GetAssetPath(layers[index].animatorController);
        if(current!=Source && current!=Output) throw new Exception("Unexpected Base controller; refusing to replace it");
        // Re-entry validates the owned graph instead of accumulating states/subassets.
        if(current==Output) { Validate(); return; }
        if(AssetDatabase.LoadAssetAtPath<AnimatorController>(Output)!=null) throw new Exception("Unassigned trial controller already exists; inspect before retry");
        Directory.CreateDirectory(Root); AssetDatabase.Refresh();
        if(!AssetDatabase.CopyAsset(Source,Output)) throw new Exception("Cannot copy locomotion controller");
        var controller=AssetDatabase.LoadAssetAtPath<AnimatorController>(Output);
        Parameter(controller,"VelocityMagnitude",AnimatorControllerParameterType.Float);
        Parameter(controller,"ArdyYawn",AnimatorControllerParameterType.Float);
        var sm=controller.layers[0].stateMachine;
        var standing=sm.states.Single(x=>x.state.name=="Standing").state;
        var originalExits=standing.transitions.ToArray();
        var walking=sm.AddState(Walk); walking.motion=standing.motion;
        walking.writeDefaultValues=standing.writeDefaultValues; walking.speed=standing.speed;
        // Preserve crouch/jump/AFK exits and their original priority, including nested destinations.
        foreach(var original in originalExits) {
            var copy=UnityEngine.Object.Instantiate(original);
            AssetDatabase.AddObjectToAsset(copy,controller);
            walking.transitions=walking.transitions.Concat(new[]{copy}).ToArray();
        }
        // SDK ownership persists across states: resolve NoChange on destination states,
        // but preserve explicit Animation settings used by landing/fall animations.
        foreach(var state in States(sm).Where(s=>s!=walking)) {
            var controls=state.behaviours.OfType<VRCAnimatorTrackingControl>().ToArray();
            if(controls.Length==0) controls=new[]{state.AddStateMachineBehaviour<VRCAnimatorTrackingControl>()};
            foreach(var control in controls) {
                if(control.trackingLeftHand==Tracking.NoChange) control.trackingLeftHand=Tracking.Tracking;
                if(control.trackingRightHand==Tracking.NoChange) control.trackingRightHand=Tracking.Tracking;
                EditorUtility.SetDirty(control);
            }
        }
        var owner=walking.AddStateMachineBehaviour<VRCAnimatorTrackingControl>();
        owner.trackingLeftHand=Tracking.Animation; owner.trackingRightHand=Tracking.Animation;
        var enter=Transition(standing,walking,"VelocityMagnitude",AnimatorConditionMode.Greater,.1f);
        enter.AddCondition(AnimatorConditionMode.Less,.001f,"ArdyYawn");
        enter.AddCondition(AnimatorConditionMode.If,0,"Grounded");
        enter.AddCondition(AnimatorConditionMode.IfNot,0,"Seated");
        enter.AddCondition(AnimatorConditionMode.IfNot,0,"AFK");
        enter.AddCondition(AnimatorConditionMode.Greater,.7f,"Upright");
        Transition(walking,standing,"VelocityMagnitude",AnimatorConditionMode.Less,.05f);
        Transition(walking,standing,"ArdyYawn",AnimatorConditionMode.Greater,.001f);
        Transition(walking,standing,"Seated",AnimatorConditionMode.If,0);
        layers[index].isDefault=false; layers[index].animatorController=controller;
        avatar.baseAnimationLayers=layers;
        EditorUtility.SetDirty(controller); EditorUtility.SetDirty(avatar);
        AssetDatabase.SaveAssets();
        Validate();
        EditorSceneManager.MarkSceneDirty(avatar.gameObject.scene);
        EditorSceneManager.SaveScene(avatar.gameObject.scene);
    }
    static void Require(bool condition,string message) { if(!condition) throw new Exception(message); }
    static bool Has(AnimatorStateTransition t,string p,AnimatorConditionMode mode,float value) {
        return t.conditions.Any(c=>c.parameter==p && c.mode==mode && Mathf.Approximately(c.threshold,value));
    }
    public static void Validate() {
        var avatar=Avatar();
        var layer=avatar.baseAnimationLayers.Single(x=>x.type==VRCAvatarDescriptor.AnimLayerType.Base);
        Require(AssetDatabase.GetAssetPath(layer.animatorController)==Output,"Trial Base controller is not assigned");
        var controller=(AnimatorController)layer.animatorController;
        var sm=controller.layers[0].stateMachine;
        var standing=sm.states.Single(x=>x.state.name=="Standing").state;
        var walking=sm.states.Single(x=>x.state.name==Walk).state;
        Require(sm.defaultState==standing,"Default must retain tracked idle");
        Require(standing.motion==walking.motion,"Walking must preserve the locomotion blend tree");
        Require(controller.layers[0].avatarMask==null,"Base must not mask arms");
        var enter=standing.transitions.Single(t=>t.destinationState==walking);
        Require(Has(enter,"VelocityMagnitude",AnimatorConditionMode.Greater,.1f) && Has(enter,"ArdyYawn",AnimatorConditionMode.Less,.001f),"Entry speed/yawn guard missing");
        Require(Has(enter,"Grounded",AnimatorConditionMode.If,0) && Has(enter,"Seated",AnimatorConditionMode.IfNot,0) && Has(enter,"AFK",AnimatorConditionMode.IfNot,0),"Entry context guards missing");
        Require(walking.transitions.Any(t=>t.destinationState==standing && Has(t,"VelocityMagnitude",AnimatorConditionMode.Less,.05f)),"Stop hysteresis missing");
        Require(walking.transitions.Any(t=>t.destinationState==standing && Has(t,"ArdyYawn",AnimatorConditionMode.Greater,.001f)),"Yawn must restore hand tracking");
        foreach(var state in States(sm)) {
            var controls=state.behaviours.OfType<VRCAnimatorTrackingControl>().ToArray();
            Require(controls.Length>0,"Hand ownership undefined: "+state.name);
            Require(controls.All(c=>c.trackingLeftHand!=Tracking.NoChange && c.trackingRightHand!=Tracking.NoChange),"Hand ownership inherited: "+state.name);
        }
        var owner=walking.behaviours.OfType<VRCAnimatorTrackingControl>().Single();
        Require(owner.trackingLeftHand==Tracking.Animation && owner.trackingRightHand==Tracking.Animation,"Walk must own both arms");
        Require(standing.behaviours.OfType<VRCAnimatorTrackingControl>().All(c=>c.trackingLeftHand==Tracking.Tracking && c.trackingRightHand==Tracking.Tracking),"Idle must restore both hands");
        var exits=standing.transitions.Where(t=>t.destinationState!=walking).ToArray();
        Require(walking.transitions.Length==exits.Length+3,"Unexpected walk exits");
        for(int i=0;i<exits.Length;i++) {
            var a=exits[i]; var b=walking.transitions[i];
            Require(a.destinationState==b.destinationState && a.destinationStateMachine==b.destinationStateMachine && a.isExit==b.isExit && a.conditions.SequenceEqual(b.conditions),"Original exit changed");
        }
    }
}
}
