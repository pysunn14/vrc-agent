import test from 'node:test';
import assert from 'node:assert/strict';
import {calibrateAvatar} from '../src/setup-calibration.js';
import {blankProfile} from '../src/setup-profile.js';
import {SetupCanceled,type Prompter} from '../src/prompts.js';
import type {CoreClient} from '../src/core.js';
import type {Bootstrap} from '../src/types.js';

function fixture(){
 const profile=blankProfile({os:'macos',python:'python',cwd:'.'} as Bootstrap,'ko');
 profile.hosts.game={os:'windows',address:'example'};profile.avatar.rig='rig.json';
 const calls:string[]=[];
 const core={call:async(action:string,payload:any)=>{
  calls.push(action);
  if(action==='calibration.prepare')return {...profile.avatar,base_pose:'candidate.json'};
  if(action==='calibration.status')return {available:true,hmd_base:[0,1,0]};
  if(action==='calibration.start')return {token:'session',revision:0,state:'active'};
  if(action==='calibration.adjust')return {...profile.avatar,base_pose:'adjusted.json'};
  if(action==='calibration.update'){assert.equal(payload.revision,0);return {token:'session',revision:1,state:'active'};}
  if(action==='calibration.complete')return {avatar:{...profile.avatar,calibration:{signature:'a'.repeat(64),tracking:'active'}}};
  if(action==='calibration.cancel')return {state:'canceled',restored:true};
  throw Error(action);
 }} as unknown as CoreClient;
 return {profile,core,calls};
}

test('first avatar generates a candidate, adjusts, confirms and saves without a pose path question',async()=>{
 const {profile,core,calls}=fixture();let selected=0;
 const ui:Prompter={select:async(_m,options)=>{const value=selected++?'done':'y+';return options.find(o=>o.value===value)!.value;},
 text:async()=>{throw Error('unexpected file entry');},confirm:async()=>true,note:()=>{}};
 assert.equal(await calibrateAvatar(core,profile,ui),true);
 assert.ok(profile.avatar.calibration);
 assert.deepEqual(calls,['calibration.prepare','calibration.status','calibration.start','calibration.adjust','calibration.update','calibration.complete']);
});
test('cancel restores the remote output without attesting the candidate',async()=>{
 const {profile,core,calls}=fixture();
 const ui:Prompter={select:async()=>{throw new SetupCanceled();},text:async()=>'',confirm:async()=>true,note:()=>{}};
 await assert.rejects(calibrateAvatar(core,profile,ui),SetupCanceled);
 assert.equal(calls.at(-1),'calibration.cancel');
 assert.equal(profile.avatar.calibration,undefined);
});
test('offline preparation leaves a resumable candidate',async()=>{
 const {profile,core,calls}=fixture();
 const ui:Prompter={select:async()=>{throw Error('unexpected');},text:async()=>'',confirm:async()=>false,note:()=>{}};
 assert.equal(await calibrateAvatar(core,profile,ui),false);
 assert.equal(profile.avatar.base_pose,'candidate.json');
 assert.deepEqual(calls,['calibration.prepare']);
});
