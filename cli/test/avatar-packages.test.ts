import test from 'node:test';
import assert from 'node:assert/strict';
import {chooseAvatarPackage} from '../src/setup-avatar-package.js';
import {blankProfile} from '../src/setup-profile.js';
import type {Bootstrap} from '../src/types.js';
import type {CoreClient} from '../src/core.js';
import type {Prompter} from '../src/prompts.js';

for (const language of ['en', 'ko'] as const) {
  test(`prepared avatar needs no individual file paths and remains unverified (${language})`, async () => {
    const profile = blankProfile({os:'macos', python:'python', cwd:'.'} as Bootstrap, language);
    const avatar = {rig:'/registered/rig.json',base_pose:'/registered/base.json',hmd_base:[0,1,0],face_channels:{}};
    const calls: string[] = [];
    const core = {call: async(action:string) => {
      calls.push(action);
      if (action==='avatars.list') return {items:[{key:'abc',name:'Example',avatar_version:'1',conditions:'Default scale',face_channels:[],source:'user'}],problems:[]};
      if (action==='avatars.select') return avatar;
      throw Error(action);
    }} as unknown as CoreClient;
    const ui: Prompter = {select:async(_m,options) => options[0].value,
      text:async()=>{throw Error('must not ask for file paths');},confirm:async()=>true,note:()=>{}};
    assert.equal(await chooseAvatarPackage(core,profile,ui),'changed');
    assert.deepEqual(profile.avatar,avatar);
    assert.equal(profile.avatar.calibration,undefined);
    assert.deepEqual(calls,['avatars.list','avatars.select']);
  });
}

test('empty registry explains preparation and allows skipping without paths', async()=>{
  const profile=blankProfile({os:'linux',python:'python',cwd:'.'} as Bootstrap,'ko');
  const notes:string[]=[];
  const core={call:async()=>({items:[],problems:[]})} as unknown as CoreClient;
  const ui:Prompter={select:async(_m,options)=>{
    assert.ok(!options.some(o=>o.value==='prepared'));
    return options.find(o=>o.value==='later')!.value;
  },text:async()=>{throw Error('unexpected');},confirm:async()=>false,note:m=>{notes.push(m);}};
  assert.equal(await chooseAvatarPackage(core,profile,ui),'later');
  assert.ok(notes.some(n=>n.includes('등록된')));
  assert.equal(profile.avatar.rig,'');
});

test('import previews the package and registers exactly the reviewed content', async()=>{
  const profile=blankProfile({os:'linux',python:'python',cwd:'.'} as Bootstrap,'ko');
  const item={key:'expected',name:'Example',avatar_version:'1',conditions:'Default',face_channels:[],source:'user'};
  const calls: string[]=[];
  const core={call:async(action:string,payload:Record<string,unknown>)=>{
    calls.push(action);
    if(action==='avatars.list') return {items:[],problems:[]};
    if(action==='avatars.inspect') return item;
    if(action==='avatars.import') {assert.equal(payload.expected_key,'expected');return item;}
    if(action==='avatars.select') return {rig:'/local/rig',base_pose:'/local/base',hmd_base:[0,1,0],face_channels:{}};
    throw Error(action);
  }} as unknown as CoreClient;
  const ui:Prompter={select:async(_m,options)=>options.find(o=>o.value==='import')!.value,
    text:async()=>'/shared/avatar.json',confirm:async()=>true,note:()=>{}};
  await chooseAvatarPackage(core,profile,ui);
  assert.deepEqual(calls,['avatars.list','avatars.inspect','avatars.import','avatars.select']);
  assert.equal(profile.avatar.rig,'/local/rig');
});

test('creating a rig-only draft asks about the missing pose once and preserves the rig', async()=>{
  const {setupAvatar}=await import('../src/setup-avatar.js');
  const state={os:'linux',python:'python',cwd:'.'} as Bootstrap;
  const profile=blankProfile(state,'ko');
  let confirmations=0;
  const core={call:async(action:string)=>{
    if(action==='avatars.list') return {items:[],problems:[]};
    if(action==='avatar.rig.inspect') return {name:'Example'};
    if(action==='calibration.prepare') return {...profile.avatar,base_pose:'/candidate.json'};
    throw Error(action);
  }} as unknown as CoreClient;
  const ui:Prompter={select:async(_m,options)=>options.find(o=>o.value==='create')?.value ?? options[0].value,
    text:async()=>'/exports/rig.json',confirm:async()=>{confirmations++;return false;},note:()=>{}};
  await setupAvatar(core,state,profile,ui);
  assert.equal(confirmations,1);
  assert.equal(profile.avatar.rig,'/exports/rig.json');
  assert.equal(profile.avatar.base_pose,'/candidate.json');
});
