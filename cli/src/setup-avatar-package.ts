import {resolve} from 'node:path';
import type {CoreClient} from './core.js';
import type {AgentProfile} from './types.js';
import type {Prompter} from './prompts.js';
import {SetupCanceled} from './prompts.js';

export interface AvatarPackage {
  key: string; name: string; avatar_version: string; conditions: string;
  source: string; face_channels: string[];
}
interface Inventory {items: AvatarPackage[]; problems: {path: string; detail: string}[]}

function describe(item: AvatarPackage, ko: boolean): string {
  return `${item.name} · ${item.avatar_version}\n${item.conditions}\n`
    + (ko ? '✓ 골격과 기준 자세\n' : '✓ Rig and base pose\n')
    + (item.face_channels.length ? `${ko ? '표정 연결' : 'Face channels'}: ${item.face_channels.join(', ')}`
      : ko ? '표정 연결 없음' : 'No face channels')
    + '\n' + (ko ? '현재 환경에서의 실제 동작 확인은 별도로 필요합니다.' : 'Visual verification is still required in your environment.');
}

export async function avatarFile(prompts: Prompter, label: string, initial?: string): Promise<string> {
  return resolve(await (prompts.path ? prompts.path(label, initial) : prompts.text(label, initial)));
}

export async function chooseAvatarPackage(core: CoreClient, profile: Pick<AgentProfile, 'avatar'|'language'>, prompts: Prompter, options: {allowCreate?: boolean} = {}): Promise<'kept'|'changed'|'created'|'later'> {
  const ko=profile.language==='ko';
  const inventory=await core.call<Inventory>('avatars.list');
  for (const problem of inventory.problems) prompts.note(`${ko?'읽을 수 없는 아바타 설정':'Unreadable avatar settings'}: ${problem.path}\n${problem.detail}`);
  prompts.note(ko ? 'VRChat에서 사용할 아바타를 설정합니다. 준비된 설정에는 골격·기준 자세·선택적인 표정 연결이 함께 들어 있습니다.'
    : 'Set up the avatar you will use in VRChat. A prepared package contains a rig, base pose and optional face mappings.');
  if (!inventory.items.length) prompts.note(ko ? (options.allowCreate === false ? '등록된 아바타 설정이 없습니다. 에이전트에서 내보낸 설정 묶음을 가져오세요.' : '등록된 아바타 설정이 없습니다. 공유받은 설정 묶음을 가져오거나 내 아바타 설정을 만들 수 있습니다.')
    : (options.allowCreate === false ? 'No avatar packages are registered. Import the package exported by the agent.' : 'No avatar packages are registered. Import a shared package or create settings for your avatar.'));
  const choices = [
    ...(profile.avatar.rig ? [{value:'keep',label:ko?'현재 아바타 설정 유지':'Keep current avatar settings'}] : []),
    ...(inventory.items.length ? [{value:'prepared',label:ko?'준비된 아바타 설정 사용':'Use a prepared avatar package'}] : []),
    {value:'import',label:ko?'아바타 설정 묶음 가져오기':'Import an avatar package'},
    ...(options.allowCreate !== false ? [{value:'create',label:ko?'내 아바타 설정 만들기':'Create my avatar settings'}] : []),
    {value:'later',label:ko?'나중에 설정 — 아바타 동작은 준비되지 않습니다':'Configure later — avatar motion will remain unavailable'},
  ];
  const choice=await prompts.select(ko?'어떻게 아바타를 준비할까요?':'How would you like to prepare your avatar?',choices,profile.avatar.rig?'keep':inventory.items.length?'prepared':'later');
  if (choice==='keep') return 'kept';
  if (choice==='later') {
    profile.avatar={rig:'',base_pose:'',hmd_base:[0,1,0],face_channels:{}};
    return 'later';
  }
  if (choice==='create') {await createAvatarSettings(core,profile,prompts); return 'created';}
  let item: AvatarPackage;
  if (choice==='import') {
    while (true) {
      const path=await avatarFile(prompts,ko?'아바타 설정 묶음 JSON 파일':'Avatar package JSON');
      try {
        item=await core.call<AvatarPackage>('avatars.inspect',{path});
        prompts.note(describe(item,ko));
        if (!await prompts.confirm(ko?'이 설정을 등록하고 사용할까요?':'Register and use these settings?')) return 'kept';
        item=await core.call<AvatarPackage>('avatars.import',{path, expected_key:item.key});
        break;
      } catch(error) {
        if (error instanceof SetupCanceled) throw error;
        prompts.note(String(error));
        if (!await prompts.confirm(ko?'다른 설정 파일을 선택할까요?':'Choose another package file?',true)) return 'kept';
      }
    }
  } else {
    const key=await prompts.select(ko?'사용할 아바타와 같은 설정을 선택하세요':'Select the settings matching your avatar',inventory.items.map(item=>({value:item.key,label:`${item.name} · ${item.avatar_version}`})));
    item=inventory.items.find(item=>item.key===key)!;
    prompts.note(describe(item,ko));
    if (!await prompts.confirm(ko?'이 설정을 사용할까요?':'Use these settings?')) return 'kept';
  }
  profile.avatar=await core.call<AgentProfile['avatar']>('avatars.select',{key:item.key});
  return 'changed';
}

export async function createAvatarSettings(core: CoreClient, profile: Pick<AgentProfile, 'avatar'|'language'>, prompts: Prompter): Promise<void> {
  const ko=profile.language==='ko';
  prompts.note(ko ? '새 아바타는 Unity에서 골격을 내보낸 뒤 기준 자세를 교정해야 합니다. 모델·텍스처를 공유할 필요는 없습니다.\nUnity 도구: native/unity_avatar_rig_exporter/README.md\n교정 안내: docs/avatar-settings.md'
    : 'Export the rig from Unity, then calibrate a base pose. Model meshes and textures are not required.\nUnity exporter: native/unity_avatar_rig_exporter/README.md\nCalibration guide: docs/avatar-settings.md');
  const rig=await avatarFile(prompts,ko?'Unity에서 내보낸 골격 JSON 파일':'Rig JSON exported from Unity');
  // Validate the export before asking the user to prepare the next asset.
  await core.call('avatar.rig.inspect',{path:rig});
  profile.avatar={rig,base_pose:'',hmd_base:[0,1,0],face_channels:{}};
  prompts.note(ko?'골격을 가져왔습니다. 기준 자세 후보를 자동으로 만들고 VRChat에서 조정할 수 있습니다.'
    :'Rig imported. A base pose candidate can now be generated and adjusted in VRChat.');
}

export async function editAvatarAssets(core: CoreClient, profile: Pick<AgentProfile, 'avatar'|'language'>, prompts: Prompter): Promise<void> {
  const ko=profile.language==='ko', avatar=profile.avatar;
  avatar.base_pose=await avatarFile(prompts,ko?'교정한 기준 자세 JSON 파일':'Calibrated base pose JSON',avatar.base_pose);
  avatar.hmd_base=JSON.parse(await prompts.text(ko?'교정에 사용한 머리 기준 위치 [x, y, z] (미터)':'Head origin used for calibration [x, y, z] (meters)',JSON.stringify(avatar.hmd_base)));
  const face=await prompts.confirm(ko?'이 아바타에 표정 파라미터 연결을 설정할까요?':'Configure face parameter mappings for this avatar?',Object.keys(avatar.face_channels).length>0);
  if (face) avatar.face_channels=JSON.parse(await prompts.text(ko?'표정 채널 → 아바타 파라미터 (JSON)':'Face channel → avatar parameter (JSON)',JSON.stringify(avatar.face_channels)));
  else avatar.face_channels={};
  delete avatar.calibration;
  const observed=await core.call<{ready:boolean;detail:string}>('avatar.files.inspect',{avatar});
  if (!observed.ready) throw new Error(observed.detail);
}

export async function registerAvatarSettings(core: CoreClient, profile: Pick<AgentProfile, 'avatar'|'language'>, prompts: Prompter): Promise<void> {
  const ko=profile.language==='ko';
  const observed=await core.call<{ready:boolean;detail:string}>('avatar.inspect',{avatar:profile.avatar});
  if (!observed.ready) throw new Error(observed.detail);
  const name=await prompts.text(ko?'목록에 표시할 아바타 이름':'Avatar display name');
  const avatar_version=await prompts.text(ko?'아바타 버전 (모르면 미확인으로 기록)':'Avatar version (use unknown if uncertain)',ko?'미확인':'unknown');
  const conditions=await prompts.text(ko?'적용 조건 (크기 변경·추적 방식·필요한 표정 설정 등)':'Conditions (scale changes, tracking mode, required face setup)');
  const item=await core.call<AvatarPackage>('avatars.register',{avatar:profile.avatar,name,avatar_version,conditions});
  prompts.note((ko?'준비된 아바타 설정으로 등록했습니다: ':'Registered a prepared avatar package: ')+item.name);
}
