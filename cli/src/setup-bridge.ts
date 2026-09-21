import {chooseAvatarPackage} from './setup-avatar-package.js';
import type { CoreClient } from './core.js';
import type { Bootstrap, BridgeProfile, CombinedProfile } from './types.js';
import type { Prompter } from './prompts.js';
import { SetupCanceled } from './prompts.js';
import { selectAudioDevice, type BridgeInventory } from './setup-audio.js';
import { t } from './i18n.js';

export async function setupBridge(core: CoreClient, state: Bootstrap, profile: BridgeProfile | CombinedProfile, prompts: Prompter,
  setup: {advanced?: boolean} = {}): Promise<void> {
  const ko = profile.language === 'ko', bridge = profile.bridge;
  bridge.project_dir ||= state.cwd;
  if (setup.advanced) bridge.project_dir = await prompts.text(t(profile.language, 'windowsProject'), bridge.project_dir);
  let observed: BridgeInventory;
  while (true) {
    try {
      // Project discovery chooses its environment, never an arbitrary globally
      // installed Python. Saved explicit environments take precedence.
      if (!bridge.python || bridge.python === state.python) bridge.python = (await core.call<{python: string}>(
        'runtime.discover', {root: bridge.project_dir, base_dir: process.cwd()})).python;
      if (setup.advanced) bridge.python = await prompts.text(t(profile.language, 'windowsPython'), bridge.python);
      Object.assign(bridge, await core.call('bridge.paths', {python: bridge.python, project_dir: bridge.project_dir, base_dir: process.cwd()}));
      prompts.note(ko ? 'Windows 환경과 음성 장치를 확인합니다.' : 'Inspecting the Windows environment and audio devices.');
      observed = await core.call<BridgeInventory>('bridge.discover', {python: bridge.python, project_dir: bridge.project_dir, base_dir: process.cwd()});
      if (observed.platform !== 'Windows') throw new Error(ko ? '선택한 Python이 Windows 환경이 아닙니다.' : 'Selected Python is not a Windows environment.');
      break;
    } catch (error) {
      prompts.note(String(error));
      const choice = await prompts.select(ko ? 'Windows 환경을 확인할 수 없습니다' : 'Cannot inspect the Windows environment', [
        {value: 'edit', label: ko ? '프로젝트·Python 경로 수정' : 'Edit project / Python paths'},
        {value: 'retry', label: ko ? '다시 확인' : 'Retry'},
        {value: 'pause', label: ko ? '초안 저장 후 나가기' : 'Save draft and exit'},
      ]);
      if (choice === 'pause') throw new SetupCanceled();
      if (choice === 'edit') {
        bridge.project_dir = await prompts.text(t(profile.language, 'windowsProject'), bridge.project_dir);
        bridge.python = await prompts.text(t(profile.language, 'windowsPython'), bridge.python);
      }
    }
  }
  const missing = Object.entries(observed.packages).filter(([, version]) => !version).map(([name]) => name);
  if (missing.length) prompts.note((ko ? 'Python에 설치되지 않은 패키지: ' : 'Missing Python packages: ') + missing.join(', '));
  if (observed.audio_error) prompts.note(observed.audio_error);
  if (profile.role === 'bridge') {
    prompts.note(ko ? '에이전트와 같은 아바타 설정 묶음을 가져오세요. 골격과 머리 기준 위치를 함께 적용합니다.'
      : 'Import the same avatar package as the agent. Rig and head origin are applied together.');
    const settings = {language: profile.language, avatar: {rig: profile.bridge.avatar_rig, base_pose: '',
      hmd_base: profile.bridge.hmd_base, face_channels: {}}};
    await chooseAvatarPackage(core, settings, prompts, {allowCreate: false});
    profile.bridge.avatar_rig = settings.avatar.rig;
    profile.bridge.hmd_base = settings.avatar.hmd_base;
  }
  bridge.body_output ??= 'vrchat-osc';
  if (setup.advanced) bridge.body_output = await prompts.select(ko ? '몸 추적 출력 방식' : 'Body tracking output', [
    {value: 'vrchat-osc' as const, label: 'VRChat OSC'}, {value: 'vmt' as const, label: 'VMT'},
  ], bridge.body_output);
  bridge.virtual_mic_device = await selectAudioDevice(prompts, observed.audio_devices, 'output', profile.language, bridge.virtual_mic_device, !!setup.advanced);
  if (setup.advanced) {
    const account = await prompts.text(ko ? '브리지 대상 VRChat 계정 이름 (선택)' : 'Target VRChat account name (optional)', bridge.vrchat_user_name, false);
    if (account) {bridge.vrchat_user_name = account; delete bridge.vrchat_pid;} else delete bridge.vrchat_user_name;
  }
  bridge.audio_source = await prompts.select(ko ? '어디서 대화를 들을까요?' : 'Where should the agent listen?', [
    {value: 'process', label: ko ? 'VRChat 게임 소리' : 'VRChat process audio'},
    {value: 'microphone', label: ko ? '마이크' : 'Microphone'},
  ], bridge.audio_source ?? 'process');
  if (bridge.audio_source === 'microphone') {
    delete bridge.process_audio_helper;
    bridge.microphone_device = await selectAudioDevice(prompts, observed.audio_devices, 'input', profile.language, bridge.microphone_device, !!setup.advanced);
  } else {
    delete bridge.microphone_device;
    if (setup.advanced) {
      const helper = await prompts.text(ko ? '프로세스 오디오 도우미 경로 (비우면 기본 위치)' : 'Process audio helper path (empty: default location)', bridge.process_audio_helper, false);
      if (helper) bridge.process_audio_helper = (await core.call<{helper: string}>('runtime.paths', {paths: {helper}, base_dir: process.cwd()})).helper;
      else delete bridge.process_audio_helper;
    }
    if (!bridge.process_audio_helper && !observed.process_audio_helper?.exists)
      prompts.note(ko ? '게임 소리 수집 도우미가 준비되지 않았습니다. doctor에서 설치 상태를 확인하세요.' : 'Process audio helper is not prepared. Use doctor to check installation.');
  }
  prompts.note(ko ? '설정한 음성 장치와 실제 추적 상태는 Windows에서 doctor로 확인하세요.' : 'Use doctor on Windows to check audio devices and actual tracking.');
}
