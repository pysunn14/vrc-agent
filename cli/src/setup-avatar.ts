import {calibrateAvatar} from './setup-calibration.js';
import {chooseAvatarPackage, registerAvatarSettings} from './setup-avatar-package.js';
import type { CoreClient } from './core.js';
import type { Bootstrap, AgentProfile } from './types.js';
import {SetupCanceled, type Prompter} from './prompts.js';

export async function setupAvatar(core: CoreClient, state: Bootstrap, profile: AgentProfile, prompts: Prompter, setup: {advanced?: boolean} = {}): Promise<void> {
  const ko = profile.language === 'ko';
  await chooseAvatarPackage(core, profile, prompts);
  let avatar = profile.avatar;
  if (avatar.rig) {
    const observed = avatar.base_pose ? await core.call<{ready: boolean}>('avatar.inspect', {avatar}) : {ready: false};
    if (!observed.ready || setup.advanced) {
      try { await calibrateAvatar(core, profile, prompts); }
      catch (error) {
        if (error instanceof SetupCanceled) throw error;
        prompts.note(String(error));
        prompts.note(ko ? '조정 파일은 보관됩니다. Windows 연결과 출력 상태를 확인한 뒤 다시 교정하세요.' : 'Adjustment files are retained. Check Windows connectivity and output, then retry calibration.');
        throw new SetupCanceled();
      }
    }
    avatar = profile.avatar;
    if (avatar.calibration && await prompts.confirm(ko ? '현재 설정을 재사용할 아바타 설정 묶음으로 등록할까요?' : 'Register the current settings as a reusable avatar package?', false))
      await registerAvatarSettings(core, profile, prompts);
  } else delete avatar.calibration;

  const choice = await prompts.select(ko ? '사용할 행동 목록' : 'Behavior library', [
    {value: 'keep', label: ko ? '현재 목록 유지' : 'Keep current library'},
    {value: 'import', label: ko ? '행동 목록 JSON 가져오기' : 'Import behavior library JSON'},
    {value: 'none', label: ko ? '등록 행동 없이 사용' : 'Use no registered behaviors'},
  ], 'keep');
  if (choice === 'none') profile.behaviors = {};
  if (choice === 'import') {
    const path = await prompts.text(ko ? '행동 목록 JSON 경로' : 'Behavior library JSON path');
    const normalized = await core.call<{library: string}>('runtime.paths', {paths: {library: path}, base_dir: process.cwd()});
    profile.behaviors = await core.call('behaviors.import', {path: normalized.library});
    const observed = await core.call<{unavailable: Record<string, string>}>('behaviors.inspect', {avatar, behaviors: profile.behaviors});
    if (avatar.calibration && Object.entries(profile.behaviors).some(([key, item]) => item.source === 'clip' && key in observed.unavailable)
        && await prompts.confirm(ko ? '가져온 저장 동작을 모두 이 아바타에서 ARDY 추적을 켠 채 직접 확인했나요?'
          : 'Have you visually verified all imported clips on this avatar with ARDY tracking active?', false)) {
      profile.behaviors = await core.call('behaviors.attest', {avatar, behaviors: profile.behaviors, tracking_active: true});
    }
  }
  profile.autonomy.behaviors = profile.autonomy.behaviors.filter(key => key in profile.behaviors);
  if (!Object.keys(profile.behaviors).length) {
    profile.autonomy.enabled = false;
    profile.autonomy.behaviors = [];
  } else {
    profile.autonomy.enabled = await prompts.confirm(ko ? '조용히 대기할 때 자율 행동을 실행할까요?' : 'Run autonomous behaviors while quiet?', profile.autonomy.enabled);
    if (profile.autonomy.enabled) {
      const key = await prompts.select(ko ? '자율 행동' : 'Autonomous behavior', Object.entries(profile.behaviors).map(([value, spec]) => ({value, label: spec.description ?? value})), profile.autonomy.behaviors[0]);
      profile.autonomy.behaviors = [key];
      if (setup.advanced) profile.autonomy.interval_seconds = Number(await prompts.text(ko ? '연속 대기 시간 (초)' : 'Continuous quiet interval (seconds)', String(profile.autonomy.interval_seconds)));
    }
  }
}
