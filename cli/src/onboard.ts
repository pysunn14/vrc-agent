import * as clack from '@clack/prompts';
import type { CoreClient } from './core.js';
import type { Bootstrap, Profile, Saved, Role } from './types.js';
import { hasAgent, hasBridge } from './types.js';
import { setupAvatar } from './setup-avatar.js';
import { setupRuntime } from './setup-runtime.js';
import { setupHosts } from './setup-hosts.js';
import { setupBridge } from './setup-bridge.js';
import { blankProfile, profileForRole } from './setup-profile.js';
import { configureProvider, editProviderDetails } from './setup-providers.js';
import { advancedSetup, sectionChoices, setupSummary } from './setup-review.js';
import { prompter, SetupCanceled, type Prompter } from './prompts.js';
import { t } from './i18n.js';
import { previousTests, testConnections } from './setup-connection-tests.js';
export { blankProfile } from './setup-profile.js';

export async function onboard(core: CoreClient, state: Bootstrap, ui?: Prompter): Promise<Saved | null> {
  let language = core.options.language ?? state.profile?.language ?? state.draft?.profile.language ?? 'en';
  let profile: Profile = structuredClone(state.profile ?? blankProfile(state, language));
  let completed = new Set(state.profile ? ['language', 'role', ...sectionChoices(profile).map(c => c.value)] : []);
  const revision = state.revision;
  const identities = () => new Map(hasAgent(profile) ? Object.entries(profile.providers).map(([id, c]) =>
    [id, JSON.stringify([c.provider, c.config.base_url, c.config.api_key_env, c.config.credential_source, c.config.credential_ref, c.config.request_extension, c.config.voices_path])]) : []);
  let connections = identities();
  const checkpoint = () => {
    const current = identities();
    // Bindings share connections. Changing an endpoint/account invalidates all
    // completed selections using it, including edits interrupted by Escape.
    if (hasAgent(profile)) for (const cap of ['llm', 'stt', 'tts'] as const) {
      const id = profile.bindings[cap].instance;
      if (connections.has(id) && connections.get(id) !== current.get(id)) {
        completed.delete(cap);
      }
    }
    connections = current;
    return core.call('draft.save', {profile, revision, completed: [...completed]});
  };
  const base = ui ?? prompter(() => language);
  // Save before the next question, after the preceding answer has been applied.
  // SIGINT/Escape also reaches the final checkpoint below. No active profile is
  // written until the user reviews the result and chooses Save.
  const prompts: Prompter = {...base,
    ...(base.password ? {password: async (message: string) => {await checkpoint(); return base.password!(message);}} : {}),
    text: async (...args) => {await checkpoint(); return base.text(...args);},
    select: async (...args) => {await checkpoint(); return base.select(...args);},
    confirm: async (...args) => {await checkpoint(); return base.confirm(...args);},
    ...(base.path ? {path: async (...args: Parameters<NonNullable<Prompter['path']>>) => {await checkpoint(); return base.path!(...args);}} : {}),
    ...(base.search ? {search: async <T extends string>(message: string, choices: {value: T; label: string}[], initial?: T) => {await checkpoint(); return base.search!(message, choices, initial);}} : {}),
  };
  if (!ui) clack.intro(t(language, 'setup'));
  try {
    // Don't overwrite a resumable draft while asking whether to use it.
    if (state.draft && state.draft.revision === revision && await base.confirm(t(language, 'resume'))) {
      profile = structuredClone(state.draft.profile);
      completed = new Set(state.draft.completed ?? []);
      connections = identities();
      language = core.options.language ?? profile.language;
    }
    if (!completed.has('language') && !core.options.language)
      language = await prompts.select(t(language, 'language'), [{value: 'en', label: 'English'}, {value: 'ko', label: '한국어'}], language);
    profile.language = language;
    completed.add('language');
    let ko = language === 'ko';
    const chooseRole = async () => {
      const role: Role = state.os === 'windows' ? await prompts.select(ko ? '이 컴퓨터에서 무엇을 실행하나요?' : 'What runs on this computer?', [
        {value: 'combined', label: ko ? '에이전트와 VRChat 함께 실행' : 'Agent and VRChat together'},
        {value: 'agent', label: ko ? '에이전트만 실행 (다른 Windows의 VRChat에 연결)' : 'Agent only (VRChat on another Windows computer)'},
        {value: 'bridge', label: ko ? 'VRChat 연결만 담당 (에이전트는 다른 컴퓨터)' : 'VRChat connection only (agent on another computer)'},
      ], state.profile || state.draft ? profile.role : 'combined') : 'agent';
      if (profile.role !== role) completed = new Set(['language']);
      profile = profileForRole(state, language, role, profile);
      completed.add('role');
      if (state.os !== 'windows') prompts.note(t(language, state.os === 'macos' ? 'separateMac' : 'separateLinux'));
    };
    if (!completed.has('role') || (state.os !== 'windows' && profile.role !== 'agent')) await chooseRole();
    const runSection = async (section: string, advanced = false, edit = true) => {
      // An interrupted edit is incomplete even if the old section was finished.
      completed.delete(section);
      await checkpoint();
      if (section === 'hosts') await setupHosts(core, state, profile, prompts, {edit});
      if (hasAgent(profile)) {
        if (['llm', 'stt', 'tts'].includes(section)) {
          if (advanced) await editProviderDetails(core, prompts, state, profile, section as 'llm'|'stt'|'tts');
          else await configureProvider(core, prompts, state, profile, section as 'llm'|'stt'|'tts', {edit, checkpoint});
        }
        if (section === 'runtime') await setupRuntime(core, state, profile, prompts, {advanced});
        if (section === 'avatar') await setupAvatar(core, state, profile, prompts, {advanced});
      }
      if (section === 'bridge' && hasBridge(profile)) await setupBridge(core, state, profile, prompts, {advanced});
      completed.add(section);
      await checkpoint();
    };
    const finishMissing = async () => {
      for (const {value} of sectionChoices(profile)) if (!completed.has(value)) await runSection(value, false, false);
    };
    await finishMissing();
    while (true) {
      prompts.note(setupSummary(profile), 'VRC AGENT');
      if (hasAgent(profile)) prompts.note((await previousTests(core, profile)).join('\n'));
      const choice = await prompts.select(ko ? '설정을 확인하고 마칩니다' : 'Review your configuration', [
        {value: '@save', label: ko ? '이 설정으로 저장 (추가 시험 없음)' : 'Save configuration (no additional tests)'},
        ...(hasAgent(profile) ? [{value: '@test-save', label: ko ? '연결 시험 후 저장 (실제 API 요청)' : 'Test connections and save (real API requests)'}] : []),
        ...sectionChoices(profile),
        ...(state.os === 'windows' ? [{value: '@role', label: ko ? '이 컴퓨터의 역할 변경' : 'Change this computer’s role'}] : []),
        {value: '@language', label: ko ? '언어 변경' : 'Change language'},
        {value: '@advanced', label: ko ? '고급 설정' : 'Advanced settings'},
        {value: '@pause', label: ko ? '초안 저장 후 나가기' : 'Save draft and exit'},
      ], '@save');
      if (choice === '@pause') throw new SetupCanceled();
      if (choice === '@test-save' && hasAgent(profile) && !await testConnections(core, profile, prompts)) {
        prompts.note(ko ? '시험을 완료하지 못했습니다. 설정을 수정하거나 시험 없이 저장할 수 있습니다.' : 'Tests did not complete successfully. Edit settings or save without testing.');
        continue;
      }
      if (choice === '@save' || choice === '@test-save') {
        let normalized: Profile;
        try { normalized = await core.call<Profile>('profile.validate', {profile}); }
        catch (error) {prompts.note(String(error)); continue;}
        const saved = await core.call<Saved>('profile.commit', {profile: normalized, revision});
        if (profile.role === 'agent') prompts.note(ko ? 'Windows에서도 vrc-agent onboard를 실행하고, VRChat 연결만 담당을 선택하세요. connection info로 두 장비의 주소와 포트를 확인할 수 있습니다.'
          : 'Run vrc-agent onboard on Windows and select VRChat connection only. Use connection info to check addresses and ports.');
        if (!ui) clack.outro(t(language, 'saved'));
        return saved;
      }
      if (choice === '@advanced') await advancedSetup(profile, prompts, runSection);
      else if (choice === '@role') {await chooseRole(); await finishMissing();}
      else if (choice === '@language') {
        profile.language = language = await prompts.select('Language / 언어', [{value: 'en', label: 'English'}, {value: 'ko', label: '한국어'}], language);
        ko = language === 'ko';
      } else await runSection(choice);
      await checkpoint();
      await finishMissing();
    }
  } catch (error) {
    await checkpoint();
    if (error instanceof SetupCanceled) {
      if (error.message) prompts.note(error.message);
      if (!ui) clack.cancel(t(language, 'canceled'));
      return null;
    }
    throw error;
  }
}
