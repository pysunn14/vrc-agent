import type { Profile } from './types.js';
import { hasAgent, hasBridge } from './types.js';
import type { Prompter } from './prompts.js';

export function sectionChoices(profile: Profile) {
  const ko = profile.language === 'ko';
  return [
    {value: 'hosts', label: ko ? '컴퓨터 연결' : 'Computers'},
    ...(hasAgent(profile) ? [
      {value: 'llm', label: ko ? '대화 모델' : 'Conversation model'},
      {value: 'stt', label: ko ? '음성 인식' : 'Transcription'},
      {value: 'tts', label: ko ? '음성 합성' : 'Speech synthesis'},
      {value: 'runtime', label: 'ARDY'}, {value: 'avatar', label: ko ? '아바타·행동' : 'Avatar / behaviors'},
    ] : []),
    ...(hasBridge(profile) ? [{value: 'bridge', label: ko ? 'Windows 음성·추적' : 'Windows audio / tracking'}] : []),
  ];
}

export function setupSummary(profile: Profile): string {
  const ko = profile.language === 'ko', ports = profile.connection;
  const lines = [`${profile.role} · ${profile.language}`,
    `${profile.hosts[profile.runner_host]?.address ?? '?'} ↔ ${profile.hosts[profile.game_host]?.address ?? '?'}`,
    `${ko ? '포트' : 'Ports'}: ${ports.stream_port} / ${ports.agent_port} / ${ports.bridge_port}`];
  if (hasAgent(profile)) {
    for (const cap of ['llm', 'stt', 'tts'] as const) {
      const b = profile.bindings[cap], c = profile.providers[b.instance];
      lines.push(`${cap.toUpperCase()} · ${c?.label ?? '?'} · ${b.model || '?'}${b.voice ? ' / ' + b.voice : ''}`);
      if (c) {
        const source = c.config.credential_source === 'keyring' ? (ko ? '운영체제 자격 증명 저장소' : 'OS credential store')
          : c.config.credential_source === 'hermes' ? (ko ? '기존 로컬 Hermes 인증' : 'Existing local Hermes credentials')
          : c.config.api_key_env || (ko ? '인증 없음' : 'no credentials');
        lines.push(`  ${c.config.base_url ?? c.deployment.kind} · ${source}`);
      }
    }
    lines.push(`ARDY · ${profile.runtime.python} · ${profile.runtime.device}`);
    lines.push(`${ko ? '아바타' : 'Avatar'} · ${profile.avatar.rig || (ko ? '준비 전' : 'not prepared')}`);
  }
  if (hasBridge(profile)) lines.push(`${ko ? '음성 출력' : 'Audio output'} · ${profile.bridge.virtual_mic_device || '?'}`);
  lines.push(ko ? '설정 요약입니다. 실제 추론·추적·연결 상태는 doctor와 providers test로 확인하세요.'
    : 'Configuration summary. Use doctor and providers test to verify inference, tracking and connections.');
  return lines.join('\n');
}

export async function advancedSetup(profile: Profile, prompts: Prompter,
  editSection: (section: string, advanced: boolean) => Promise<void>) {
  const ko = profile.language === 'ko';
  const section = await prompts.select(ko ? '고급 설정 항목' : 'Advanced settings', [
    {value: 'ports', label: ko ? '연결 포트' : 'Connection ports'},
    ...sectionChoices(profile).filter(c => c.value !== 'hosts'),
    {value: 'back', label: ko ? '돌아가기' : 'Back'},
  ]);
  if (section === 'back') return;
  if (section === 'ports') {
    for (const key of ['stream_port', 'agent_port', 'bridge_port'] as const)
      profile.connection[key] = Number(await prompts.text(key, String(profile.connection[key])));
  } else await editSection(section, true);
}
