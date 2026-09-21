import type { AgentProfile, Capability } from './types.js';
import { CoreCanceled, type CoreClient } from './core.js';
import { SetupCanceled } from './prompts.js';
import type { Prompter } from './prompts.js';

interface ProbeResult {state: string; reason?: string; recorded?: boolean; fingerprint?: string}
export function connectionTestPayload(profile: AgentProfile, capability: Capability) {
  const {instance, ...settings} = profile.bindings[capability];
  const connection = profile.providers[instance];
  return {provider: connection.provider, config: connection.config, capability, settings,
    runtime: {python: profile.runtime.python, project_dir: profile.runtime.project_dir}};
}

export async function testConnections(core: CoreClient, profile: AgentProfile, prompts: Prompter): Promise<boolean> {
  const ko = profile.language === 'ko';
  prompts.note(ko ? '실제 API 요청을 보냅니다. 음성은 재생하지 않고, 마이크와 VRChat은 사용하지 않습니다. 화면 입력은 합성 이미지로 시험합니다.'
    : 'Sends real API requests. No audio playback, microphone or VRChat. Vision uses a synthetic image.');
  let success = true;
  for (const capability of ['llm', 'stt', 'tts'] as const) {
    const payload: Record<string, unknown> = connectionTestPayload(profile, capability);
    const identity = JSON.stringify(payload);
    if (capability === 'tts') payload.text = ko ? '안녕하세요. 음성 연결을 확인합니다.' : 'This is a voice connection test.';
    if (capability === 'stt') {
      const language = String(profile.bindings.stt.language ?? '');
      if (language && language !== 'en') {
        const file = await (prompts.path?.bind(prompts) ?? prompts.text.bind(prompts))(
          ko ? `${language} 시험 음성 파일 (16 kHz 모노 WAV)` : `${language} test audio file (16 kHz mono WAV)`, '', false);
        if (!file) {prompts.note(ko ? 'STT 시험을 건너뛰었습니다.' : 'STT test skipped.'); success = false; continue;}
        payload.audio_file = file;
      } else prompts.note(ko ? 'STT: 포함된 영어 시험 음성을 사용합니다. 인식 품질 평가는 아닙니다.' : 'STT uses the bundled English sample; this is not a quality evaluation.');
    }
    prompts.note(ko ? `${capability.toUpperCase()} 연결 시험 중…` : `Testing ${capability.toUpperCase()} connection…`);
    let result: ProbeResult;
    try {result = await core.call<ProbeResult>('connection.test', payload);}
    catch (error) {
      if (error instanceof CoreCanceled) throw new SetupCanceled(ko ? '연결 시험을 취소했습니다.' : 'Connection test canceled.');
      result = {state: 'failed', reason: 'core-error'};
    }
    if (identity !== JSON.stringify(connectionTestPayload(profile, capability)) || result.recorded === false) {
      prompts.note(ko ? '설정 또는 시험이 변경되어 이전 결과를 적용하지 않습니다.' : 'Settings or test changed; ignoring the superseded result.');
      success = false; continue;
    }
    if (result.state === 'success') prompts.note(ko ? `${capability.toUpperCase()}: 시험 응답 정상` : `${capability.toUpperCase()}: test response received`);
    else {
      const reasons: Record<string,string> = {'auth-error':'인증 거절', 'needs-auth':'인증 필요', 'network-error':'서버 연결 실패',
        'sample-language-mismatch':'설정한 언어의 시험 음성 파일 필요', 'sample-unavailable':'시험 음성 파일 없음', 'runtime-unavailable':'로컬 실행 환경 준비 필요', 'runtime-error':'로컬 실행 환경 오류',
        'timeout':'시험 시간 초과', 'invalid-response':'요청·응답 규격 확인 필요', 'http-error':'서버 오류', 'core-error':'코어 실행 실패'};
      prompts.note(`${capability.toUpperCase()}: ${ko ? reasons[result.reason ?? ''] ?? result.reason : result.reason}`);
      success = false;
    }
  }
  return success;
}

export async function previousTests(core: CoreClient, profile: AgentProfile): Promise<string[]> {
  const ko = profile.language === 'ko', lines: string[] = [];
  for (const capability of ['llm', 'stt', 'tts'] as const) {
    const result = await core.call<{observation: {state: string; finished_at?: number; result?: ProbeResult} | null}>(
      'connection.observation', connectionTestPayload(profile, capability));
    const o = result.observation;
    const status = o?.state === 'finished' ? (o.result?.state === 'success' ? (ko ? '이전 시험 성공' : 'Previous test succeeded') : (ko ? '이전 시험 실패' : 'Previous test failed'))
      : o ? (ko ? '이전 시험 미완료' : 'Previous test incomplete') : (ko ? '미시험' : 'Untested');
    lines.push(`${capability.toUpperCase()}: ${status}${o?.finished_at ? ' · '+new Date(o.finished_at*1000).toLocaleString() : ''}`);
  }
  lines.push(ko ? '과거 시험 기록이며 현재 접속 상태를 보장하지 않습니다.' : 'Historical results do not establish current availability.');
  return lines;
}
