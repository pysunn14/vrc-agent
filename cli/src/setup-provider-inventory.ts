import type { CoreClient } from './core.js';
import type { Capability, Connection, Language } from './types.js';
import type { Prompter } from './prompts.js';
import { configureCredentials } from './setup-credentials.js';
import { SetupCanceled } from './prompts.js';

export interface Inventory {
  state: string; detail: string; items: {id: string; label: string}[];
  next_cursor: string | null; credential_env?: string; source?: string;
}

/** Only explicit user action permits an unverified ID after discovery fails. */
export async function inventoryChoice(core: CoreClient, prompts: Prompter, connection: Connection,
  capability: Capability, kind: 'models' | 'voices', language: Language, current = '', edit = false,
  repair: (reason: string) => Promise<void> = async () => {}, checkpoint: () => Promise<unknown> = async () => {}): Promise<string> {
  const ko = language === 'ko', label = kind === 'models' ? (ko ? '모델' : 'Model') : (ko ? '목소리' : 'Voice');
  let cursor: string | undefined;
  const seen = new Set<string>();
  while (true) {
    prompts.note(ko ? `${connection.label}에서 ${label} 목록을 확인합니다.` : `Checking ${label.toLowerCase()} inventory from ${connection.label}.`);
    const result = await core.call<Inventory>('connection.discover', {provider: connection.provider,
      config: connection.config, capability, kind, ...(cursor ? {cursor} : {})});
    if (result.state === 'needs-auth' && prompts.password && await configureCredentials(core, prompts, connection, language, true, checkpoint)) continue;
    if (result.source === 'preset') prompts.note(ko ? '프리셋이 제공하는 선택지입니다. 선택한 모델과의 호환성은 시험으로 확인합니다.' : 'Preset choices; test compatibility with the selected model.');
    if (result.state === 'ready') {
      if (!edit && current && result.items.some(item => item.id === current)) {
        prompts.note(`${label}: ${current}`); return current;
      }
      if (!current && !result.next_cursor && !cursor && result.items.length === 1) {
        prompts.note((ko ? '자동 선택: ' : 'Selected automatically: ') + `${label} · ${result.items[0].label}`);
        return result.items[0].id;
      }
    } else {
      const reasons: Record<string, string> = {
        'needs-auth': '연결에 사용할 인증이 필요합니다', 'credential-store-error': '운영체제 자격 증명 저장소를 확인해 주세요', 'auth-error': '서버가 인증을 거절했습니다',
        'network-error': '서버에 연결할 수 없습니다', 'http-error': '서버에서 오류를 반환했습니다',
        'invalid-response': '목록 응답을 해석할 수 없습니다', 'empty': '사용 가능한 목록이 비어 있습니다',
        'unsupported': '이 연결에는 목록 조회가 제공되지 않습니다. 저장된 값이나 ID를 사용할 수 있습니다',
      };
      prompts.note(ko && ['needs-auth', 'auth-error', 'credential-store-error'].includes(result.state) ? reasons[result.state] : `${ko ? reasons[result.state] ?? result.state : result.state}\n${result.detail}`);
    }
    const more = result.next_cursor && !seen.has(result.next_cursor) && result.next_cursor !== cursor;
    if (result.next_cursor && !more) prompts.note(ko ? '서버가 같은 페이지를 반복했습니다. 다음 페이지 조회를 중단합니다.' : 'Server repeated a page token; pagination stopped.');
    const options = [
      ...result.items.map((item, i) => ({value: `item:${i}`, label: item.label === item.id ? item.id : `${item.label} · ${item.id}`})),
      ...(current ? [{value: '@keep', label: ko ? `기존 값 유지: ${current} (목록 확인과 별개)` : `Keep saved value: ${current} (independent of inventory)`}] : []),
      ...(more ? [{value: '@more', label: ko ? '다음 페이지' : 'Next page'}] : []),
      ...(prompts.password ? [{value: '@auth', label: ko ? 'API 키 등록·변경' : 'Register / replace API key'}] : []),
      {value: '@retry', label: ko ? '다시 조회' : 'Retry'},
      {value: '@repair', label: ko ? '주소·인증 수정' : 'Edit address / credentials'},
      {value: '@manual', label: ko ? 'ID 직접 입력 (추론 검증 전)' : 'Enter an ID (inference unverified)'},
      {value: '@pause', label: ko ? '초안 저장 후 나가기' : 'Save draft and exit'},
    ];
    const select = prompts.search?.bind(prompts) ?? prompts.select.bind(prompts);
    const choice = await select(`${label} · ${ko ? '목록 조회는 추론 검증과 별개입니다' : 'Inventory does not verify inference'}`, options,
      current ? '@keep' : result.items.length ? 'item:0' : result.state === 'needs-auth' || result.state === 'auth-error' ? (prompts.password ? '@auth' : '@repair') : result.state === 'unsupported' ? '@manual' : '@retry');
    if (choice.startsWith('item:')) return result.items[Number(choice.slice(5))].id;
    if (choice === '@keep') return current;
    if (choice === '@manual') return prompts.text(`${label} ID`, current);
    if (choice === '@pause') throw new SetupCanceled();
    if (choice === '@auth') await configureCredentials(core, prompts, connection, language, false, checkpoint);
    if (choice === '@repair') await repair(result.state);
    if (choice === '@more') { if (cursor) seen.add(cursor); cursor = result.next_cursor!; }
    else { cursor = undefined; seen.clear(); }
  }
}
