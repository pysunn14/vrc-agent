import type { CoreClient } from './core.js';
import type { Connection, Language } from './types.js';
import { SetupCanceled, type Prompter } from './prompts.js';

export async function configureCredentials(core: CoreClient, prompts: Prompter, connection: Connection,
  language: Language, discover: boolean, checkpoint: () => Promise<unknown>): Promise<boolean> {
  const ko = language === 'ko';
  if (discover) {
    const prepared = await core.call<{state: string; config: Connection['config']; source?: string}>('connection.auth.prepare', {
      provider: connection.provider, config: connection.config});
    if (prepared.state === 'ready') {
      connection.config = prepared.config;
      await checkpoint();
      prompts.note(ko ? '이 컴퓨터의 기존 Hermes 인증을 연결했습니다. 키를 복사할 필요가 없습니다.'
        : 'Connected the existing local Hermes credentials. No key copying is needed.');
      return true;
    }
  }
  if (!prompts.password) return false;
  while (true) {
    prompts.note(ko ? 'API 키를 입력하면 운영체제 자격 증명 저장소에 보관합니다. 설정 파일에는 키 대신 참조만 저장합니다.'
      : 'Your API key is stored in the OS credential store. Configuration contains a reference, not the key.');
    const key = await prompts.password(ko ? `${connection.label} API 키` : `${connection.label} API key`);
    let saved: {config: Connection['config']};
    try {
      saved = await core.call<{config: Connection['config']}>('connection.auth.save', {
        provider: connection.provider, config: connection.config, key});
    } catch (error) {
      prompts.note(String(error));
      const choice = await prompts.select(ko ? '인증을 저장하지 못했습니다' : 'Could not store credentials', [
        {value: 'retry', label: ko ? '저장소 확인 후 다시 입력' : 'Check credential storage and try again'},
        {value: 'environment', label: ko ? '환경변수 방식 사용' : 'Use an environment variable'},
        {value: 'pause', label: ko ? '초안 저장 후 나가기' : 'Save draft and exit'},
      ]);
      if (choice === 'pause') throw new SetupCanceled();
      if (choice === 'environment') {
        connection.config.credential_source = 'environment'; connection.config.credential_ref = '';
        await checkpoint(); return false;
      }
      continue;
    }
    connection.config = saved.config;
    await checkpoint();
    prompts.note(ko ? '인증을 저장했습니다. 연결을 다시 확인합니다.' : 'Credentials saved. Checking the connection again.');
    return true;
  }
}
