import type { Bootstrap, Capability } from './types.js';
import type { Prompter } from './prompts.js';

/** Service names are presets; protocol and process ownership are not questions. */
export async function chooseConnectionPreset(state: Bootstrap, prompts: Prompter, capability: Capability, ko: boolean): Promise<string> {
  const entries = Object.entries(state.catalog).filter(([, d]) => d.operations[capability]);
  const categories = [
    {value: 'service', label: ko ? '서비스 선택' : 'Choose a service'},
    {value: 'server', label: ko ? '서버 주소로 연결' : 'Connect a server URL'},
    {value: 'local', label: ko ? '이 컴퓨터에서 직접 실행' : 'Run on this computer'},
  ].filter(c => entries.some(([, d]) => (d.setup?.category ?? 'service') === c.value));
  const category = categories.length === 1 ? categories[0].value
    : await prompts.select(ko ? `${capability.toUpperCase()} · 연결 추가` : `${capability.toUpperCase()} · Add connection`, categories);
  const choices = entries.filter(([, d]) => (d.setup?.category ?? 'service') === category)
    .map(([value, d]) => ({value, label: d.name}));
  if (category !== 'service' && choices.length === 1) return choices[0].value;
  return prompts.select(ko ? '사용할 서비스' : 'Choose a service', choices);
}
