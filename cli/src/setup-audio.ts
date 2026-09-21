import type { Prompter } from './prompts.js';
import type { Language } from './types.js';

export interface AudioDevice {index: number; name: string; host_api?: string; max_input_channels: number; max_output_channels: number}
export interface BridgeInventory {platform: string; packages: Record<string, string | null>;
  audio_devices: AudioDevice[]; audio_error?: string; process_audio_helper?: {path: string; exists: boolean}; helper_error?: string}

export async function selectAudioDevice(prompts: Prompter, devices: AudioDevice[], direction: 'input'|'output', language: Language,
  current = '', edit = false): Promise<string> {
  const ko = language === 'ko', channel = direction === 'output' ? 'max_output_channels' : 'max_input_channels';
  const eligible = devices.filter(d => d[channel] > 0);
  const matching = eligible.filter(d => /^\d+$/.test(current) ? String(d.index) === current : current && d.name.toLowerCase().includes(current.toLowerCase()));
  if (!edit && matching.length === 1 && !/^\d+$/.test(current)) {prompts.note(`${current}`); return current;}
  const title = direction === 'output' ? (ko ? 'VRChat 가상 마이크로 보낼 출력 장치' : 'Output device routed to the VRChat virtual microphone')
    : (ko ? '음성을 받을 마이크' : 'Microphone input');
  if (!eligible.length) prompts.note(ko ? '사용 가능한 음성 장치를 찾지 못했습니다. 장치·드라이버·Python 환경을 확인하세요.' : 'No audio devices found. Check devices, drivers and the Python environment.');
  const select = prompts.search?.bind(prompts) ?? prompts.select.bind(prompts);
  const selected = await select(title, [
    ...eligible.map(d => ({value: `device:${d.index}`, label: `${d.name}${d.host_api ? ' · ' + d.host_api : ''}`})),
    {value: '@manual', label: ko ? '장치 이름 직접 입력' : 'Enter device name'},
    {value: '@later', label: ko ? '나중에 연결' : 'Connect later'},
  ], matching.length === 1 ? `device:${matching[0].index}` : eligible.length ? `device:${eligible[0].index}` : '@later');
  if (selected === '@later') return '';
  if (selected === '@manual') {
    prompts.note(ko ? '직접 입력한 장치는 doctor에서 연결 상태를 확인하세요.' : 'Check the entered device with doctor.');
    return prompts.text(title, current);
  }
  const device = eligible.find(d => `device:${d.index}` === selected)!;
  // PortAudio matches substrings. Use a name only when it resolves uniquely;
  // duplicate host API entries require an explicit index and later rechecking.
  if (eligible.filter(d => d.name.toLowerCase().includes(device.name.toLowerCase())).length === 1) return device.name;
  prompts.note(ko ? `장치 이름이 중복되어 번호 ${device.index}를 저장합니다. 장치 구성 변경 후 다시 확인하세요.`
    : `Duplicate device names: saving index ${device.index}. Recheck after changing audio devices.`);
  return String(device.index);
}
