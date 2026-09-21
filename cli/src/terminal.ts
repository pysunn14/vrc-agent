import { Input, ProcessTerminal, Text, TuiMainScreen, matchesKey } from '@earendil-works/pi-tui';
import chalk from 'chalk';
import figlet from 'figlet';
import type { Language, HostDiscovery } from './types.js';
import type { Inventory } from './setup-provider-inventory.js';
import { progressText, type RuntimeJob } from './setup-runtime.js';
import { t } from './i18n.js';

export const commands = ['/status', '/hosts', '/connection', '/start', '/stop', '/doctor', '/providers', '/configure', '/onboard', '/settings', '/bridge', '/runtime', '/behaviors', '/snapshot', '/help', '/quit'];

export function banner(language: Language, columns = process.stdout.columns || 80): string {
  const art = columns >= 78 ? figlet.textSync('VRC AGENT', {font: 'Small'}) : 'VRC AGENT';
  return `${chalk.bold.cyan(art)}\n${chalk.dim(t(language, 'welcome'))}\n`;
}

/** One main-screen Pi input, yielding the terminal to Clack during questions.
 * No alternate screen: commands and results remain in normal scrollback.
 */
export function readCommand(language: Language, history: string[]): Promise<string | null> {
  return new Promise(resolve => {
    const terminal = new ProcessTerminal();
    const tui = new TuiMainScreen(terminal, true);
    const input = new Input();
    let cursor = history.length, done = false;
    const finish = (value: string | null) => {
      if (done) return;
      done = true;
      // Submission can arrive in the same input chunk as the entire command.
      // Let Pi render that value before stopping, so paste+Enter stays visible.
      tui.renderNow();
      tui.stop();
      process.stdout.write('\n');
      resolve(value);
    };
    tui.addChild(new Text(chalk.dim(t(language, 'hint')), 0, 0));
    tui.addChild(input);
    tui.setFocus(input);
    input.onSubmit = value => finish(value);
    tui.addInputListener(data => {
      if (matchesKey(data, 'ctrl+c') || (matchesKey(data, 'ctrl+d') && !input.getValue())) {
        finish(null); return {consume: true};
      }
      if (matchesKey(data, 'up') || matchesKey(data, 'down')) {
        cursor = Math.max(0, Math.min(history.length, cursor + (matchesKey(data, 'up') ? -1 : 1)));
        input.setValue(history[cursor] ?? ''); tui.requestRender(); return {consume: true};
      }
      if (matchesKey(data, 'tab')) {
        const candidates = commands.filter(command => command.startsWith(input.getValue()));
        if (candidates.length === 1) input.setValue(candidates[0] + ' ');
        tui.requestRender(); return {consume: true};
      }
      return undefined;
    });
    tui.start();
  });
}

export function render(result: unknown, language: Language): string {
  if (result && typeof result === 'object' && 'items' in result && 'next_cursor' in result) {
    const inventory = result as Inventory;
    return [inventory.state, inventory.detail, ...inventory.items.map(item => `${item.id}  ${item.label}`),
      ...(inventory.next_cursor ? [`${language === 'ko' ? '다음 페이지: 같은 명령에 추가' : 'Next page: add to this command'} --cursor '${inventory.next_cursor}'`] : [])].join('\n');
  }
  if (result && typeof result === 'object' && 'peers' in result && 'self' in result) {
    const discovery = result as HostDiscovery;
    return [`Tailscale · ${discovery.state}`, discovery.detail,
      ...(discovery.self ? [`${language === 'ko' ? '이 컴퓨터' : 'This computer'}: ${discovery.self.name} · ${discovery.self.address}`] : []),
      ...discovery.peers.map(peer => `${peer.name} · ${peer.os} · ${peer.address} · ${peer.online ? (language === 'ko' ? '온라인' : 'online') : (language === 'ko' ? '오프라인' : 'offline')}`),
    ].join('\n');
  }
  if (result && typeof result === 'object' && 'ports' in result && 'local_host' in result) {
    const info = result as {local_host: string; role: string; agent: {address: string}; game: {address: string}; ports: {stream_port: number; agent_port: number; bridge_port: number}};
    return `${info.role} · ${info.agent.address} ↔ ${info.game.address}\n${language === 'ko' ? '동작·음성 / 에이전트 상태 / Windows 상태' : 'Motion/audio / Agent status / Windows status'}: ${info.ports.stream_port} / ${info.ports.agent_port} / ${info.ports.bridge_port}`;
  }
  if (result && typeof result === 'object' && 'state' in result && 'plan' in result && 'id' in result) {
    const job = result as {id: string; state: string; step?: string; completed?: number; total?: number; error?: string; detail?: string};
    return `${chalk.bold('ARDY')} ${job.state} · ${job.id}\n${progressText(result as unknown as RuntimeJob, language)}\n${job.error ?? job.detail ?? ''}`;
  }
  if (result && typeof result === 'object' && 'state' in result && result.state === 'not-configured') return language === 'ko' ? 'ARDY 설치 기록이 없습니다.' : 'No ARDY setup recorded.';
  if (result && typeof result === 'object' && 'state' in result) return render([result], language);
  if (result && typeof result === 'object' && 'connections' in result && 'bindings' in result) {
    const value = result as {connections: Record<string, {label: string; provider: string; deployment: {kind: string}}>;
      bindings: Record<string, {instance: string; model: string; voice?: string}>};
    const bindings = Object.entries(value.bindings).map(([capability, binding]) =>
      `${chalk.bold(capability.toUpperCase().padEnd(5))} ${binding.instance} / ${binding.model}${binding.voice ? ` / ${binding.voice}` : ''}`);
    const connections = Object.entries(value.connections).map(([id, connection]) =>
      `${id}  ${connection.label} · ${connection.provider} · ${connection.deployment.kind}`);
    return [...bindings, '', ...connections].join('\n');
  }
  if (result && typeof result === 'object' && 'available' in result && 'unavailable' in result) {
    const value = result as {available: {name: string; source: string; description: string}[];
      unavailable: Record<string, string>; avatar: {ready: boolean; detail: string}};
    return [value.avatar.ready ? chalk.green(language === 'ko' ? '아바타 교정 기록 확인' : 'Avatar attestation verified') : chalk.yellow(value.avatar.detail),
      ...value.available.map(item => `${chalk.green(item.name)}  ${item.source} · ${item.description}`),
      ...Object.entries(value.unavailable).map(([key, error]) => `${chalk.yellow(key)}  ${error}`),
      ...(value.available.length ? [] : [language === 'ko' ? '사용 가능한 등록 행동이 없습니다.' : 'No registered behaviors are available.'])].join('\n');
  }
  if (Array.isArray(result)) return result.map(row => {
    if (row && typeof row === 'object' && 'check' in row) {
      const value = row as {service: string; check: string; result: string; detail?: string};
      const color = value.result === 'pass' ? chalk.green : value.result === 'error' ? chalk.red : chalk.yellow;
      const label = ['pass', 'error', 'warning', 'pending'].includes(value.result)
        ? t(language, value.result as 'pass' | 'error' | 'warning' | 'pending') : value.result;
      return `${color(label.padEnd(9))} ${value.service} · ${value.check}\n          ${value.detail ?? ''}`;
    }
    if (row && typeof row === 'object' && 'state' in row) {
      const value = row as {id: string; state: string; pid?: number; health: {detail?: string}; error?: string};
      return `${chalk.bold(value.id.padEnd(24))} ${value.state}${value.pid ? ` · PID ${value.pid}` : ''}\n  ${value.error || value.health.detail || ''}`;
    }
    return JSON.stringify(row);
  }).join('\n');
  if (result && typeof result === 'object' && 'output' in result) return String((result as {output: unknown}).output);
  if (result && typeof result === 'object' && 'powershell' in result) return String((result as {powershell: unknown}).powershell);
  return JSON.stringify(result, null, 2);
}
