import type { CoreClient } from './core.js';
import type { Bootstrap, HostDiscovery, OS, Profile } from './types.js';
import type { Prompter } from './prompts.js';
import { t } from './i18n.js';

export async function setupHosts(core: CoreClient, state: Bootstrap, profile: Profile, prompts: Prompter, options: {edit?: boolean} = {}): Promise<void> {
  const ko = profile.language === 'ko';
  if (profile.role === 'combined') {
    profile.game_host = profile.runner_host;
    profile.hosts[profile.runner_host] = {os: state.os, address: '127.0.0.1'};
    return;
  }
  if (profile.runner_host === profile.game_host) {
    let candidate = 'game', suffix = 2;
    while (candidate in profile.hosts) candidate = `game-${suffix++}`;
    profile.game_host = candidate;
  }
  const localID = profile.role === 'agent' ? profile.runner_host : profile.game_host;
  const peerID = profile.role === 'agent' ? profile.game_host : profile.runner_host;
  const savedLocal = profile.hosts[localID];
  const savedPeer = profile.hosts[peerID];
  const usable = (value?: string) => value && !['127.0.0.1', 'localhost', '::1'].includes(value) ? value : '';
  const localAddress = usable(savedLocal?.address), peerAddress = usable(savedPeer?.address);
  const canKeep = !!localAddress && !!peerAddress && savedLocal?.os === state.os;
  if (canKeep && !options.edit) {
    prompts.note(ko ? `기존 연결 주소를 재사용합니다: ${localAddress} ↔ ${peerAddress}` : `Reusing connection addresses: ${localAddress} ↔ ${peerAddress}`);
    return;
  }
  while (true) {
    const discovery = await core.call<HostDiscovery>('hosts.discover');
    const peers = discovery.peers.filter(peer => profile.role === 'agent' ? peer.os === 'windows'
      : ['macos', 'linux', 'windows'].includes(peer.os));
    if (discovery.state !== 'ready') prompts.note((ko ? 'Tailscale 자동 검색을 사용할 수 없습니다: ' : 'Tailscale discovery unavailable: ') + discovery.detail);
    else if (!peers.length) prompts.note(ko ? '연결할 장비를 찾지 못했습니다. 같은 Tailscale 네트워크에 로그인했는지 확인하세요.'
      : 'No matching computer found. Check that both computers use the same Tailscale network.');
    const choices = [
      ...(canKeep ? [{value: 'keep', label: ko ? `기존 연결 유지 (${peerAddress})` : `Keep connection (${peerAddress})`}] : []),
      ...peers.map(peer => ({value: `peer:${peer.id}`, label: `${peer.name} · ${peer.address} · ${peer.online ? (ko ? '온라인' : 'online') : (ko ? '오프라인' : 'offline')}`})),
      {value: 'manual', label: ko ? '주소 직접 입력 (내부망 또는 Tailscale)' : 'Enter addresses (LAN or Tailscale)'},
      {value: 'retry', label: ko ? '다시 검색' : 'Search again'},
    ];
    const selection = await prompts.select(profile.role === 'agent' ? (ko ? '연결할 Windows 컴퓨터' : 'Windows computer to connect')
      : (ko ? '에이전트를 실행하는 컴퓨터' : 'Computer running the agent'), choices, canKeep ? 'keep' : peers.length ? choices[0].value : 'manual');
    if (selection === 'retry') continue;
    if (selection === 'keep') return;
    if (selection.startsWith('peer:')) {
      const peer = peers.find(peer => `peer:${peer.id}` === selection);
      if (!peer || !discovery.self) throw new Error('Selected host is no longer in the discovery result; search again');
      profile.hosts[localID] = {os: state.os, address: discovery.self.address};
      profile.hosts[peerID] = {os: peer.os as OS, address: peer.address};
      prompts.note(ko ? `연결 주소를 설정했습니다: ${discovery.self.address} ↔ ${peer.address}\n실제 브리지 연결 상태는 doctor에서 확인합니다.`
        : `Connection addresses: ${discovery.self.address} ↔ ${peer.address}\nUse doctor to verify the bridge connection.`);
      return;
    }
    const remoteAddress = await prompts.text(profile.role === 'agent' ? t(profile.language, 'gameAddress')
      : (ko ? '에이전트 컴퓨터 주소 (내부망 또는 Tailscale)' : 'Agent computer address (LAN or Tailscale)'), peerAddress);
    const remoteOS = profile.role === 'agent' ? 'windows' : await prompts.select(t(profile.language, 'remoteOS'),
      [{value: 'macos' as const, label: 'macOS'}, {value: 'linux' as const, label: 'Linux'}, {value: 'windows' as const, label: 'Windows'}], savedPeer?.os);
    const ownAddress = await prompts.text(profile.role === 'agent' ? t(profile.language, 'runnerAddress')
      : (ko ? '에이전트에서 접속할 이 Windows 컴퓨터 주소' : 'This Windows computer’s address reachable from the agent'), localAddress);
    profile.hosts[peerID] = {os: remoteOS, address: remoteAddress};
    profile.hosts[localID] = {os: state.os, address: ownAddress};
    return;
  }
}
