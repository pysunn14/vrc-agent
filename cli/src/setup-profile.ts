import type { AgentOnlyProfile, Bootstrap, BridgeSettings, Language, Profile, Role } from './types.js';
import { hasAgent, hasBridge } from './types.js';

export function blankProfile(state: Bootstrap, language: Language): AgentOnlyProfile {
  return {version: 3, role: 'agent', language,
    hosts: {runner: {os: state.os, address: '127.0.0.1'}}, runner_host: 'runner', game_host: 'game',
    connection: {stream_port: 8766, agent_port: 8765, bridge_port: 8767}, providers: {},
    bindings: {llm: {instance: '', model: ''}, stt: {instance: '', model: ''}, tts: {instance: '', model: ''}},
    runtime: {python: state.python, project_dir: state.cwd, device: 'auto'},
    avatar: {rig: '', base_pose: '', hmd_base: [0, 1, 0], face_channels: {}},
    behaviors: {}, autonomy: {enabled: false, interval_seconds: 30, behaviors: []}};
}

export function profileForRole(state: Bootstrap, language: Language, role: Role, previous?: Profile): Profile {
  const defaults = blankProfile(state, language);
  const base = {version: 3 as const, language, role,
    hosts: structuredClone(previous?.hosts ?? defaults.hosts),
    runner_host: previous?.runner_host ?? defaults.runner_host, game_host: previous?.game_host ?? defaults.game_host,
    connection: {...defaults.connection, ...previous?.connection}};
  const bridge: BridgeSettings = {python: state.python, project_dir: state.cwd, virtual_mic_device: '',
    ...(previous && hasBridge(previous) ? previous.bridge : {})};
  if (role === 'bridge') return {...base, role, bridge: {...bridge,
    avatar_rig: previous?.role === 'bridge' ? previous.bridge.avatar_rig : '',
    hmd_base: previous?.role === 'bridge' ? previous.bridge.hmd_base : [0, 1, 0]}};
  const agent = previous && hasAgent(previous) ? previous : defaults;
  const settings = {providers: agent.providers ?? {}, bindings: {...defaults.bindings, ...agent.bindings},
    runtime: {...defaults.runtime, ...agent.runtime}, avatar: {...defaults.avatar, ...agent.avatar},
    behaviors: agent.behaviors ?? {}, autonomy: {...defaults.autonomy, ...agent.autonomy}};
  // Geometry is owned by the agent in combined mode; never keep a second,
  // potentially divergent copy from a former connection-only profile.
  const {avatar_rig: _rig, hmd_base: _origin, ...localBridge} = bridge as BridgeSettings & {avatar_rig?: string; hmd_base?: number[]};
  return role === 'combined' ? {...base, ...settings, role, bridge: localBridge} : {...base, ...settings, role};
}
