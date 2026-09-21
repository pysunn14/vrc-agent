import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { CoreClient } from '../src/core.js';
import { blankProfile, onboard } from '../src/onboard.js';
import { SetupCanceled, type Prompter } from '../src/prompts.js';
import { t } from '../src/i18n.js';
import type { AgentProfile, Bootstrap, Language, OS, Profile, Role, Saved } from '../src/types.js';

const unavailable = {state: 'unavailable', detail: 'not installed', peers: [], self: null};
const python = resolve(process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');

async function roleStep(os: OS, language: Language, role: Role = 'agent') {
  const state: Bootstrap = {protocol: 3, catalog: {}, profile: null, revision: null, draft: null,
    path: '', os, python: 'python', cwd: '.'};
  const drafts: Profile[] = [];
  const core = new CoreClient();
  core.call = async <T>(action: string, payload: Record<string, unknown> = {}): Promise<T> => {
    if (action === 'hosts.discover') return unavailable as T;
    if (action !== 'draft.save') throw new SetupCanceled();
    drafts.push(structuredClone(payload.profile as Profile));
    return {saved: true} as T;
  };
  const events: {kind: string; message: string; values?: string[]; initial?: string}[] = [];
  const ui: Prompter = {
    select: async (message, options, initial) => {
      events.push({kind: 'select', message, values: options.map(option => option.value), initial});
      if (options.some(option => option.value === 'en')) return options.find(option => option.value === language)!.value;
      if (options.some(option => option.value === 'combined')) return options.find(option => option.value === role)!.value;
      if (options.some(option => option.value === 'manual')) return options.find(option => option.value === 'manual')!.value;
      if (options.some(option => option.value === 'linux')) return options.find(option => option.value === 'linux')!.value;
      throw new SetupCanceled();
    },
    text: async (message, initial) => {
      events.push({kind: 'text', message, initial});
      if (message === t(language, 'gameAddress')) return 'game-host';
      if (message === t(language, 'runnerAddress')) return 'agent-host';
      if (role === 'bridge') return message.includes('Windows') ? 'game-host' : 'agent-host';
      throw new Error(`Unexpected address question: ${message}`);
    },
    confirm: async (_message, initial) => {if (initial === false) throw new SetupCanceled(); return true;},
    note: message => { events.push({kind: 'note', message}); },
  };
  assert.equal(await onboard(core, state, ui), null);
  assert.ok(drafts.length);
  return {events, profile: drafts.at(-1)!};
}

for (const os of ['macos', 'linux'] as const) for (const language of ['en', 'ko'] as const) {
  test(`${os} explains its detected role without asking OS or offering a single choice (${language})`, async () => {
    const {events, profile} = await roleStep(os, language);
    assert.deepEqual(events[0].values, ['en', 'ko']);
    assert.equal(events[1].kind, 'note');
    assert.ok(events[1].message.includes(os === 'macos' ? 'Mac' : 'Linux'));
    assert.ok(events[1].message.includes('Windows'));
    assert.deepEqual(events.filter(event => event.kind === 'text').map(event => event.message),
      [t(language, 'gameAddress'), t(language, 'runnerAddress')]);
    assert.equal(profile.language, language);
    assert.equal(profile.role, 'agent');
    assert.deepEqual(profile.hosts, {runner: {os, address: 'agent-host'}, game: {os: 'windows', address: 'game-host'}});
  });
}

for (const role of ['combined', 'agent', 'bridge'] as const) {
  test(`Windows offers three roles and assigns local ownership for ${role}`, async () => {
    const {events, profile} = await roleStep('windows', 'ko', role);
    assert.deepEqual(events[1].values, ['combined', 'agent', 'bridge']);
    assert.equal(events[1].initial, 'combined');
    assert.equal(profile.role, role);
    assert.equal(profile.game_host, role === 'combined' ? 'runner' : 'game');
    if (role === 'combined') {
      assert.equal(profile.hosts.runner.address, '127.0.0.1');
      assert.equal(events.filter(event => event.kind === 'text').length, 0);
    }
    if (role === 'bridge') {
      assert.deepEqual(profile.hosts, {runner: {os: 'linux', address: 'agent-host'}, game: {os: 'windows', address: 'game-host'}});
    }
  });
}

function seeded(state: Bootstrap): AgentProfile {
  const profile = blankProfile(state, 'en');
  profile.hosts.runner.address = 'agent-host';
  profile.hosts.game = {os: 'windows', address: 'game-host'};
  profile.providers.api = {provider: 'openai', label: 'API', config: {api_key_env: 'EXAMPLE_KEY'}, deployment: {kind: 'api'}};
  for (const capability of ['llm', 'stt', 'tts'] as const) profile.bindings[capability] = {instance: 'api', model: `example-${capability}`};
  profile.bindings.tts.voice = 'voice';
  profile.avatar.rig = 'shinano';
  return profile;
}

test('cancel preserves active config and a draft can be reopened and committed', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vrc-agent-setup-'));
  try {
    const core = new CoreClient({config: join(directory, 'agent.json'), python});
    const call = core.call.bind(core);
    core.call = async <T>(action: string, payload?: Record<string, unknown>) => action === 'hosts.discover' ? unavailable as T : call<T>(action, payload);
    const initial = await core.call<Bootstrap>('bootstrap');
    const saved = await core.call<Saved>('profile.commit', {profile: seeded(initial), revision: null});
    let calls = 0;
    const canceling: Prompter = {
      select: async (_message, options, initial) => { if (++calls === 1) throw new SetupCanceled(); return initial ?? options[0].value; },
      text: async (_message, initial) => initial ?? '', confirm: async () => true, note: () => {},
    };
    assert.equal(await onboard(core, await core.call<Bootstrap>('bootstrap'), canceling), null);
    const paused = await core.call<Bootstrap>('bootstrap');
    assert.equal(paused.revision, saved.revision);
    assert.ok(paused.draft);
    const accepting: Prompter = {
      select: async (_message, options, initial) => initial && options.some(o => o.value === initial) ? initial : options[0].value,
      text: async (_message, initial) => initial ?? '', confirm: async (_message, initial) => initial ?? true, note: () => {},
    };
    const resumed = await onboard(core, paused, accepting);
    assert.ok(resumed);
    assert.ok(resumed.profile.role !== 'bridge');
    assert.equal(resumed.profile.bindings.tts.voice, 'voice');
    assert.equal((await core.call<Bootstrap>('bootstrap')).draft, null);
  } finally { await rm(directory, {recursive: true, force: true}); }
});

test('connection-only Windows setup commits through the core without ARDY or provider calls', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vrc-bridge-setup-'));
  try {
    const core = new CoreClient({config: join(directory, 'bridge.json'), python});
    const call = core.call.bind(core), calls: string[] = [];
    core.call = async <T>(action: string, payload?: Record<string, unknown>) => {
      calls.push(action);
      if (action === 'hosts.discover') return {state: 'ready', self: {address: '100.64.0.2'}, peers: [{id: 'runner', name: 'runner', address: '100.64.0.1', os: 'linux', online: true}]} as T;
      if (action === 'runtime.discover') return {python: 'C:/Agent/.venv/Scripts/python.exe'} as T;
      if (action === 'bridge.discover') return {platform: 'Windows', packages: {}, audio_devices: [], process_audio_helper: {exists: true}} as T;
      if (action === 'bridge.paths') return {python: 'C:/Agent/.venv/Scripts/python.exe', project_dir: 'C:/Agent'} as T;
      assert.ok(!['runtime.options', 'runtime.setup', 'runtime.verify', 'connection.validate'].includes(action), action);
      return call<T>(action, payload);
    };
    const state = {...await core.call<Bootstrap>('bootstrap'), os: 'windows' as const, python: 'C:/Agent/.venv/Scripts/python.exe', cwd: 'C:/Agent'};
    const ui: Prompter = {
      select: async (_message, options, initial) => options.some(o => o.value === 'combined') ? options.find(o => o.value === 'bridge')!.value
        : options.some(o => o.value === 'peer:runner') ? options.find(o => o.value === 'peer:runner')!.value : initial ?? options[0].value,
      text: async (_message, initial) => initial ?? '', confirm: async (_message, initial) => initial ?? true, note: () => {},
    };
    const saved = await onboard(core, state, ui);
    assert.equal(saved?.profile.role, 'bridge');
    const status = await call<{id: string; local: boolean}[]>('status');
    assert.deepEqual(status.filter(row => row.local).map(row => row.id), ['bridge']);
    assert.equal((await core.call<Bootstrap>('bootstrap')).draft, null);
  } finally { await rm(directory, {recursive: true, force: true}); }
});

test('role changes retain agent settings between agent and combined, and separate local geometry', async () => {
  const {profileForRole} = await import('../src/setup-profile.js');
  const state = {os: 'windows', python: 'python.exe', cwd: 'C:/Agent'} as unknown as Bootstrap;
  const agent = seeded(state);
  const combined = profileForRole(state, 'ko', 'combined', agent);
  assert.ok(combined.role === 'combined');
  assert.deepEqual(combined.bindings, agent.bindings);
  const bridge = profileForRole(state, 'ko', 'bridge', combined);
  assert.ok(bridge.role === 'bridge');
  bridge.bridge.avatar_rig = 'custom.json';
  const restored = profileForRole(state, 'ko', 'combined', bridge);
  assert.ok(restored.role === 'combined');
  // The combined role uses its agent avatar as the only geometry source.
  restored.avatar.rig = 'agent-rig';
  assert.equal(restored.avatar.rig, 'agent-rig');
});
