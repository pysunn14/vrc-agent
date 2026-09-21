import test from 'node:test';
import assert from 'node:assert/strict';
import { blankProfile } from '../src/onboard.js';
import { setupHosts } from '../src/setup-hosts.js';
import type { Bootstrap } from '../src/types.js';
import type { Prompter } from '../src/prompts.js';
import type { CoreClient } from '../src/core.js';

const state = {os: 'macos', python: 'python', cwd: '.'} as Bootstrap;
const ready = {state: 'ready', detail: '', self: {id: 'self', name: 'agent', os: 'macos', address: '100.64.0.1', online: true},
  peers: [{id: 'game', name: 'game', os: 'windows', address: '100.64.0.2', online: false},
    {id: 'other', name: 'other', os: 'linux', address: '100.64.0.3', online: true}]};

test('selecting an offline Windows peer fills both Tailscale addresses without typing', async () => {
  const profile = blankProfile(state, 'ko');
  const labels: string[] = [], notes: string[] = [];
  const core = {call: async () => ready} as unknown as CoreClient;
  const ui: Prompter = {select: async (_message, options) => {
    labels.push(...options.map(option => option.label));
    assert.ok(!options.some(option => option.value === 'peer:other'));
    return options.find(option => option.value === 'peer:game')!.value;
  }, text: async () => {throw new Error('No address typing needed');}, confirm: async () => true, note: message => notes.push(message)};
  await setupHosts(core, state, profile, ui);
  assert.equal(profile.hosts.runner.address, '100.64.0.1');
  assert.equal(profile.hosts.game.address, '100.64.0.2');
  assert.ok(labels.some(label => label.includes('오프라인')));
  assert.ok(notes.some(note => note.includes('연결')));
});

test('failed discovery is visible and retry uses the new authoritative result', async () => {
  const profile = blankProfile(state, 'en');
  let calls = 0;
  const core = {call: async () => ++calls === 1 ? {state: 'error', detail: 'daemon unavailable', peers: [], self: null} : ready} as unknown as CoreClient;
  const notes: string[] = [];
  const ui: Prompter = {select: async (_message, options) => options.find(option => option.value === (calls === 1 ? 'retry' : 'peer:game'))!.value,
    text: async () => {throw new Error('retry must not fall back to manual');}, confirm: async () => true, note: message => notes.push(message)};
  await setupHosts(core, state, profile, ui);
  assert.equal(calls, 2);
  assert.ok(notes.some(note => note.includes('daemon unavailable')));
  assert.equal(profile.hosts.runner.address, '100.64.0.1');
});

test('manual LAN selection does not inject the Tailscale self address', async () => {
  const profile = blankProfile(state, 'en');
  const core = {call: async () => ready} as unknown as CoreClient;
  let typed = 0;
  const ui: Prompter = {select: async (_message, options) => options.find(option => option.value === 'manual')!.value,
    text: async (_message, initial) => {assert.equal(initial ?? '', ''); return ++typed === 1 ? 'game.lan' : 'agent.lan';},
    confirm: async () => true, note: () => {}};
  await setupHosts(core, state, profile, ui);
  assert.equal(profile.hosts.runner.address, 'agent.lan');
  assert.equal(profile.hosts.game.address, 'game.lan');
});

test('leaving combined mode allocates a host without replacing provider host entries', async () => {
  const profile = blankProfile(state, 'en');
  profile.hosts = {game: {os: 'windows', address: '127.0.0.1'}, 'game-2': {os: 'linux', address: 'provider.lan'}};
  profile.runner_host = profile.game_host = 'game';
  const core = {call: async () => ready} as unknown as CoreClient;
  const ui: Prompter = {select: async (_message, options) => options.find(o => o.value === 'peer:game')!.value,
    text: async () => {throw new Error('no typing');}, confirm: async () => true, note: () => {}};
  await setupHosts(core, state, profile, ui);
  assert.equal(profile.game_host, 'game-3');
  assert.equal(profile.hosts['game-2'].address, 'provider.lan');
});
