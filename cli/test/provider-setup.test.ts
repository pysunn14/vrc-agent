import test from 'node:test';
import assert from 'node:assert/strict';
import { configureProvider } from '../src/setup-providers.js';
import { blankProfile } from '../src/setup-profile.js';
import { SetupCanceled, type Prompter } from '../src/prompts.js';
import { inventoryChoice } from '../src/setup-provider-inventory.js';
import type { CoreClient } from '../src/core.js';
import type { Bootstrap } from '../src/types.js';

const field = (defaultValue: string | number | boolean, required = false) => ({type: typeof defaultValue, default: defaultValue, required, label: {en: 'setting', ko: '설정'}});
const state = {os: 'macos', python: 'python', cwd: '.', catalog: {
  hermes: {name: 'Hermes', protocol: 'openai-compatible', extension: 'hermes', deployment_kinds: ['api', 'external', 'managed'],
    fields: {base_url: field('http://localhost:8642/v1', true), api_key_env: field('API_SERVER_KEY', true), timeout_seconds: field(90)},
    operations: {llm: {model: field('', true), vision: field(false), reasoning_effort: field(''), upstream_provider: field('')}}},
}} as unknown as Bootstrap;
const ready = {state: 'ready', detail: 'inventory only', items: [{id: 'server-alias', label: 'server-alias'}], next_cursor: null};

test('a single discovered model fills defaults, asks only provider/address/vision, and checkpoints the connection', async () => {
  const profile = blankProfile(state, 'ko'), snapshots: unknown[] = [], prompts: string[] = [];
  const core = {call: async (action: string, payload: any) => {
    if (action === 'connection.validate') return payload.config;
    if (action === 'binding.validate') return payload.settings;
    assert.equal(action, 'connection.discover');
    return ready;
  }} as unknown as CoreClient;
  const ui: Prompter = {
    select: async (message, choices) => {prompts.push(message); assert.equal(choices[0].value, 'hermes'); return choices[0].value;},
    text: async (message, initial) => {prompts.push(message); return initial!;},
    confirm: async (message) => {prompts.push(message); return true;}, note: () => {},
  };
  await configureProvider(core, ui, state, profile, 'llm', {checkpoint: async () => {snapshots.push(structuredClone(profile));}});
  assert.equal(prompts.length, 3);
  assert.equal(profile.bindings.llm.model, 'server-alias');
  assert.equal(profile.bindings.llm.vision, true);
  assert.equal(profile.providers[profile.bindings.llm.instance].deployment.kind, 'api');
  assert.ok(snapshots.length >= 2);
});

test('cancel while choosing a model retains the entered connection for resume', async () => {
  const profile = blankProfile(state, 'en');
  const core = {call: async (action: string, payload: any) => action === 'connection.validate' ? payload.config
    : {...ready, items: [{id: 'one', label: 'one'}, {id: 'two', label: 'two'}]}} as unknown as CoreClient;
  const ui: Prompter = {
    select: async (_message, choices) => {if (choices[0].value !== 'hermes') throw new SetupCanceled(); return choices[0].value;},
    text: async (_message, initial) => initial!, confirm: async () => false, note: () => {},
  };
  await assert.rejects(configureProvider(core, ui, state, profile, 'llm'), SetupCanceled);
  const connection = profile.providers[profile.bindings.llm.instance];
  assert.equal(connection.provider, 'hermes');
  assert.equal(connection.config.base_url, 'http://localhost:8642/v1');
});

test('saved model absent from inventory is never silently replaced by the sole new model', async () => {
  const profile = blankProfile(state, 'en');
  profile.providers.existing = {provider: 'hermes', label: 'Saved', config: {}, deployment: {kind: 'api'}};
  profile.bindings.llm = {instance: 'existing', model: 'saved-model', vision: true};
  let kept = false;
  const core = {call: async (action: string, payload: any) => action === 'binding.validate' ? payload.settings : action === 'connection.validate' ? payload.config : ready} as unknown as CoreClient;
  const ui: Prompter = {
    select: async (_message, choices) => {const keep = choices.find(c => c.value === '@keep'); assert.ok(keep); kept = true; return keep.value;},
    text: async () => {throw new Error('no text expected');}, confirm: async () => {throw new Error('no question expected');}, note: () => {},
  };
  await configureProvider(core, ui, state, profile, 'llm');
  assert.ok(kept);
  assert.equal(profile.bindings.llm.model, 'saved-model');
  assert.equal(profile.bindings.llm.vision, true);
});

test('authentication failure can be repaired and retried without silently entering an offline model', async () => {
  const connection = {provider: 'hermes', label: 'Hermes', config: {api_key_env: 'MISSING'}, deployment: {kind: 'api'}};
  let calls = 0, repaired = false;
  const core = {call: async (_action: string, payload: any) => {
    calls++;
    if (calls === 1) return {state: 'needs-auth', detail: 'MISSING', items: [], next_cursor: null};
    assert.equal(payload.config.api_key_env, 'AVAILABLE');
    return ready;
  }} as unknown as CoreClient;
  const ui = {note: () => {}, select: async (_message: string, choices: any[]) => {
    assert.ok(choices.some(c => c.value === '@repair')); return '@repair';
  }} as unknown as Prompter;
  const result = await inventoryChoice(core, ui, connection, 'llm', 'models', 'ko', '', false,
    async () => {connection.config.api_key_env = 'AVAILABLE'; repaired = true;});
  assert.equal(result, 'server-alias');
  assert.ok(repaired);
  assert.equal(calls, 2);
});

test('voice pagination is requested only by the user and a later page is selectable', async () => {
  const cursors: unknown[] = [];
  const core = {call: async (_action: string, payload: any) => {
    cursors.push(payload.cursor);
    return {...ready, items: [{id: payload.cursor ? 'second' : 'first', label: 'Voice'}], next_cursor: payload.cursor ? null : 'next'};
  }} as unknown as CoreClient;
  const ui = {note: () => {}, select: async () => cursors.length === 1 ? '@more' : 'item:0'} as unknown as Prompter;
  assert.equal(await inventoryChoice(core, ui, {provider: 'elevenlabs', label: 'Voices', config: {}, deployment: {kind: 'api'}},
    'tts', 'voices', 'en'), 'second');
  assert.deepEqual(cursors, [undefined, 'next']);
});

test('an interrupted vision choice is asked again on resume, even after the model has been selected', async () => {
  const profile = blankProfile(state, 'en');
  const core = {call: async (action: string, payload: any) => action === 'connection.validate' ? payload.config
    : action === 'binding.validate' ? payload.settings : ready} as unknown as CoreClient;
  const ui: Prompter = {note: () => {}, select: async (_message, choices) => choices[0].value,
    text: async (_message, initial) => initial!, confirm: async () => {throw new SetupCanceled();}};
  await assert.rejects(configureProvider(core, ui, state, profile, 'llm'), SetupCanceled);
  let asked = false;
  ui.confirm = async () => {asked = true; return true;};
  await configureProvider(core, ui, state, profile, 'llm');
  assert.ok(asked);
  assert.equal(profile.bindings.llm.vision, true);
});
