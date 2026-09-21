import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { CoreClient } from '../src/core.js';
import { onboard } from '../src/onboard.js';
import type { Bootstrap } from '../src/types.js';
import type { Prompter } from '../src/prompts.js';

test('fresh setup reuses a custom connection and manual voice, commits once, then opens review', {timeout: 20000}, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vrc-wizard-discovery-'));
  const requests: string[] = [];
  const server = createServer((req, res) => {
    requests.push(`${req.method} ${req.url}`);
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(req.url?.endsWith('/v1/models') ? {data: [{id: 'test-model'}]} : {voices: [{id: 'test-voice', name: 'Test voice'}]}));
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const address = server.address() as {port: number};
    const core = new CoreClient({config: join(directory, 'agent.json'), python: resolve('.venv/bin/python')});
    const rawCall = core.call.bind(core);
    const actions: string[] = [];
    core.call = async <T>(action: string, payload?: Record<string, unknown>): Promise<T> => {
      actions.push(action);
      if (action === 'hosts.discover') return {state: 'ready', self: {address: '100.64.0.1'},
        peers: [{id: 'game', name: 'Game', address: '100.64.0.2', os: 'windows', online: false}]} as T;
      if (action === 'runtime.options') return {host: {device: 'cpu'}, recipe: null} as T;
      if (action === 'runtime.status') return {id: null, state: 'not-configured'} as T;
      return rawCall<T>(action, payload);
    };
    let textInputs = 0;
    const ui: Prompter = {
      note: () => {}, confirm: async () => false,
      text: async (message) => {textInputs++; return message.includes('ID') ? 'test-voice' : `http://127.0.0.1:${address.port}/v1`;},
      select: async (_message, options, initial) => {
        for (const choice of ['ko', 'peer:game', 'server', 'openai-compatible', 'llm-1', 'later']) {
          if (options.some(o => o.value === choice)) return options.find(o => o.value === choice)!.value;
        }
        return initial ?? options[0].value;
      },
    };
    const saved = await onboard(core, await core.call<Bootstrap>('bootstrap'), ui);
    assert.ok(saved && saved.profile.role === 'agent');
    assert.equal(saved.profile.bindings.tts.voice, 'test-voice');
    assert.equal(saved.profile.bindings.llm.model, 'test-model');
    assert.equal(textInputs, 2, 'custom server address and voice ID without a declared inventory path');
    assert.deepEqual(requests, ['GET /v1/models', 'GET /v1/models', 'GET /v1/models']);
    assert.equal(actions.filter(a => a === 'profile.commit').length, 1);
    const state = await core.call<Bootstrap>('bootstrap');
    assert.equal(state.draft, null);
    const reused: string[] = [];
    ui.select = async (_message, options, initial) => {reused.push(initial!); assert.ok(options.some(o => o.value === '@save')); return '@save' as typeof initial & string;};
    ui.text = async () => {throw new Error('existing setup should not ask for fields again');};
    await onboard(core, state, ui);
    assert.deepEqual(reused, ['@save']);
    assert.equal(requests.length, 3);
    let reviewing = 0;
    ui.select = async (_message, options, initial) => {
      if (options.some(o => o.value === '@save')) {const value = ++reviewing === 1 ? '@advanced' : '@save'; return options.find(o => o.value === value)!.value;}
      if (options.some(o => o.value === 'ports')) return options.find(o => o.value === 'llm')!.value;
      return initial ?? options[0].value;
    };
    ui.text = async (_message, initial) => initial?.startsWith('http://') ? `http://127.0.0.1:${address.port}/changed/v1` : initial ?? '';
    await onboard(core, await core.call<Bootstrap>('bootstrap'), ui);
    assert.deepEqual(requests.slice(3), ['GET /changed/v1/models', 'GET /changed/v1/models'],
      'editing a shared connection rechecks the other capability selections');
  } finally {
    await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
    await rm(directory, {recursive: true, force: true});
  }
});
