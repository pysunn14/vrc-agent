import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { CoreClient } from '../src/core.js';
import type { Bootstrap } from '../src/types.js';

test('npm client reads the Python registry without installing any model', async () => {
  const root = await mkdtemp(join(tmpdir(), 'vrc-agent-cli-'));
  try {
    const client = new CoreClient({config: join(root, 'agent.json'), python: resolve('.venv/bin/python')});
    const state = await client.call<Bootstrap>('bootstrap');
    assert.equal(state.profile, null);
    assert.equal(state.protocol, 3);
    assert.ok(state.catalog.elevenlabs.operations.tts);
    assert.equal(state.catalog.elevenlabs.operations.llm, undefined);
    const value = await client.call<Record<string, unknown>>('connection.validate', {
      provider: 'gemini', config: {api_key_env: 'EXAMPLE_KEY'},
    });
    assert.equal(value.base_url, 'https://generativelanguage.googleapis.com/v1beta/openai');
    await assert.rejects(client.call('connection.validate', {provider: 'missing', config: {}}), /unknown provider/);
  } finally { await rm(root, {recursive: true, force: true}); }
});

test('missing core interpreter fails with an actionable error', async () => {
  const client = new CoreClient({python: '/missing/python'});
  await assert.rejects(client.call('bootstrap'), /Python core/);
});
