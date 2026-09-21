import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { CoreClient, CoreCanceled } from '../src/core.js';

test('canceling a running HTTP check stops its core and records no success', {timeout: 10000}, async () => {
  const root = await mkdtemp(join(tmpdir(), 'vrc-cancel-'));
  let observed!: () => void;
  const started = new Promise<void>(accept => {observed = accept;});
  const server = createServer((_req, _res) => observed());
  await new Promise<void>(accept => server.listen(0, '127.0.0.1', accept));
  const previous = process.listenerCount('SIGINT');
  try {
    const port = (server.address() as {port: number}).port;
    const core = new CoreClient({python: resolve('.venv/bin/python'), config: join(root, 'profile.json'), stateDir: root});
    const payload = {provider: 'openai-compatible', config: {base_url: `http://127.0.0.1:${port}/v1`}, capability: 'llm', settings: {model: 'test'}};
    const pending = core.call('connection.test', payload);
    const rejected = assert.rejects(pending, CoreCanceled);
    await started;
    process.emit('SIGINT');
    await rejected;
    assert.equal(process.listenerCount('SIGINT'), previous);
    const status = await core.call<{observation: {state: string}}>('connection.observation', payload);
    assert.equal(status.observation.state, 'interrupted');
  } finally {
    server.closeAllConnections();
    await new Promise<void>(accept => server.close(() => accept()));
    await rm(root, {recursive: true, force: true});
  }
});
