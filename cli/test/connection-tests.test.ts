import test from 'node:test';
import assert from 'node:assert/strict';
import { testConnections } from '../src/setup-connection-tests.js';
import { blankProfile } from '../src/setup-profile.js';
import type { Bootstrap } from '../src/types.js';
import type { Prompter } from '../src/prompts.js';
import type { CoreClient } from '../src/core.js';

function profile() {
  const p = blankProfile({os: 'macos', python: 'python', cwd: '.'} as Bootstrap, 'ko');
  p.providers.c = {provider: 'openai-compatible', label: 'Connection', config: {base_url: 'http://example.test/v1'}, deployment: {kind: 'api'}};
  p.bindings = {llm: {instance: 'c', model: 'chat'}, stt: {instance: 'c', model: 'stt'}, tts: {instance: 'c', model: 'tts', voice: 'voice'}};
  return p;
}
test('one explicit test action checks all three bindings without starting the agent', async () => {
  const calls: string[] = [];
  const core = {call: async (action: string, payload: any) => {assert.equal(action, 'connection.test'); calls.push(payload.capability); return {state: 'success', recorded: true};}} as unknown as CoreClient;
  assert.equal(await testConnections(core, profile(), {note: () => {}} as Prompter), true);
  assert.deepEqual(calls, ['llm', 'stt', 'tts']);
});
test('a late successful response cannot validate modified settings', async () => {
  const p = profile();
  const core = {call: async () => {p.bindings.llm.model = 'changed'; return {state: 'success', recorded: true};}} as unknown as CoreClient;
  assert.equal(await testConnections(core, p, {note: () => {}} as Prompter), false);
});
test('failure keeps configuration intact and reports unsuccessful tests', async () => {
  const p = profile(), before = structuredClone(p);
  const core = {call: async () => ({state: 'failed', reason: 'auth-error'})} as unknown as CoreClient;
  assert.equal(await testConnections(core, p, {note: () => {}} as Prompter), false);
  assert.deepEqual(p, before);
});

test('canceling one test never starts the next provider request', async () => {
  const { CoreCanceled } = await import('../src/core.js');
  const { SetupCanceled } = await import('../src/prompts.js');
  let calls = 0;
  const core = {call: async () => {calls++; throw new CoreCanceled();}} as unknown as CoreClient;
  await assert.rejects(testConnections(core, profile(), {note: () => {}} as Prompter), SetupCanceled);
  assert.equal(calls, 1);
});
