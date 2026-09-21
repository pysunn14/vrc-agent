import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseConnectionPreset } from '../src/setup-connection-choice.js';
import type { Bootstrap } from '../src/types.js';
import type { Prompter } from '../src/prompts.js';

const state = {catalog: {
  cloud: {name: 'Cloud', setup: {category: 'service'}, operations: {llm: {}, stt: {}, tts: {}}},
  'openai-compatible': {name: 'Compatible', setup: {category: 'server'}, operations: {llm: {}, stt: {}, tts: {}}},
  local: {name: 'Local', setup: {category: 'local'}, operations: {stt: {}}},
}} as unknown as Bootstrap;
for (const cap of ['llm', 'stt', 'tts'] as const) test(`custom ${cap} server bypasses service and protocol questions`, async () => {
  let count = 0;
  const ui = {select: async (_message: string, choices: {value: string}[]) => {
    count++;
    assert.equal(choices.some(c => c.value === 'local'), cap === 'stt');
    return 'server';
  }} as Prompter;
  assert.equal(await chooseConnectionPreset(state, ui, cap, true), 'openai-compatible');
  assert.equal(count, 1);
});
