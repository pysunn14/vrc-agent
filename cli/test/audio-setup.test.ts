import test from 'node:test';
import assert from 'node:assert/strict';
import { selectAudioDevice } from '../src/setup-audio.js';
import type { Prompter } from '../src/prompts.js';

test('audio routing lists only matching channel direction and reuses a unique saved device', async () => {
  const devices = [{index: 0, name: 'Mic', max_input_channels: 1, max_output_channels: 0},
    {index: 1, name: 'Virtual cable', max_input_channels: 0, max_output_channels: 2}];
  const ui = {note: () => {}, select: async (_message: string, options: any[]) => {
    assert.deepEqual(options.filter(o => o.value.startsWith('device:')).map(o => o.label), ['Virtual cable']);
    return 'device:1';
  }} as unknown as Prompter;
  assert.equal(await selectAudioDevice(ui, devices, 'output', 'en'), 'Virtual cable');
  ui.select = async () => {throw new Error('valid saved route should be reused');};
  assert.equal(await selectAudioDevice(ui, devices, 'output', 'en', 'Virtual cable'), 'Virtual cable');
});

test('ambiguous saved device requires selection and duplicate names are stored as explicit indices', async () => {
  const devices = [0, 1].map(index => ({index, name: 'Cable', host_api: `API ${index}`, max_output_channels: 2, max_input_channels: 0}));
  let asked = false;
  const ui = {note: () => {}, select: async () => {asked = true; return 'device:1';}} as unknown as Prompter;
  assert.equal(await selectAudioDevice(ui, devices, 'output', 'ko', 'Cable'), '1');
  assert.ok(asked);
});
