import test from 'node:test';
import assert from 'node:assert/strict';
import { setupAvatar } from '../src/setup-avatar.js';
import { blankProfile } from '../src/onboard.js';
import type { Bootstrap } from '../src/types.js';
import type { CoreClient } from '../src/core.js';
import type { Prompter } from '../src/prompts.js';
import { coreAction, parse } from '../src/commands.js';

for (const language of ['en', 'ko'] as const) {
  test(`avatar setup can finish with a calibrated base and no registered actions (${language})`, async () => {
    const state = {os: 'macos', python: 'python', cwd: '/project'} as Bootstrap;
    const profile = blankProfile(state, language);
    profile.avatar = {rig: 'test-rig', base_pose: '/assets/base.json', hmd_base: [0, 1, 0], face_channels: {},
      calibration: {signature: 'a'.repeat(64), tracking: 'active'}};
    const calls: string[] = [];
    const core = {call: async (action: string) => {
      calls.push(action);
      if (action === 'avatars.list') return {items: [], problems: []};
      assert.equal(action, 'avatar.inspect');
      return {ready: true};
    }} as unknown as CoreClient;
    const ui: Prompter = {select: async (_message, options, initial) => initial ?? options[0].value,
      text: async (_message, initial) => initial ?? '',
      confirm: async (_message, initial) => {assert.equal(initial, false); return false;}, note: () => {}};
    await setupAvatar(core, state, profile, ui);
    assert.deepEqual(profile.behaviors, {});
    assert.equal(profile.autonomy.enabled, false);
    assert.equal(profile.avatar.calibration?.signature, 'a'.repeat(64));
    assert.deepEqual(calls, ['avatars.list', 'avatar.inspect']);
  });
}

test('behavior discovery uses the same headless core from CLI and Pi input', () => {
  assert.deepEqual(coreAction(parse(['behaviors'])), {action: 'behaviors', payload: {}});
  assert.deepEqual(coreAction(parse(['/behaviors'])), {action: 'behaviors', payload: {}});
});
