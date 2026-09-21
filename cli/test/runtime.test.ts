import test from 'node:test';
import assert from 'node:assert/strict';
import { coreAction, parse, unsuccessful } from '../src/commands.js';
import { progressText, setupRuntime } from '../src/setup-runtime.js';
import type { CoreClient } from '../src/core.js';
import { blankProfile } from '../src/onboard.js';
import type { Bootstrap } from '../src/types.js';
import type { Prompter } from '../src/prompts.js';

test('runtime commands keep model Python separate from the core interpreter', () => {
  const parsed = parse(['runtime', 'setup', '--existing', './ARDY folder', '--runtime-python', './env/bin/python', '--python', '/core/python', '--wait']);
  const command = coreAction(parsed);
  assert.equal(command.action, 'runtime.setup');
  assert.equal(command.payload.mode, 'connect');
  assert.equal(command.payload.root, './ARDY folder');
  assert.equal(command.payload.python, './env/bin/python');
  assert.equal(parsed.options.python, '/core/python');
  assert.equal(parsed.wait, true);
  assert.deepEqual(coreAction(parse(['runtime', 'cancel', 'abc'])), {action: 'runtime.cancel', payload: {id: 'abc'}});
  assert.deepEqual(coreAction(parse(['runtime', 'verify'])), {action: 'runtime.verify', payload: {}});
  assert.throws(() => parse(['runtime', 'setup', '--existing', 'a', '--root', 'b']));
  assert.ok(unsuccessful({state: 'needs-auth'}));
});

test('completed setup supplies paths, while failed setup leaves the draft runtime intact', async () => {
  const state = {os: 'macos', python: '/core/python', cwd: '/project'} as Bootstrap;
  const profile = blankProfile(state, 'en');
  const calls: string[] = [];
  let succeeded = false;
  const core = {call: async (action: string) => {
    calls.push(action);
    if (action === 'runtime.options') return {default_root: '/managed', project_dir: '/project', host: {device: 'mps'}, recipe: {device: 'mps'}};
    if (action === 'runtime.status') return {id: null, state: 'not-configured'};
    if (action === 'runtime.plan') return {paths: {root: '/managed', models: '/models'}, model_bytes: 16, checks: []};
    if (action === 'runtime.setup') return succeeded ? {id: 'job', state: 'complete'} : {id: 'job', state: 'failed', error: 'installation failed'};
    if (action === 'runtime.result') return {python: '/managed/runtime/bin/python', checkpoints_dir: '/models/checkpoints', device: 'mps'};
    throw new Error(action);
  }} as unknown as CoreClient;
  const prompts: Prompter = {select: async (_message, opts) => opts[0].value, text: async (_message, initial) => initial ?? '',
    confirm: async (_message, initial) => initial ?? true, note: () => {}};
  await assert.rejects(setupRuntime(core, state, profile, prompts), /installation failed/);
  assert.equal(profile.runtime.python, '/core/python');
  assert.ok(!calls.includes('runtime.result'));
  succeeded = true;
  await setupRuntime(core, state, profile, prompts);
  assert.equal(profile.runtime.python, '/managed/runtime/bin/python');
  assert.equal(profile.runtime.checkpoints_dir, '/models/checkpoints');
});

test('progress describes the current stage and model bytes in both languages', () => {
  const job = {id: 'job', state: 'running', step: 'models', completed: 2, total: 5, bytes_completed: 1024 ** 3, bytes_total: 2 * 1024 ** 3};
  assert.match(progressText(job, 'ko'), /모델 준비/);
  assert.match(progressText(job, 'en'), /Preparing models/);
  assert.match(progressText(job, 'en'), /1.0\/2.0 GiB/);
});

test('path suggestions accept a new installation directory with spaces and Korean text', async () => {
  const {pathOptions} = await import('../src/path-input.js');
  const {mkdtemp, rm, mkdir} = await import('node:fs/promises');
  const {join, sep} = await import('node:path');
  const {tmpdir} = await import('node:os');
  const root = await mkdtemp(join(tmpdir(), 'vrc-path-'));
  try {
    const target = join(root, '새 설치 폴더');
    assert.equal(pathOptions(target, true, 'ko')[0].value, target);
    await mkdir(join(root, 'existing folder'));
    const rows = pathOptions(root, true, 'en');
    assert.equal(rows[0].value, root + sep);
    assert.ok(rows.some(row => row.value === join(root, 'existing folder') + sep));
  } finally { await rm(root, {recursive: true, force: true}); }
});
