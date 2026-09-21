import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

test('linked CLI runs from outside the project directory', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vrc-agent-bin-'));
  try {
    const executable = join(directory, 'vrc-agent.ts');
    await symlink(fileURLToPath(new URL('../src/cli.ts', import.meta.url)), executable);
    const result = spawnSync(process.execPath, ['--import', import.meta.resolve('tsx'), executable, '--help'], {
      cwd: directory, encoding: 'utf8', timeout: 10_000,
    });
    assert.equal(result.error, undefined);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /VRC AGENT/);
    assert.match(result.stdout, /doctor/);
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});
