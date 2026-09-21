import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, mkdirSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, sep} from 'node:path';
import {pathOptions} from '../src/path-input.js';

test('Tab candidate completes a partial folder with a trailing separator', () => {
  const root = mkdtempSync(join(tmpdir(), 'path-completion-'));
  try {
    mkdirSync(join(root, 'projects'));
    mkdirSync(join(root, 'projects', 'avatar'));
    const options = pathOptions(join(root, 'proj'), false, 'ko');
    assert.equal(options[0].value, join(root, 'projects') + sep);
    assert.ok(pathOptions(options[0].value, false, 'ko').some(row => row.value === join(root, 'projects', 'avatar') + sep));
    // A new path remains available for installation directory prompts.
    assert.ok(options.some(row => row.value === join(root, 'proj')));
    assert.equal(pathOptions(join(root, 'new folder'), true, 'ko')[0].value, join(root, 'new folder'));
  } finally { rmSync(root, {recursive: true, force: true}); }
});

test('file completion preserves exact files and directory-only filtering', () => {
  const root = mkdtempSync(join(tmpdir(), 'path-completion-'));
  try {
    writeFileSync(join(root, 'avatar.json'), '{}');
    assert.equal(pathOptions(join(root, 'ava'), false, 'en')[0].value, join(root, 'avatar.json'));
    assert.equal(pathOptions(join(root, 'avatar.json'), false, 'en')[0].value, join(root, 'avatar.json'));
    assert.equal(pathOptions(join(root, 'ava'), true, 'en').length, 1);
  } finally { rmSync(root, {recursive: true, force: true}); }
});

test('file picker enters directories and returns only a file', async () => {
  const {pickPath} = await import('../src/path-input.js');
  const root = mkdtempSync(join(tmpdir(), 'file-picker-'));
  try {
    const file = join(root, 'avatar.json');
    writeFileSync(file, '{}');
    const visited: string[] = [];
    const choices = [root, file];
    const result = await pickPath(async initial => {
      visited.push(initial);
      return choices.shift()!;
    }, '', false);
    assert.equal(result, file);
    assert.deepEqual(visited, ['', root + sep]);
    assert.equal(await pickPath(async () => root, '', true), root);
  } finally { rmSync(root, {recursive: true, force: true}); }
});
