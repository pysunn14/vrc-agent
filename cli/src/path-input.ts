import { readdirSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve, sep } from 'node:path';
import type { Language } from './types.js';

/** Complete existing matches first; retain typed paths for new install directories. */
export function pathOptions(input: string, directory: boolean, language: Language) {
  const expanded = input === '~' ? homedir() : input.startsWith('~/') || input.startsWith('~\\') ? join(homedir(), input.slice(2)) : input;
  const selected = resolve(expanded || '.');
  const rows = [{value: selected, label: selected, hint: language === 'ko' ? '이 경로 사용' : 'Use this path'}];
  let parent = dirname(selected), prefix = basename(selected);
  let exists = false;
  try {
    const stat = statSync(selected);
    exists = true;
    if (stat.isDirectory()) {
      parent = selected; prefix = '';
      rows[0].value = selected.endsWith(sep) ? selected : selected + sep;
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
  }
  let children;
  try { children = readdirSync(parent, {withFileTypes: true}); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return rows;
    throw error;
  }
  const matches = children.filter(entry => entry.name.startsWith(prefix) && (!directory || entry.isDirectory()))
    .sort((a, b) => a.name.localeCompare(b.name)).slice(0, 100).map(entry => ({
      value: join(parent, entry.name) + (entry.isDirectory() ? sep : ''), label: entry.name + (entry.isDirectory() ? sep : ''), hint: '',
    })).filter(row => resolve(row.value) !== selected);
  // Clack completes the focused value on Tab. A partial input must not be
  // first, otherwise Tab only copies the unfinished path back into itself.
  return exists ? rows.concat(matches) : matches.concat(rows);
}

/** Enter navigates a folder in file mode; only directory mode submits it. */
export async function pickPath(select: (initial: string) => Promise<string>, initial = '', directory = false): Promise<string> {
  let current = initial;
  while (true) {
    const selected = await select(current);
    if (directory || !statSync(selected).isDirectory()) return selected;
    current = selected.endsWith(sep) ? selected : selected + sep;
  }
}

export function validatePath(value: string, directory: boolean, language: Language): string | undefined {
  try {
    const stat = statSync(value);
    if (directory && !stat.isDirectory()) return language === 'ko' ? '폴더를 선택하세요.' : 'Select a directory.';
    if (!stat.isDirectory() && !stat.isFile()) return language === 'ko' ? '일반 파일을 선택하세요.' : 'Select a regular file.';
  } catch (error) {
    if (directory && (error as NodeJS.ErrnoException).code === 'ENOENT') return;
    return language === 'ko' ? '읽을 수 있는 파일이나 폴더를 선택하세요.' : 'Select an accessible file or directory.';
  }
}
