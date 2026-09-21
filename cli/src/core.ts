import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import type { CoreEvent, Options } from './types.js';

export class CoreCanceled extends Error {}

export class CoreClient {
  constructor(readonly options: Options = {}, readonly onEvent?: (event: CoreEvent) => void) {}

  async call<T = unknown>(action: string, payload: Record<string, unknown> = {}): Promise<T> {
    const python = this.options.python ?? process.env.VRC_AGENT_PYTHON;
    // uv selects the project's lightweight core environment; ARDY's interpreter
    // is a separate profile field. --no-sync avoids changing an active runtime.
    const command = python ?? 'uv';
    const args = [...(python ? [] : ['run', '--no-sync', 'python']), '-m', 'vrc_ardy_agent.runner.control', action];
    if (this.options.config) args.push('--config', resolve(this.options.config));
    if (this.options.stateDir) args.push('--state-dir', resolve(this.options.stateDir));
    const cwd = fileURLToPath(new URL(import.meta.url.endsWith('.ts') ? '../../' : '../', import.meta.url));
    return new Promise<T>((accept, reject) => {
      const child = spawn(command, args, {cwd, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true});
      let buffered = '', stderr = '', result: T, hasResult = false, failure = '';
      let bytes = 0, canceled = false;
      const cancel = () => {canceled = true; failure = 'Core request canceled'; child.kill('SIGINT');};
      process.once('SIGINT', cancel);
      process.once('SIGTERM', cancel);
      const cleanup = () => {process.removeListener('SIGINT', cancel); process.removeListener('SIGTERM', cancel);};
      const deadline = setTimeout(() => {
        failure = `Python core request timed out: ${action}`;
        child.kill();
      }, 660_000);
      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      child.stderr.on('data', (data: string) => { stderr = (stderr + data).slice(-8192); });
      child.stdout.on('data', (data: string) => {
        buffered += data;
        bytes += Buffer.byteLength(data);
        if (bytes > 8 * 1024 * 1024) { failure = 'Python core output exceeded its size limit'; child.kill(); return; }
        let newline: number;
        while ((newline = buffered.indexOf('\n')) >= 0) {
          const line = buffered.slice(0, newline); buffered = buffered.slice(newline + 1);
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line) as CoreEvent;
            if (event.event === 'result') {
              if (action === 'bootstrap' && (!event.data || typeof event.data !== 'object'
                || !('protocol' in event.data) || event.data.protocol !== 3)) {
                failure = 'Unsupported Python core protocol; install matching CLI and core versions';
              } else { result = event.data as T; hasResult = true; }
            }
            else if (event.event === 'error') failure = `${event.type}: ${event.message}`;
            else this.onEvent?.(event);
          } catch { failure = 'Python core returned an invalid event'; child.kill(); }
        }
      });
      child.on('error', error => {
        cleanup();
        clearTimeout(deadline);
        reject(new Error(`Python core could not start (${command}): ${error.message}. Run uv sync or set VRC_AGENT_PYTHON.`));
      });
      child.on('close', code => {
        cleanup();
        clearTimeout(deadline);
        if (canceled) {reject(new CoreCanceled('Core request canceled')); return;}
        if (failure || code !== 0 || !hasResult) reject(new Error(failure || `Python core exited ${code}: ${stderr || 'no result received'}`));
        else accept(result!);
      });
      child.stdin.on('error', () => { /* Process failure is reported by close/error. */ });
      child.stdin.end(JSON.stringify(payload));
    });
  }
}
