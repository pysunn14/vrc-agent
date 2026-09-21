/** Development-only exploratory control. Deliberately outside the release build. */
import { readFile, writeFile, rename, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { parseArgs } from 'node:util';
import { CoreClient } from '../src/core.js';

const {values, positionals} = parseArgs({allowPositionals: true, options: {
  config: {type: 'string'}, python: {type: 'string'}, 'state-dir': {type: 'string'},
  input: {type: 'string'}, output: {type: 'string'},
}});
const core = new CoreClient({config: values.config, python: values.python, stateDir: values['state-dir']},
  event => process.stderr.write(JSON.stringify(event) + '\n'));
const actions = ['calibration.prepare', 'calibration.adjust', 'calibration.status', 'calibration.start', 'calibration.update', 'calibration.heartbeat', 'calibration.cancel', 'calibration.complete', 'avatars.list', 'avatars.inspect', 'avatars.import', 'avatars.select', 'avatars.register', 'avatars.export', 'avatar.files.inspect', 'avatar.rig.inspect', 'providers.check', 'providers.observations', 'connection.describe', 'connection.test', 'connection.observation', 'connection.auth.prepare', 'connection.auth.save', 'connection.discover', 'binding.validate', 'bridge.discover', 'hosts.discover', 'connection.info', 'bridge.paths', 'bootstrap', 'catalog', 'connection.validate', 'profile.validate', 'profile.commit', 'draft.save',
  'settings', 'bridge.command', 'providers', 'providers.list', 'providers.test', 'providers.remove', 'runtime.probe', 'runtime.options', 'runtime.discover', 'runtime.paths', 'runtime.plan', 'runtime.setup', 'runtime.status', 'runtime.resume', 'runtime.cancel', 'runtime.logs', 'runtime.verify', 'runtime.result', 'runtime.auth', 'doctor',
  'avatar.inspect', 'avatar.attest', 'behaviors', 'behaviors.import', 'behaviors.attest', 'behaviors.inspect', 'snapshot', 'status', 'logs', 'start', 'stop'];

try {
  if (positionals[0] === 'actions') console.log(JSON.stringify({actions, schedule: true}, null, 2));
  else if (positionals[0] === 'schedule') {
    if (!values.input || !values.output) throw new Error('schedule requires --input and --output');
    const jobs = JSON.parse(await readFile(values.input, 'utf8')) as {at_seconds: number; action: string; payload?: Record<string, unknown>}[];
    if (!Array.isArray(jobs) || !jobs.length || jobs.length > 32 || jobs.some(job =>
      !Number.isFinite(job.at_seconds) || job.at_seconds < 0 || job.at_seconds > 30 || !actions.includes(job.action)))
      throw new Error('Schedule requires 1..32 valid operations within 30 seconds');
    const output = resolve(values.output), results: unknown[] = Array(jobs.length).fill(null);
    await mkdir(dirname(output), {recursive: true});
    let checkpoint = Promise.resolve();
    const save = () => {
      const body = JSON.stringify({completed: results.filter(Boolean).length, total: jobs.length, results}, null, 2);
      checkpoint = checkpoint.then(async () => { await writeFile(output + '.tmp', body); await rename(output + '.tmp', output); });
      return checkpoint;
    };
    await save();
    const started = performance.now();
    const heartbeat = setInterval(() => process.stderr.write(JSON.stringify({event: 'heartbeat', completed: results.filter(Boolean).length, total: jobs.length}) + '\n'), 1000);
    try {
      await Promise.all(jobs.map(async (job, index) => {
        await new Promise(resolve => setTimeout(resolve, Math.max(0, job.at_seconds*1000 - (performance.now()-started))));
        const at = (performance.now()-started)/1000;
        try { results[index] = {action: job.action, started_seconds: at, result: await core.call(job.action, job.payload)}; }
        catch (error) { results[index] = {action: job.action, started_seconds: at, error: String(error)}; process.exitCode = 1; }
        await save();
        process.stderr.write(JSON.stringify({event: 'progress', completed: results.filter(Boolean).length, total: jobs.length}) + '\n');
      }));
    } finally { clearInterval(heartbeat); }
    console.log(JSON.stringify({output, results}, null, 2));
  } else {
    const action = positionals[0];
    if (!actions.includes(action)) throw new Error('Use actions, schedule, or a listed core operation');
    const payload = values.input ? JSON.parse(await readFile(values.input, 'utf8')) : {};
    console.log(JSON.stringify(await core.call(action, payload), null, 2));
  }
} catch (error) { console.error(String(error)); process.exitCode = 2; }
