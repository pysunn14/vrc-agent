import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import type { Options } from './types.js';

export interface Parsed {command: string; args: string[]; options: Options; offline: boolean; text?: string; audio?: string; cursor?: string; wait: boolean; runtime: Record<string, unknown>}
export function parse(argv: string[]): Parsed {
  const {values, positionals} = parseArgs({args: argv, allowPositionals: true, options: {
    config: {type: 'string'}, 'state-dir': {type: 'string'}, python: {type: 'string'}, language: {type: 'string'},
    cursor: {type: 'string'}, json: {type: 'boolean'}, offline: {type: 'boolean'}, text: {type: 'string'}, audio: {type: 'string'},
    root: {type: 'string'}, 'models-dir': {type: 'string'}, existing: {type: 'string'},
    'runtime-python': {type: 'string'}, device: {type: 'string'}, 'checkpoints-dir': {type: 'string'},
    'hf-cache-dir': {type: 'string'}, 'project-dir': {type: 'string'}, wait: {type: 'boolean'},
    help: {type: 'boolean', short: 'h'},
  }});
  if (values.root && values.existing) throw new Error('Use --root or --existing, not both');
  if (values.language && !['en', 'ko'].includes(values.language)) throw new Error('language must be en or ko');
  return {command: values.help ? 'help' : (positionals[0] ?? '').replace(/^\//, ''), args: positionals.slice(1),
    options: {config: values.config, stateDir: values['state-dir'], python: values.python,
      language: values.language as Options['language'], json: values.json},
    offline: values.offline ?? false, text: values.text, audio: values.audio, cursor: values.cursor, wait: values.wait ?? false,
    runtime: Object.fromEntries(Object.entries({mode: values.existing ? 'connect' : 'install',
      root: values.existing ?? values.root, models_dir: values['models-dir'], python: values['runtime-python'],
      device: values.device, checkpoints_dir: values['checkpoints-dir'], hf_cache_dir: values['hf-cache-dir'],
      project_dir: values['project-dir'], base_dir: process.cwd()}).filter(([, value]) => value !== undefined))};
}

export function splitCommand(line: string): string[] {
  const tokens: string[] = [];
  let token = '', quote = '', started = false;
  for (const character of line) {
    if (quote) {
      if (character === quote) quote = ''; else token += character;
    } else if (character === '"' || character === "'") { quote = character; started = true; }
    else if (/\s/.test(character)) {
      if (started) { tokens.push(token); token = ''; started = false; }
    } else { token += character; started = true; }
  }
  if (quote) throw new Error('Unclosed quote');
  if (started) tokens.push(token);
  return tokens;
}

export function coreAction(parsed: Parsed): {action: string; payload: Record<string, unknown>} {
  const {command, args} = parsed;
  if (command === 'avatars') {
    if (args.join(' ') === 'list') return {action:'avatars.list',payload:{}};
    if (args[0] === 'import' && args.length === 2) return {action:'avatars.import',payload:{path:resolve(args[1])}};
    if (args[0] === 'export' && args.length === 3) return {action:'avatars.export',payload:{key:args[1],path:resolve(args[2])}};
  }
  if (command === 'hosts' && args.join(' ') === 'discover') return {action: 'hosts.discover', payload: {}};
  if (command === 'connection' && args.join(' ') === 'info') return {action: 'connection.info', payload: {}};
  if (command === 'doctor' && !args.length) return {action: 'doctor', payload: {offline: parsed.offline}};
  if (['status', 'providers', 'snapshot', 'behaviors'].includes(command) && !args.length) return {action: command, payload: {}};
  if (['start', 'stop', 'logs'].includes(command) && args.length === 1) return {action: command, payload: {service: args[0]}};
  if (command === 'settings' && args.length <= 1) return {action: 'settings', payload: args[0] ? {language: args[0]} : {}};
  if (command === 'bridge' && args.join(' ') === 'command') return {action: 'bridge.command', payload: {}};
  if (command === 'runtime') {
    if (['setup', 'plan'].includes(args[0]) && args.length === 1) return {action: `runtime.${args[0]}`, payload: parsed.runtime};
    if (['status', 'resume', 'cancel', 'logs', 'result', 'auth'].includes(args[0]) && args.length <= 2)
      return {action: `runtime.${args[0]}`, payload: args[1] ? {id: args[1]} : {}};
    if (args.join(' ') === 'verify') return {action: 'runtime.verify', payload: parsed.runtime.root || parsed.runtime.python ? parsed.runtime : {}};
    if (args.join(' ') === 'probe') return {action: 'runtime.probe', payload: {}};
  }
  if (command === 'providers' && ['models', 'voices'].includes(args[0]) && args.length === 2)
    return {action: 'providers.list', payload: {instance: args[1], kind: args[0], ...(parsed.cursor ? {cursor: parsed.cursor} : {})}};
  if (command === 'providers' && args[0] === 'remove' && args.length === 2)
    return {action: 'providers.remove', payload: {instance: args[1]}};
  if (command === 'providers' && ['check', 'observations'].includes(args[0]) && ['llm', 'stt', 'tts'].includes(args[1]) && args.length === 2)
    return {action: `providers.${args[0]}`, payload: {capability: args[1], ...(parsed.text ? {text: parsed.text} : {}), ...(parsed.audio ? {audio_file: resolve(parsed.audio)} : {})}};
  if (command === 'providers' && args[0] === 'test' && ['llm', 'stt', 'tts'].includes(args[1]) && args.length === 2) {
    if (args[1] === 'stt' ? !parsed.audio : !parsed.text) throw new Error('Use --audio for STT, or --text for LLM/TTS');
    return {action: 'providers.test', payload: {capability: args[1], text: parsed.text, audio_file: parsed.audio ? resolve(parsed.audio) : undefined}};
  }
  throw new Error('Invalid command or arguments. Use /help.');
}

export function unsuccessful(result: unknown): boolean {
  if (result && typeof result === 'object' && 'items' in result && 'next_cursor' in result && 'state' in result) return result.state !== 'ready';
  if (result && typeof result === 'object' && 'peers' in result && 'state' in result) return result.state !== 'ready';
  if (Array.isArray(result)) return result.some(row => row?.result === 'error');
  return !!result && typeof result === 'object' && 'state' in result && ['failed', 'interrupted', 'needs-auth', 'changed', 'canceled', 'orphaned'].includes(String(result.state));
}
