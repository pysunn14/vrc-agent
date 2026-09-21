import test from 'node:test';
import assert from 'node:assert/strict';
import { coreAction, parse, splitCommand } from '../src/commands.js';

test('terminal input preserves quoted paths and never interprets shell substitutions', () => {
  assert.deepEqual(splitCommand('profile import "C:\\My Project\\agent.json"'), ['profile', 'import', 'C:\\My Project\\agent.json']);
  assert.deepEqual(splitCommand("providers test llm --text '$(literal text)'"), ['providers', 'test', 'llm', '--text', '$(literal text)']);
  assert.throws(() => splitCommand('"unfinished'), /Unclosed/);
});

test('CLI and interactive commands resolve to the same core operation', () => {
  const action = coreAction(parse(['/providers', 'test', 'tts', '--text', '안녕하세요', '--json']));
  assert.equal(action.action, 'providers.test');
  assert.equal(action.payload.text, '안녕하세요');
  assert.throws(() => coreAction(parse(['providers', 'test', 'tts'])), /--text/);
  assert.throws(() => coreAction(parse(['start', 'agent', 'unexpected'])), /Invalid/);
  assert.throws(() => parse(['settings', '--language', 'xx']), /language/);
});

test('connection-only check and historical observation are explicit CLI operations', () => {
  assert.deepEqual(coreAction(parse(['providers', 'check', 'llm'])), {action: 'providers.check', payload: {capability: 'llm'}});
  assert.deepEqual(coreAction(parse(['providers', 'observations', 'tts'])), {action: 'providers.observations', payload: {capability: 'tts'}});
});
