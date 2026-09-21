#!/usr/bin/env node
import { readFile, realpath } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import * as clack from '@clack/prompts';
import { CoreClient } from './core.js';
import { parse, coreAction, splitCommand, unsuccessful, type Parsed } from './commands.js';
import { authenticateRuntime, setupRuntime, waitRuntime, type RuntimeJob } from './setup-runtime.js';
import { onboard, blankProfile } from './onboard.js';
import { prompter, SetupCanceled } from './prompts.js';
import { banner, readCommand, render } from './terminal.js';
import { t } from './i18n.js';
import {calibrateAvatar} from './setup-calibration.js';
import {createAvatarSettings} from './setup-avatar-package.js';
import { registerAvatarSettings } from './setup-avatar-package.js';
import { hasAgent } from './types.js';
import type { Bootstrap, Language, Options } from './types.js';

export const help = `VRC AGENT

  onboard | configure            Initial setup / edit configuration
  status                         Observe services
  start <service>                Start an owned local service
  stop <service>                 Stop an owned local service
  logs <service>                 Read recent process output
  doctor [--offline]             Diagnose configuration and connections
  providers                      Show connections and bindings
  providers models <instance>    Query available model IDs
  providers voices <instance>    Query voices (--cursor TOKEN for next page)
  providers remove <instance>    Remove an unused connection
  providers check <llm|stt|tts>    Test a connection without ARDY / VRChat
  providers observations <llm|stt|tts>  Show historical test results
  providers test llm --text TEXT  Test a decision without executing it
  providers test stt --audio WAV  Test a 16 kHz mono WAV transcription
  providers test tts --text TEXT  Test speech format without playback
  hosts discover                 Discover Tailscale computers
  connection info                Show peer addresses and shared ports
  bridge command                 Print the local Windows bridge command
  runtime setup                  Install ARDY / connect an existing environment
  runtime status [id]            Observe setup progress and readiness
  runtime resume [id]             Resume setup after interruption or authentication
  runtime auth [id]               Log in to Hugging Face for model downloads
  runtime cancel [id]             Cancel setup, retaining resumable files
  runtime logs [id]               Read setup logs
  runtime verify                 Load ARDY and generate motion without VRChat
  runtime plan --root DIR         Preview a managed installation
  runtime probe                  Inspect the configured Python / GPU environment
  avatars calibrate              Preview, adjust and save an avatar pose in VRChat
  avatars list                   List prepared avatar packages and file problems
  avatars import FILE            Register a portable avatar package
  avatars register               Package current visually verified avatar settings
  avatars export KEY FILE        Export a registered package for another machine
  behaviors                      List available actions and asset problems
  snapshot                       Record versions and avatar / behavior hashes
  profile export                 Print the active profile
  profile import FILE            Validate and save a profile
  settings [en|ko]                Display or change language

  --config FILE  --state-dir DIR  --python EXECUTABLE  --language en|ko  --json
  runtime setup: --root DIR | --existing DIR  --models-dir DIR  --runtime-python FILE
                 --device mps|cuda|cpu  --checkpoints-dir DIR  --hf-cache-dir DIR  --wait

  Run without a command for the Pi terminal. Use /quit to leave it.`;

export const helpKo = `VRC AGENT

  onboard | configure            초기 설정 / 설정 변경
  status                         서비스 상태 확인
  start <service>                로컬 서비스 실행
  stop <service>                 러너가 시작한 서비스 종료
  logs <service>                 최근 실행 로그
  doctor [--offline]             구성과 연결 진단
  providers                      등록한 제공자와 기능별 선택
  providers models <instance>    모델 목록 조회
  providers voices <instance>    목소리 목록 조회 (--cursor 토큰으로 다음 페이지)
  providers remove <instance>    사용하지 않는 연결 삭제
  providers check <llm|stt|tts>    ARDY·VRChat 없이 연결 시험
  providers observations <llm|stt|tts>  과거 시험 결과 조회
  providers test llm --text TEXT  대화 응답만 검증
  providers test stt --audio WAV  16 kHz 모노 WAV 전사 검증
  providers test tts --text TEXT  음성 합성만 검증 (재생하지 않음)
  hosts discover                 Tailscale 장비 검색
  connection info                두 장비의 주소와 공통 포트 확인
  bridge command                 이 Windows의 브리지 실행 명령 확인
  runtime setup                  ARDY 설치 / 기존 환경 연결
  runtime status [id]             설치 진행 상황과 준비 상태 확인
  runtime resume [id]             중단·인증 대기 후 재개
  runtime auth [id]               모델 다운로드용 Hugging Face 로그인
  runtime cancel [id]             설치 중단 (재개할 파일 유지)
  runtime logs [id]               설치 로그 확인
  runtime verify                 VRChat 없이 모델 로딩·동작 생성 검증
  runtime plan --root DIR         설치 계획 확인
  runtime probe                  실행 환경의 Python·GPU 확인
  avatars calibrate              VRChat에서 자세 확인·조정·저장
  avatars list                   준비된 아바타 설정과 파일 문제 확인
  avatars import FILE            아바타 설정 묶음 등록
  avatars register               현재 검증한 아바타 설정을 묶음으로 등록
  avatars export KEY FILE        등록한 설정 묶음을 파일로 내보내기
  behaviors                      사용 가능한 행동과 파일 문제 확인
  snapshot                       버전과 아바타·행동 파일 해시 기록
  profile export                 실행 설정 출력
  profile import FILE            설정 파일 검증 후 저장
  settings [en|ko]                언어 확인·변경

  --config FILE  --state-dir DIR  --python EXECUTABLE  --language en|ko  --json
  runtime setup: --root DIR | --existing DIR  --models-dir DIR  --runtime-python FILE
                 --device mps|cuda|cpu  --checkpoints-dir DIR  --hf-cache-dir DIR  --wait

  명령 없이 실행하면 Pi 터미널을 엽니다. /quit로 나갑니다.`;

async function execute(parsed: Parsed): Promise<number> {
  if (parsed.command === 'help') { console.log(parsed.options.language === 'ko' ? helpKo : help); return 0; }
  let language: Language = parsed.options.language ?? 'en';
  let spinner: ReturnType<typeof clack.spinner> | undefined;
  const core = new CoreClient(parsed.options, event => {
    if (event.event === 'progress' && spinner) {
      const done = Number(event.completed), total = Number(event.total);
      const count = Math.round(done / total * 12);
      spinner.message(`${t(language, 'progress')} [${'='.repeat(count)}${' '.repeat(12-count)}] ${done}/${total}`);
    }
  });
  const state = await core.call<Bootstrap>('bootstrap');
  language = parsed.options.language ?? state.profile?.language ?? 'en';
  if (['onboard', 'configure'].includes(parsed.command)) {
    if (!process.stdin.isTTY || !process.stdout.isTTY) throw new Error(t(language, 'noTTY'));
    const saved = await onboard(core, state);
    return saved ? 0 : 130;
  }
  if (parsed.command === 'runtime' && parsed.args.join(' ') === 'setup' && !Object.keys(parsed.runtime).some(key => !['mode', 'base_dir'].includes(key)) && process.stdin.isTTY && !parsed.options.json) {
    const profile = structuredClone(state.profile ?? state.draft?.profile ?? blankProfile(state, language));
    if (!hasAgent(profile)) throw new Error(language === 'ko' ? 'ARDY 준비는 에이전트 역할의 컴퓨터에서 실행하세요.' : 'ARDY setup requires an agent role on this computer.');
    await setupRuntime(core, state, profile, prompter(() => language));
    if (state.profile) await core.call('profile.commit', {profile, revision: state.revision});
    else await core.call('draft.save', {profile, revision: state.revision, completed: [...new Set([...(state.draft?.completed ?? []), 'runtime'])]});
    console.log(state.profile ? t(language, 'saved') : language === 'ko'
      ? 'ARDY 설정을 초안에 저장했습니다. vrc-agent onboard로 나머지 설정을 완료하세요.'
      : 'ARDY settings saved to the draft. Run vrc-agent onboard to finish configuration.');
    return 0;
  }
  if (parsed.command === 'avatars' && parsed.args.join(' ') === 'calibrate') {
    if (!process.stdin.isTTY || !process.stdout.isTTY) throw new Error(t(language, 'noTTY'));
    const profile = structuredClone(state.draft?.profile ?? state.profile);
    if (!profile || !hasAgent(profile)) throw new Error(language === 'ko' ? 'onboard에서 Windows 연결 주소를 먼저 설정하세요.' : 'Configure your Windows connection with onboard first.');
    const prompts = prompter(() => language);
    try {
      if (!profile.avatar.rig) await createAvatarSettings(core, profile, prompts);
      if (await calibrateAvatar(core, profile, prompts) && await prompts.confirm(language === 'ko' ? '이 자세를 아바타 설정 묶음으로 등록할까요?' : 'Register this pose as an avatar package?', true))
        await registerAvatarSettings(core, profile, prompts);
    } finally {
      await core.call('draft.save', {profile, revision: state.revision, completed: state.draft?.completed ?? []});
    }
    console.log(language === 'ko' ? '교정 설정을 초안에 보관했습니다. configure에서 확인하고 저장하세요.' : 'Calibration saved to the draft. Review and save with configure.');
    return 0;
  }
  if (parsed.command === 'avatars' && parsed.args.join(' ') === 'register') {
    if (!process.stdin.isTTY || !process.stdout.isTTY) throw new Error(t(language, 'noTTY'));
    const profile = state.draft?.profile ?? state.profile;
    if (!profile || !hasAgent(profile)) throw new Error(language === 'ko' ? 'configure에서 아바타 설정과 실제 동작 확인을 먼저 완료하세요.' : 'Complete avatar configuration and visual verification with configure first.');
    await registerAvatarSettings(core, profile, prompter(() => language));
    return 0;
  }
  if (parsed.command === 'profile') {
    if (parsed.args.length === 1 && parsed.args[0] === 'export') {
      console.log(JSON.stringify(state.profile, null, 2)); return 0;
    }
    if (parsed.args.length === 2 && parsed.args[0] === 'import') {
      const profile: unknown = JSON.parse(await readFile(resolve(parsed.args[1]), 'utf8'));
      const saved = await core.call('profile.commit', {profile, revision: state.revision});
      console.log(parsed.options.json ? JSON.stringify(saved) : t(language, 'saved')); return 0;
    }
    throw new Error('Use profile export or profile import FILE');
  }
  if (['start', 'stop', 'logs'].includes(parsed.command) && !parsed.args.length && process.stdin.isTTY) {
    const rows = (await core.call<{id: string; state: string; local: boolean}[]>('status')).filter(row => row.local);
    if (!rows.length) throw new Error(language === 'ko' ? '이 컴퓨터에서 관리하는 서비스가 없습니다.' : 'No services are managed on this computer.');
    parsed.args = [await prompter(() => language).select(t(language, 'service'), rows.map(row => ({value: row.id, label: `${row.id} · ${row.state}`})))];
  }
  const {action, payload} = coreAction(parsed);
  if (action === 'runtime.auth' && process.stdin.isTTY && !parsed.options.json) {
    await authenticateRuntime(core, payload.id as string | undefined); return 0;
  }
  if (process.stdout.isTTY && !parsed.options.json) { spinner = clack.spinner(); spinner.start(t(language, 'progress')); }
  try {
    let result = await core.call(action, payload);
    if (parsed.wait && ['runtime.setup', 'runtime.resume', 'runtime.verify'].includes(action)) {
      spinner?.clear(); spinner = undefined;
      result = await waitRuntime(core, result as RuntimeJob, language, !parsed.options.json);
    }
    spinner?.clear();
    if (action === 'settings' && !parsed.options.json) {
      const saved = result as {language?: string; profile?: {language: string}};
      console.log(`${t(language, 'language')}: ${saved.profile?.language ?? saved.language}`);
    } else console.log(parsed.options.json ? JSON.stringify(result, null, 2) : render(result, language));
    return unsuccessful(result) ? 1 : 0;
  } catch (error) { spinner?.error(t(language, 'failed')); throw error; }
}

async function consoleSession(options: Options): Promise<number> {
  const core = new CoreClient(options);
  let state = await core.call<Bootstrap>('bootstrap');
  let language = options.language ?? state.profile?.language ?? 'en';
  console.log(banner(language));
  if (!state.profile) {
    await onboard(core, state);
    state = await core.call<Bootstrap>('bootstrap');
    language = options.language ?? state.profile?.language ?? language;
  }
  const history: string[] = [];
  while (true) {
    const line = await readCommand(language, history);
    if (line === null || ['/quit', 'quit', '/exit', 'exit'].includes(line.trim())) break;
    if (!line.trim()) continue;
    history.push(line);
    try {
      const parsed = parse(splitCommand(line));
      parsed.options = Object.fromEntries(Object.entries({...options, ...Object.fromEntries(
        Object.entries(parsed.options).filter(([, value]) => value !== undefined))}));
      parsed.options.language ??= language;
      await execute(parsed);
      if (['settings', 'onboard', 'configure'].includes(parsed.command)) {
        state = await core.call<Bootstrap>('bootstrap');
        language = options.language ?? state.profile?.language ?? language;
      }
    } catch (error) {
      if (!(error instanceof SetupCanceled)) console.error(error instanceof Error ? error.message : String(error));
    }
  }
  console.log(t(language, 'bye'));
  return 0;
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  try {
    const parsed = parse(argv);
    if (parsed.command) return await execute(parsed);
    if (!process.stdin.isTTY || !process.stdout.isTTY) { console.log(parsed.options.language === 'ko' ? helpKo : help); return 0; }
    return await consoleSession(parsed.options);
  } catch (error) {
    if (error instanceof SetupCanceled) { if (error.message) console.error(error.message); return 130; }
    const message = error instanceof Error ? error.message : String(error);
    if (argv.includes('--json')) console.log(JSON.stringify({error: message}));
    else console.error(message);
    return 2;
  }
}

// npm links the executable; Node resolves the module URL but preserves the link in argv.
if (process.argv[1] && import.meta.url === pathToFileURL(await realpath(process.argv[1])).href) process.exitCode = await main();
