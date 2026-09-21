import { spawn } from 'node:child_process';
import * as clack from '@clack/prompts';
import type { CoreClient } from './core.js';
import type { Bootstrap, Language, AgentProfile } from './types.js';
import { SetupCanceled, type Prompter } from './prompts.js';

export interface RuntimeJob {
  id: string | null; state: string; step?: string; completed?: number; total?: number;
  error?: string; detail?: string; bytes_completed?: number; bytes_total?: number;
  result?: {runtime: Partial<AgentProfile['runtime']>; installation_root: string};
}
interface SetupOptions {
  default_root: string; project_dir: string; model_bytes: number;
  host: {os: string; device: string}; recipe: {id: string; device: string; validation: string} | null;
}
const active = new Set(['starting', 'running', 'canceling']);
const pick = (language: Language, ko: string, en: string) => language === 'ko' ? ko : en;

export function progressText(job: RuntimeJob, language: Language): string {
  const stages: Record<string, [string, string]> = {
    preflight: ['환경 확인', 'Preflight'], source: ['ARDY 소스 준비', 'Preparing ARDY source'],
    environment: ['Python 환경 설치', 'Installing Python environment'], models: ['모델 준비', 'Preparing models'],
    verification: ['모델 로딩·동작 생성 검증', 'Verifying model loading and motion'], complete: ['완료', 'Complete'],
  };
  const stage = stages[job.step ?? 'preflight'];
  const label = stage ? stage[language === 'ko' ? 0 : 1] : job.step;
  const done = job.completed ?? 0, total = job.total ?? 1;
  const count = Math.max(0, Math.min(12, Math.round(done / total * 12)));
  const bytes = job.bytes_total ? ` · ${((job.bytes_completed ?? 0) / 1024 ** 3).toFixed(1)}/${(job.bytes_total / 1024 ** 3).toFixed(1)} GiB` : '';
  return `[${'='.repeat(count)}${' '.repeat(12 - count)}] ${done}/${total} ${label}${bytes}`;
}

export async function waitRuntime(core: CoreClient, initial: RuntimeJob, language: Language, show = true): Promise<RuntimeJob> {
  let job = initial, detached = false;
  const spinner = show && process.stdout.isTTY ? clack.spinner({onCancel: () => { detached = true; }}) : undefined;
  const interrupt = () => { detached = true; };
  process.once('SIGINT', interrupt);
  spinner?.start(progressText(job, language));
  try {
    while (active.has(job.state)) {
      if (detached) throw new SetupCanceled(pick(language,
        '설치는 백그라운드에서 계속됩니다. runtime status로 확인하세요.',
        'Setup continues in the background. Use runtime status to observe it.'));
      await new Promise(resolve => setTimeout(resolve, 750));
      job = await core.call<RuntimeJob>('runtime.status', {id: job.id});
      spinner?.message(progressText(job, language));
    }
    spinner?.clear();
    return job;
  } finally {
    spinner?.clear();
    process.removeListener('SIGINT', interrupt);
  }
}


export async function authenticateRuntime(core: CoreClient, id?: string | null): Promise<void> {
  const auth = await core.call<{command: string[]; url: string}>('runtime.auth', id ? {id} : {});
  console.log(auth.url);
  await new Promise<void>((accept, reject) => {
    const child = spawn(auth.command[0], auth.command.slice(1), {stdio: 'inherit', shell: false});
    child.on('error', reject);
    child.on('exit', code => code === 0 ? accept() : reject(new Error(`Hugging Face login exited ${code}`)));
  });
}

export async function setupRuntime(core: CoreClient, state: Bootstrap, profile: AgentProfile, prompts: Prompter, setup: {advanced?: boolean} = {}): Promise<void> {
  const lang = profile.language;
  const pathInput = prompts.path?.bind(prompts) ?? ((message: string, initial?: string) => prompts.text(message, initial));
  const options = await core.call<SetupOptions>('runtime.options');
  const current = await core.call<RuntimeJob>('runtime.status');
  const choices = [
    ...(state.profile || profile.runtime.installation_root ? [{value: 'keep', label: pick(lang, '현재 설정 유지', 'Keep current configuration')}] : []),
    ...(options.recipe ? [{value: 'install', label: pick(lang, '자동 설치 (기본 위치)', 'Install automatically (default location)')},
      {value: 'install-custom', label: pick(lang, '설치 위치·모델 폴더 지정', 'Choose installation / model directories')}] : []),
    {value: 'connect', label: pick(lang, '기존 ARDY 환경 연결', 'Connect an existing ARDY environment')},
    ...(current.id ? [{value: 'resume', label: pick(lang, '이전 설치 확인·재개', 'Inspect / resume previous setup')}] : []),
    {value: 'later', label: pick(lang, '나중에 준비', 'Set up later')},
  ];
  let mode = await prompts.select(pick(lang, 'ARDY 준비 방법', 'ARDY setup'), choices,
    choices.some(c => c.value === 'keep') ? 'keep' : current.id ? 'resume' : options.recipe ? 'install' : 'connect');
  if (mode === 'install-custom') {mode = 'install'; setup = {...setup, advanced: true};}
  if (mode === 'keep' || mode === 'later') {
    if (mode === 'later') prompts.note(pick(lang, '설정은 저장할 수 있지만 ARDY를 준비하기 전에는 에이전트를 실행할 수 없습니다.',
      'You can save configuration, but the agent cannot run until ARDY is ready.'));
    return;
  }
  let job: RuntimeJob;
  if (mode === 'resume') {
    prompts.note(`${current.id} · ${current.state}\n${current.error ?? ''}`);
    job = active.has(current.state) || current.state === 'complete' ? current
      : await core.call<RuntimeJob>('runtime.resume', {id: current.id});
  } else {
    const request: Record<string, unknown> = {mode, base_dir: process.cwd(), project_dir: options.project_dir};
    const root = mode === 'install' && !setup.advanced ? options.default_root : await pathInput(pick(lang, mode === 'install' ? 'ARDY 설치 폴더' : '기존 ARDY 폴더 또는 Python 실행 파일',
      mode === 'install' ? 'ARDY installation directory' : 'Existing ARDY directory or Python executable'),
      mode === 'install' ? options.default_root : profile.runtime.installation_root ?? '', mode === 'install');
    request.root = root;
    if (mode === 'install') {
      if (setup.advanced && await prompts.confirm(pick(lang, '모델을 별도 폴더에 저장할까요?', 'Store models in a different directory?'), false))
        request.models_dir = await pathInput(pick(lang, '모델 저장 폴더', 'Model directory'), undefined, true);
      request.device = options.recipe!.device;
    } else {
      try { request.python = (await core.call<{python: string}>('runtime.discover', {root, base_dir: process.cwd()})).python; }
      catch (error) {
        prompts.note(String(error));
        request.python = await pathInput(pick(lang, 'ARDY 환경의 Python 실행 파일', 'Python executable in the ARDY environment'));
      }
      request.device = !setup.advanced ? options.host.device : await prompts.select(pick(lang, 'ARDY 실행 장치', 'ARDY device'),
        [options.host.device, 'cpu'].filter((v, i, a) => a.indexOf(v) === i).map(value => ({value, label: value})), options.host.device);
      request.checkpoints_dir = !setup.advanced ? profile.runtime.checkpoints_dir : await prompts.text(pick(lang, '기존 체크포인트 폴더 (비우면 기존 캐시)', 'Existing checkpoint directory (empty: existing cache)'), profile.runtime.checkpoints_dir, false);
      request.hf_cache_dir = !setup.advanced ? profile.runtime.hf_cache_dir : await prompts.text(pick(lang, '기존 Hugging Face 캐시 폴더 (선택)', 'Existing Hugging Face cache directory (optional)'), profile.runtime.hf_cache_dir, false);
    }
    const plan = await core.call<{paths: {root: string; models: string}; model_bytes: number; recipe?: {validation: string}; checks: {result: string; detail: string}[]}>('runtime.plan', request);
    prompts.note([`${pick(lang, '위치', 'Location')}: ${plan.paths.root}`,
      ...(mode === 'install' ? [`${pick(lang, '모델', 'Models')}: ${plan.paths.models} · ${(plan.model_bytes / 1024 ** 3).toFixed(1)} GiB`,
        pick(lang, '로컬 문장 인코더에 Hugging Face의 Llama 접근 권한이 필요합니다. 기존 로그인 또는 HF_TOKEN을 사용합니다.',
          'The local text encoder requires Hugging Face Llama access. Uses your existing login or HF_TOKEN.')] : []),
      ...plan.checks.filter(row => row.result === 'error').map(row => row.detail),
      ...(plan.recipe?.validation === 'hardware-test-pending' ? [pick(lang, '이 운영체제의 새 설치 구성은 실제 장비 검증 전입니다.', 'Fresh installation on this OS is pending hardware validation.')] : []),
    ].join('\n'), 'ARDY');
    if (plan.checks.some(row => row.result === 'error')) throw new Error(pick(lang, '설치 전 검사 오류를 해결해 주세요.', 'Resolve preflight errors before installing.'));
    if (!await prompts.confirm(pick(lang, '환경 준비와 실제 동작 생성 검증을 시작할까요?', 'Prepare the environment and verify motion generation?'))) throw new SetupCanceled();
    job = await core.call<RuntimeJob>('runtime.setup', request);
  }
  let finished = await waitRuntime(core, job, lang);
  while (finished.state === 'needs-auth') {
    prompts.note(`${finished.error}\nhttps://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct`);
    if (!await prompts.confirm(pick(lang, 'Hugging Face에 로그인한 뒤 재개할까요?', 'Log in to Hugging Face and resume?'))) throw new SetupCanceled();
    await authenticateRuntime(core, finished.id);
    job = await core.call<RuntimeJob>('runtime.resume', {id: finished.id});
    finished = await waitRuntime(core, job, lang);
  }
  if (finished.state !== 'complete') throw new Error(`${finished.error ?? finished.state}\nruntime logs · runtime resume`);
  const runtime = await core.call<Partial<AgentProfile['runtime']>>('runtime.result', {id: finished.id});
  Object.assign(profile.runtime, runtime);
  prompts.note(pick(lang, 'ARDY 추론 검증을 통과했습니다. 아바타 동작 파일은 다음 단계에서 설정합니다.',
    'ARDY inference passed. Configure avatar motion assets in the next step.'));
}
