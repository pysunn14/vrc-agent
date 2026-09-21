import type { Bootstrap, Capability, Connection, AgentProfile, Scalar, Field, Binding, Definition } from './types.js';
import type { CoreClient } from './core.js';
import { fields, SetupCanceled, type Prompter } from './prompts.js';
import { inventoryChoice } from './setup-provider-inventory.js';
import { t } from './i18n.js';
import { chooseConnectionPreset } from './setup-connection-choice.js';

export const defaults = (specs: Record<string, Field>): Record<string, Scalar> => Object.fromEntries(
  Object.entries(specs).filter(([, spec]) => spec.default !== undefined).map(([key, spec]) => [key, spec.default!]));

export async function configureProvider(core: CoreClient, prompts: Prompter, state: Bootstrap, profile: AgentProfile,
  capability: Capability, options: {edit?: boolean; checkpoint?: () => Promise<unknown>} = {}) {
  const language = profile.language, ko = language === 'ko';
  const checkpoint = options.checkpoint ?? (async () => {});
  const eligible = Object.entries(profile.providers).filter(([, c]) => {
    const spec = state.catalog[c.provider], extension = String(c.config.request_extension ?? '');
    return spec?.operations[capability] && (!extension || spec.extensions?.[extension]?.includes(capability));
  });
  let instance = profile.bindings[capability]?.instance;
  if (options.edit || !eligible.some(([id]) => id === instance)) {
    instance = eligible.length ? await prompts.select(`${capability.toUpperCase()} · ${t(language, 'connection')}`,
      [...eligible.map(([value, c]) => ({value, label: `${c.label} · ${state.catalog[c.provider].name}${state.catalog[c.provider].support === 'unknown' ? (ko ? ' · 기능 확인 필요' : ' · capability unverified') : ''}`})),
        {value: '@new', label: t(language, 'add')}], instance || eligible[0]?.[0] || '@new') : '@new';
  }
  if (instance === '@new') {
    const provider = await chooseConnectionPreset(state, prompts, capability, ko);
    const definition = state.catalog[provider];
    instance = `${capability}-${Object.keys(profile.providers).length + 1}`;
    while (profile.providers[instance]) instance += '-new';
    // Put defaults in the draft before the first input. Cancellation at model
    // selection must never discard an address the user has already supplied.
    profile.providers[instance] = {provider, label: `${definition.name} ${capability.toUpperCase()}`,
      config: defaults(definition.fields), deployment: definition.deployment_kinds.includes('embedded')
        ? {kind: 'embedded', host: profile.runner_host} : {kind: 'api'}};
    profile.bindings[capability] = {instance, model: ''};
    await checkpoint();
    const config = profile.providers[instance].config;
    if (definition.fields.base_url) {
      const initial = String(config.base_url ?? '');
      // Cloud endpoints are supplied by the catalog. Local/custom services have
      // a user-selected address; process ownership is configured separately.
      if (definition.setup?.ask_base_url !== false) config.base_url = await prompts.text(definition.fields.base_url.label[language], initial);
      else prompts.note(`API: ${initial}`);
    }
  }
  const connection = profile.providers[instance];
  let definition = state.catalog[connection.provider];
  connection.config = {...defaults(definition.fields), ...connection.config};
  // A resumed draft can contain a provider whose address was never entered.
  if (definition.fields.base_url && !connection.config.base_url)
    connection.config.base_url = await prompts.text(definition.fields.base_url.label[language]);
  const editAddress = async (reason = '') => {
    const keys = reason === 'needs-auth' || reason === 'auth-error' ? ['api_key_env'] : ['base_url', 'api_key_env'];
    for (const [key, spec] of Object.entries(definition.fields).filter(([key]) => keys.includes(key))) {
      connection.config[key] = await prompts.text(spec.label[language], String(connection.config[key] ?? spec.default ?? ''), spec.required);
      if (key === 'api_key_env') {connection.config.credential_source = 'environment'; connection.config.credential_ref = '';}
      await checkpoint();
    }
  };
  const validate = async () => {
    while (true) {
      try {
        connection.config = await core.call<Connection['config']>('connection.validate', {provider: connection.provider, config: connection.config});
        return;
      } catch (error) {
        prompts.note(String(error));
        const choice = await prompts.select(ko ? '연결 설정을 수정해 주세요' : 'Correct the connection settings', [
          {value: 'edit', label: ko ? '입력 수정' : 'Edit input'},
          {value: 'pause', label: ko ? '초안 저장 후 나가기' : 'Save draft and exit'},
        ]);
        if (choice === 'pause') throw new SetupCanceled();
        await editAddress();
      }
    }
  };
  await validate();
  if (connection.config.request_extension) definition = await core.call<Definition>('connection.describe', {provider: connection.provider, config: connection.config});
  const previous = profile.bindings[capability]?.instance === instance ? profile.bindings[capability] : {instance, model: ''};
  const settings: Binding = {...defaults(definition.operations[capability]!), ...previous, instance};
  if (!('vision' in previous)) delete settings.vision;
  profile.bindings[capability] = settings;
  await checkpoint();
  const repair = async (reason: string) => {await editAddress(reason); await validate(); await checkpoint();};
  while (true) {
    settings.model = definition.protocol === 'whisper-local'
      ? await (prompts.path?.bind(prompts) ?? prompts.text.bind(prompts))(ko ? '설치된 Whisper 모델 폴더' : 'Installed Whisper model directory', settings.model, true)
      : await inventoryChoice(core, prompts, connection, capability, 'models', language, settings.model, !!options.edit, repair, checkpoint);
    await checkpoint();
    const selectedConnection = JSON.stringify(connection.config);
    if ('voice' in definition.operations[capability]!) {
      settings.voice = await inventoryChoice(core, prompts, connection, capability, 'voices', language,
        String(settings.voice ?? ''), !!options.edit, repair, checkpoint);
      await checkpoint();
    }
    if (selectedConnection === JSON.stringify(connection.config)) break;
    prompts.note(ko ? '연결이 변경되어 모델 선택을 다시 확인합니다.' : 'Connection changed; checking the model selection again.');
  }
  if ('vision' in definition.operations[capability]! && !('vision' in settings)) {
    settings.vision = await prompts.confirm(ko ? '이 연결로 화면도 볼까요? (지원 여부는 연결 시험에서 확인)' : 'Use this connection to see the screen? (verify support with a connection test)', false);
  }
  const {instance: _instance, ...binding} = settings;
  profile.bindings[capability] = {instance, ...await core.call<{model: string}>('binding.validate', {
    provider: connection.provider, config: connection.config, capability, settings: binding})};
  await checkpoint();
}

/** Advanced editing is opt-in and never interrupts the normal setup path. */
export async function editProviderDetails(core: CoreClient, prompts: Prompter, state: Bootstrap, profile: AgentProfile, capability: Capability) {
  const binding = profile.bindings[capability], connection = profile.providers[binding.instance];
  const definition = state.catalog[connection.provider], language = profile.language;
  connection.label = await prompts.text(t(language, 'name'), connection.label);
  connection.config = await core.call<Connection['config']>('connection.validate', {provider: connection.provider,
    config: {...connection.config, ...await fields(prompts, Object.fromEntries(Object.entries(definition.fields).filter(([key]) => !key.startsWith('credential_'))), connection.config, language)}});
  const resolved = await core.call<Definition>('connection.describe', {provider: connection.provider, config: connection.config});
  const {instance, ...settings} = binding;
  if (!resolved.operations[capability]) throw new Error(language === 'ko' ? '이 확장은 현재 기능을 지원하지 않습니다.' : 'This extension does not support this operation.');
  profile.bindings[capability] = {instance, model: binding.model, ...await fields(prompts, resolved.operations[capability]!, settings, language)};
  const labels: Record<string, string> = {api: t(language, 'api'), external: t(language, 'external'), managed: t(language, 'managed'), embedded: t(language, 'embedded')};
  const kind = await prompts.select(t(language, 'deployment'), definition.deployment_kinds.map(value => ({value, label: labels[value]})), connection.deployment.kind);
  const deployment: Connection['deployment'] = kind === connection.deployment.kind ? {...connection.deployment} : {kind};
  if (kind !== 'api') {
    deployment.host = kind === 'external' ? await prompts.select(t(language, 'host'), [
      ...Object.entries(profile.hosts).map(([value, host]) => ({value, label: `${value} · ${host.address}`})),
      {value: '@remote', label: t(language, 'remote')}], deployment.host ?? profile.runner_host) : profile.runner_host;
    if (deployment.host === '@remote') {
      let id = 'provider-host'; while (id in profile.hosts) id += '-new';
      profile.hosts[id] = {os: await prompts.select(t(language, 'remoteOS'), ['linux', 'windows', 'macos'].map(value => ({value: value as 'linux'|'windows'|'macos', label: value}))), address: await prompts.text(t(language, 'remoteAddress'))};
      deployment.host = id;
    }
    if (kind === 'managed') {
      const command: unknown = JSON.parse(await prompts.text(t(language, 'command'), JSON.stringify(deployment.command ?? ['python', 'server.py'])));
      if (!Array.isArray(command) || !command.length || command.some(v => typeof v !== 'string' || !v)) throw new Error('Expected a JSON array of command arguments');
      deployment.command = command as string[];
      deployment.cwd = await prompts.text(t(language, 'cwd'), deployment.cwd ?? state.cwd);
    }
  }
  connection.deployment = deployment;
}
