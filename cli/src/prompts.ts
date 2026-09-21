import * as clack from '@clack/prompts';
import type { Field, Language, Scalar } from './types.js';
import { pathOptions, pickPath, validatePath } from './path-input.js';
import { t } from './i18n.js';

export class SetupCanceled extends Error {}
export function answer<T>(value: T): Exclude<T, symbol> {
  if (clack.isCancel(value)) throw new SetupCanceled();
  return value as Exclude<T, symbol>;
}
export interface Prompter {
  password?(message: string): Promise<string>;
  text(message: string, initial?: string, required?: boolean): Promise<string>;
  path?(message: string, initial?: string, directory?: boolean): Promise<string>;
  select<T extends string>(message: string, options: {value: T; label: string}[], initial?: T): Promise<T>;
  search?<T extends string>(message: string, options: {value: T; label: string}[], initial?: T): Promise<T>;
  confirm(message: string, initial?: boolean): Promise<boolean>;
  note(message: string, title?: string): void;
}
export function prompter(language: () => Language): Prompter {
  return {
    async password(message) {
      return answer(await clack.password({message, mask: '*',
        validate: value => !value?.trim() ? t(language(), 'required') : undefined})).trim();
    },
    async text(message, initial = '', required = true) {
      return answer(await clack.text({message, initialValue: initial,
        validate: value => required && !value?.trim() ? t(language(), 'required') : undefined})).trim();
    },
    async path(message, initial, directory = false) {
      return pickPath(async current => answer(await clack.autocomplete<string>({message, initialUserInput: current,
        completeOnTab: true, filter: () => true,
        options() { return pathOptions(this.userInput, directory, language()); },
        validate: value => typeof value !== 'string' || !value.trim() ? t(language(), 'required') : validatePath(value, directory, language())})).trim(), initial, directory);
    },
    async select(message, options, initial) {
      return answer(await clack.select<string>({message, options, initialValue: initial,
        showInstructions: language() !== 'ko'})) as typeof initial & string;
    },
    async search(message, options, initial) {
      return answer(await clack.autocomplete<string>({message, options, initialValue: initial})) as typeof initial & string;
    },
    async confirm(message, initial = true) { return answer(await clack.confirm({message, initialValue: initial,
      active: language() === 'ko' ? '예' : 'Yes', inactive: language() === 'ko' ? '아니요' : 'No'})); },
    note: (message, title) => clack.note(message, title),
  };
}

export async function fields(prompts: Prompter, definitions: Record<string, Field>, values: Record<string, Scalar>, language: Language) {
  const result: Record<string, Scalar> = {};
  for (const [key, spec] of Object.entries(definitions)) {
    const initial = values[key] ?? spec.default;
    if (spec.type === 'boolean') result[key] = await prompts.confirm(spec.label[language], initial === true);
    else if (spec.choices) result[key] = await prompts.select(spec.label[language], spec.choices.map(value => ({value, label: value || (language === 'ko' ? '없음' : 'None')})), String(initial));
    else {
      const raw = await prompts.text(spec.label[language], initial === undefined ? '' : String(initial), spec.required);
      result[key] = spec.type === 'number' ? Number(raw) : raw;
    }
  }
  return result;
}
