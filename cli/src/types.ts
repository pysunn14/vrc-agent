export type Language = 'en' | 'ko';
export type Capability = 'llm' | 'stt' | 'tts';
export type OS = 'macos' | 'linux' | 'windows';
export type Scalar = string | number | boolean;
export interface Field {
  label: Record<Language, string>; type: string; required: boolean;
  default?: Scalar; choices?: string[];
}
export interface Definition {
  setup?: {ask_base_url: boolean; category?: string};
  name: string; protocol: string; extension?: string; support?: string; extensions?: Record<string, string[]>; deployment_kinds: string[];
  fields: Record<string, Field>; operations: Partial<Record<Capability, Record<string, Field>>>;
}
export interface Connection {
  provider: string; label: string; config: Record<string, Scalar>;
  deployment: {kind: string; host?: string; command?: string[]; cwd?: string; health_url?: string};
}
export interface Binding { instance: string; model: string; [key: string]: Scalar }
export type Role = 'agent' | 'bridge' | 'combined';
export interface BaseProfile {
  version: 3; language: Language; role: Role;
  hosts: Record<string, {os: OS; address: string}>; runner_host: string; game_host: string;
  connection: {stream_port: number; agent_port: number; bridge_port: number};
}
export interface AgentSettings {
  providers: Record<string, Connection>; bindings: Record<Capability, Binding>;
  avatar: {rig: string; base_pose: string; hmd_base: number[]; face_channels: Record<string, string>;
    calibration?: {signature: string; tracking: 'active'}};
  behaviors: Record<string, {source: 'clip' | 'ardy' | 'locomotion'; description?: string;
    frames?: string; face_tracks?: Record<string, string>; avatar_signature?: string; asset_signature?: string;
    prompt?: string; velocity?: number[]; duration_seconds?: number}>;
  autonomy: {enabled: boolean; interval_seconds: number; behaviors: string[]};
  runtime: {python: string; project_dir: string; device: string;
    model?: string; checkpoints_dir?: string; hf_cache_dir?: string; text_encoder_mode?: string; installation_root?: string};
}
export interface BridgeSettings {
  python: string; project_dir: string; virtual_mic_device: string;
  audio_source?: string; vrchat_pid?: number; vrchat_user_name?: string;
  microphone_device?: string; process_audio_helper?: string;
  body_output?: 'vrchat-osc' | 'vmt';
}
export type AgentOnlyProfile = BaseProfile & AgentSettings & {role: 'agent'};
export type CombinedProfile = BaseProfile & AgentSettings & {role: 'combined'; bridge: BridgeSettings};
export type AgentProfile = AgentOnlyProfile | CombinedProfile;
export type BridgeProfile = BaseProfile & {role: 'bridge'; bridge: BridgeSettings & {avatar_rig: string; hmd_base: number[]}};
export type Profile = AgentProfile | BridgeProfile;
export function hasAgent(profile: Profile): profile is AgentProfile { return profile.role !== 'bridge'; }
export function hasBridge(profile: Profile): profile is BridgeProfile | CombinedProfile { return profile.role !== 'agent'; }
export interface DiscoveredHost {id: string; name: string; os: string; address: string; addresses?: string[]; online: boolean}
export interface HostDiscovery {state: 'ready' | 'unavailable' | 'needs-login' | 'stopped' | 'error'; detail: string;
  self: DiscoveredHost | null; peers: DiscoveredHost[]}
export interface Saved {profile: Profile; revision: string}
export interface Bootstrap {
  protocol: number; catalog: Record<string, Definition>; profile: Profile | null; revision: string | null;
  draft: {profile: Profile; revision: string | null; completed?: string[]} | null;
  path: string; os: OS; python: string; cwd: string;
}
export interface CoreEvent {event: string; [key: string]: unknown}
export interface Options {config?: string; stateDir?: string; python?: string; language?: Language; json?: boolean}
