import os

from .adapters import ChatAdapter, LocalWhisperAdapter, SpeechAdapter, TranscriptionAdapter
from .catalog import definition, validate_connection, validate_binding
from .http import HttpClient
from .credentials import resolve_key


def build_connection(provider, config, capability, binding, *, environment=None, opener=None):
    config = validate_connection(provider, config)
    binding = validate_binding(provider, capability, binding, config=config)
    spec = definition(provider, config)
    if spec['protocol'] == 'whisper-local': return LocalWhisperAdapter(config, binding)
    http = HttpClient(config, environment=os.environ if environment is None else environment,
                      auth=spec['auth'], opener=opener, api_key=resolve_key(provider, config, environment=environment))
    if capability == 'llm': return ChatAdapter(http, binding, provider_id=provider, extension=spec['extension'])
    if capability == 'stt': return TranscriptionAdapter(http, binding, extension=spec['extension'])
    if capability == 'tts': return SpeechAdapter(http, binding, protocol=spec['protocol'], streaming_wav=config.get('streaming_wav',False))
    raise ValueError(f'unsupported capability: {capability}')


def build_provider(profile, capability, *, environment=None, opener=None):
    connection, binding = profile.resolve(capability)
    return build_connection(connection['provider'], connection['config'], capability,
                            {k:v for k,v in binding.items() if k != 'instance'}, environment=environment, opener=opener)
