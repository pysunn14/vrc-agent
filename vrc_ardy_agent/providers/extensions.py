"""Explicit, typed differences within the compatible HTTP contract."""
from .protocols import field

EXTENSIONS = {
    '': {},
    'hermes': {'llm': {'upstream_provider':field('Upstream provider override (empty = server routing)',
                '상위 제공자 재지정 (비우면 서버 설정 사용)',default='')}},
    'crisper': {'stt': {'mode':field('Transcription mode','전사 방식',default='intended',choices=['intended','verbatim'])}},
}


def chat_payload(binding, messages, extension=''):
    payload = {'model':binding['model'], 'stream':False, 'messages':messages}
    effort = binding.get('reasoning_effort')
    if extension == 'hermes':
        if binding.get('upstream_provider'): payload['provider'] = binding['upstream_provider']
        if effort: payload['model_options'] = {'reasoning_effort':effort}
    elif effort: payload['reasoning_effort'] = effort
    return payload


def transcription_fields(binding, extension=''):
    result = {'model':binding['model'], 'response_format':'json'}
    if binding.get('language'): result['language'] = binding['language']
    if extension == 'crisper': result['mode'] = binding['mode']
    return result
