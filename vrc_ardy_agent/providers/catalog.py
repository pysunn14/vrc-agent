"""Compose setup presets with protocol contracts for every core/UI consumer."""
from copy import deepcopy
import math
import re
from urllib.parse import urlsplit, unquote
from .protocols import field, TIMEOUT, protocol_definition
from .presets import PRESETS
from .extensions import EXTENSIONS


def definition(provider_id, config=None):
    try: preset = deepcopy(PRESETS[provider_id])
    except (KeyError, TypeError): raise ValueError(f'unknown provider: {provider_id}') from None
    protocol = preset['protocol']
    contract = protocol_definition(protocol)
    extension = (config or {}).get('request_extension', preset.get('extension',''))
    if extension not in EXTENSIONS or (extension and protocol != 'openai-compatible'):
        raise ValueError('unsupported request extension')
    operations = {key:contract['operations'][key] for key in preset['operations']}
    if extension:
        operations = {key:fields | EXTENSIONS[extension][key] for key,fields in operations.items() if key in EXTENSIONS[extension]}
    if protocol == 'whisper-local':
        fields = {'device':field('Device','실행 장치',default='cpu',choices=['cpu','cuda']),
                  'compute_type':field('Compute type','연산 정밀도',default='int8',choices=['int8','float16','float32','int8_float16'])}
    else:
        fields = {'base_url':field('API base URL','API 기본 주소',default=preset['base_url'],required=True,kind='url'),
                  'api_key_env':field('API key environment variable','API 키 환경변수 이름',default=preset.get('key',''),required=bool(preset.get('key')),kind='env'),
                  'timeout_seconds':TIMEOUT,
                  'credential_source':field('Credential source','인증 저장 위치',default='environment',choices=['environment','keyring','hermes']),
                  'credential_ref':field('Credential reference','인증 참조',default='')}
        if 'tts' in preset['operations']:
            fields['streaming_wav'] = field('Server returns streaming WAV with unknown length','길이 미정 스트리밍 WAV 응답',default=preset.get('streaming_wav',False),kind='boolean')
            fields['voices_path'] = field('Voice inventory path (relative or origin-relative)','목소리 목록 경로 (선택)',default=preset.get('voices_path',''),kind='path')
        if provider_id == 'openai-compatible':
            fields['request_extension'] = field('Request extension','서버별 요청 확장',default='',choices=['','hermes','crisper'])
    return {'name':preset['name'], 'protocol':protocol, 'extension':extension, 'auth':contract['auth'],
            'operations':operations, 'fields':fields,
            'inventory':{'voices':preset.get('voices','path')},
            'support':'unknown' if preset.get('category') == 'server' else 'declared',
            'extensions':{name:list(ops) for name,ops in EXTENSIONS.items() if name},
            'deployment_kinds':['embedded'] if protocol == 'whisper-local' else ['api','external','managed'],
            'setup':{'ask_base_url':not preset.get('public',False),'category':preset.get('category','service')}}


def catalog(): return {key:definition(key) for key in PRESETS}


def validate_fields(fields, values, *, context):
    if not isinstance(values, dict) or set(values) - set(fields):
        raise ValueError(f"{context}: unknown or invalid fields")
    result = {}
    for name, spec in fields.items():
        if name in values and values[name] is None:
            raise ValueError(f"{context}.{name}: null is not a configured value")
        value = values.get(name, spec.get("default"))
        if value is None or (value == "" and spec["type"] in ("string", "env", "url", "path")):
            if spec["required"]: raise ValueError(f"{context}.{name} is required")
            if value is not None: result[name] = value
            continue
        kind = spec["type"]
        if kind == "boolean":
            valid = isinstance(value, bool)
        elif kind == "number":
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 < value <= 600
        else:
            valid = isinstance(value, str) and bool(value.strip()) and len(value) <= 4096
        if not valid: raise ValueError(f"{context}.{name}: invalid {kind}")
        if isinstance(value, str): value = value.strip()
        if spec.get("choices") and value not in spec["choices"]:
            raise ValueError(f"{context}.{name}: choose one of {spec['choices']}")
        if kind == "url":
            parsed = urlsplit(value)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment):
                raise ValueError(f"{context}.{name}: expected HTTP(S) URL without credentials, query or fragment")
            _ = parsed.port
            value = value.rstrip("/")
        if kind == "env" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"{context}.{name}: expected environment variable name")
        result[name] = value
    return result



def validate_connection(provider_id, config):
    result = validate_fields(definition(provider_id)['fields'], config, context=provider_id)
    path = result.get('voices_path','')
    if path:
        decoded = unquote(path)
        if (urlsplit(path).scheme or urlsplit(path).netloc or '?' in path or '#' in path
                or decoded.startswith('//') or '\\' in decoded or any(ord(c)<32 for c in decoded)
                or any(part in ('.','..') for part in decoded.split('/'))):
            raise ValueError('voice inventory path must stay on the configured server without traversal or query')
    definition(provider_id,result)
    return result


def validate_binding(provider_id, capability, values, *, config=None):
    fields = definition(provider_id, config)['operations'].get(capability)
    if fields is None: raise ValueError(f'{provider_id} does not support {capability}')
    return validate_fields(fields, values, context=f'{provider_id}.{capability}')
