"""Transport contracts independent of service names and installation ownership."""
from copy import deepcopy

def field(en, ko, *, default=None, required=False, choices=None, kind="string"):
    result = {"label": {"en": en, "ko": ko}, "type": kind, "required": required}
    if default is not None: result["default"] = default
    if choices: result["choices"] = choices
    return result


MODEL = field("Model ID or local model path", "모델 ID 또는 로컬 모델 경로", required=True)
LANGUAGE = field("Transcription language (empty = detect)", "전사 언어 (비우면 자동 감지)", default="")
VOICE = field("Voice ID", "목소리 ID", required=True)
VISION = field("This model accepts images", "이 모델은 이미지를 입력받을 수 있음", default=False, kind="boolean")
REASONING = field("Reasoning effort (empty = model default)", "추론 수준 (비우면 모델 기본값)", default="")
KEY = field("API key environment variable (not the key itself)", "API 키를 담은 환경변수 이름 (키 자체가 아님)", default="", kind="env")
TIMEOUT = field("Request timeout in seconds", "요청 제한 시간 (초)", default=90, kind="number")
LLM = {"model": MODEL, "vision": VISION, "reasoning_effort": REASONING}
STT = {"model": MODEL, "language": LANGUAGE}
TTS = {"model": MODEL, "voice": VOICE}



PROTOCOLS = {
    'openai-compatible': {'auth':'bearer', 'operations':{'llm':LLM, 'stt':STT, 'tts':TTS}},
    'elevenlabs': {'auth':'elevenlabs', 'operations':{'tts':TTS}},
    'whisper-local': {'auth':None, 'operations':{'stt':STT}},
}


def protocol_definition(name):
    try: return deepcopy(PROTOCOLS[name])
    except KeyError: raise ValueError(f'unsupported connection protocol: {name}') from None
