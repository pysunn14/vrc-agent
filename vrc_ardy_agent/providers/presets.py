"""Setup defaults, not duplicate inference implementations.

IDs also identify persisted credential references. Keep them stable when changing
presentation or request implementation. Defaults never replace saved choices.
"""
PRESETS = {
    'openai': dict(name='OpenAI', protocol='openai-compatible', operations=['llm','stt','tts'],
                   base_url='https://api.openai.com/v1', key='OPENAI_API_KEY', public=True, voices='openai'),
    'openai-compatible': dict(name='OpenAI-compatible API', protocol='openai-compatible',
                              operations=['llm','stt','tts'], base_url='', category='server'),
    'gemini': dict(name='Gemini', protocol='openai-compatible', operations=['llm'],
                   base_url='https://generativelanguage.googleapis.com/v1beta/openai', key='GEMINI_API_KEY', public=True),
    'hermes': dict(name='Hermes', protocol='openai-compatible', operations=['llm'],
                   base_url='http://127.0.0.1:8642/v1', key='API_SERVER_KEY', extension='hermes'),
    'crisperwhisper': dict(name='CrisperWhisper', protocol='openai-compatible', operations=['stt'],
                          base_url='http://127.0.0.1:8002/v1', extension='crisper'),
    # This preset describes our documented compatible wrapper, not api_v2.py.
    # /health is an explicit origin-relative inventory path, never guessed by
    # removing /v1 (which would corrupt reverse proxy prefixes).
    'gpt-sovits': dict(name='GPT-SoVITS (compatible server)', protocol='openai-compatible', operations=['tts'],
                       base_url='http://127.0.0.1:8001/v1', voices_path='/health', streaming_wav=True),
    'elevenlabs': dict(name='ElevenLabs', protocol='elevenlabs', operations=['tts'],
                       base_url='https://api.elevenlabs.io/v1', key='ELEVENLABS_API_KEY', public=True, voices='elevenlabs'),
    'whisper-local': dict(name='Local Whisper (faster-whisper)', protocol='whisper-local', operations=['stt'],category='local'),
}

# Documented built-in choices, not an account inventory or model compatibility
# claim. User voice IDs remain editable. Source: OpenAI create-speech reference.
OPENAI_VOICES = ('alloy','ash','ballad','coral','echo','fable','onyx','nova','sage','shimmer','verse','marin','cedar')
