"""Connection-only smoke tests. No avatar, game, microphone or ARDY imports."""
import base64
from http.client import HTTPException
from io import BytesIO
from pathlib import Path
import struct
import time
import wave
import zlib
from urllib.error import URLError

from .factory import build_connection
from .credentials import CredentialError
from .adapters import validate_wav
from .extensions import chat_payload
from .http import ProviderHTTPError


def synthetic_image():
    def chunk(kind, body):
        return struct.pack('>I',len(body))+kind+body+struct.pack('>I',zlib.crc32(kind+body))
    raw = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR',struct.pack('>IIBBBBB',8,8,8,2,0,0,0))
    raw += chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*8)*8))+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(raw).decode()


def probe_connection(provider, config, capability, settings, *, text=None, audio_file=None, environment=None, opener=None):
    started=time.monotonic()
    checks={}
    try:
        adapter=build_connection(provider,config,capability,settings,environment=environment,opener=opener)
        if capability=='llm':
            def complete(content):
                data=adapter.http.json('chat/completions',payload=chat_payload(adapter.binding,
                    [{'role':'user','content':content}],adapter.extension),limit=128*1024)
                try: answer=data['choices'][0]['message']['content']
                except (KeyError,IndexError,TypeError): raise ValueError('missing text response') from None
                if not isinstance(answer,str) or not answer.strip(): raise ValueError('empty text response')
            complete(text or 'Reply with OK only. Do not use tools.')
            checks['text']='success'
            if adapter.binding.get('vision'):
                complete([{'type':'text','text':'Name the color of this synthetic test image. Do not use tools.'},
                          {'type':'image_url','image_url':{'url':synthetic_image()}}])
                checks['image_input']='success'
        elif capability=='stt':
            if not audio_file and adapter.binding.get('language', '') not in ('', 'en'):
                return {'state':'failed','reason':'sample-language-mismatch','checks':{},'executed_in_game':False}
            path=Path(audio_file) if audio_file else Path(__file__).with_name('assets')/'speech-sample.wav'
            if path.stat().st_size>32*1024*1024: raise ValueError('audio sample exceeds limit')
            raw=validate_wav(path.read_bytes())
            with wave.open(BytesIO(raw)) as wav:
                if wav.getframerate()!=16000 or wav.getnchannels()!=1: raise ValueError('audio must be 16 kHz mono PCM WAV')
                pcm=wav.readframes(wav.getnframes())
            adapter.transcribe_pcm(pcm)
            checks['transcription']='success'
        elif capability=='tts':
            raw=validate_wav(adapter.synthesize(text or 'This is a voice connection test.'))
            with wave.open(BytesIO(raw)) as wav:
                checks['speech']='success'
                checks['sample_rate']=wav.getframerate()
                checks['duration_seconds']=wav.getnframes()/wav.getframerate()
        return {'state':'success','checks':checks,'elapsed_seconds':time.monotonic()-started,'executed_in_game':False}
    except CredentialError as exc: reason=exc.state
    except ProviderHTTPError as exc: reason='auth-error' if exc.status in (401,403) else 'http-error'
    except FileNotFoundError: reason='sample-unavailable'
    except ImportError: reason="runtime-unavailable"
    except (RuntimeError,ValueError,OSError,TimeoutError,HTTPException) as exc:
        reason='network-error' if isinstance(exc,(OSError,TimeoutError,HTTPException)) or isinstance(exc.__cause__,(URLError,OSError,TimeoutError,HTTPException)) else 'invalid-response'
    return {'state':'failed','reason':reason,'checks':checks,'elapsed_seconds':time.monotonic()-started,'executed_in_game':False}


if __name__=='__main__':
    import contextlib
    import json
    import sys
    import os
    import threading
    parent=os.environ.get("VRC_AGENT_PROBE_PARENT")
    if parent:
        from .probe_lifecycle import watch_parent
        threading.Thread(target=watch_parent,args=(int(parent),),daemon=True).start()
    # Model libraries may print initialization output; only JSON goes to stdout.
    with contextlib.redirect_stdout(sys.stderr):
        result=probe_connection(**json.loads(sys.stdin.read(2*1024*1024)))
    print(json.dumps(result))
