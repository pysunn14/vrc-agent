from io import BytesIO
import threading
import time
from urllib.parse import quote
import uuid
import wave

from .decision import decode_decision, messages
from .audio import complete_streaming_wav
from .extensions import chat_payload, transcription_fields
from ..crisper_stt import _multipart_body, _pcm_to_wav
from ..device_payloads import MAX_WAV_PAYLOAD_BYTES


class ChatAdapter:
    def __init__(self, http, binding, *, provider_id, extension=""):
        self.http, self.binding, self.provider, self.extension = http, binding, provider_id, extension
        self.model = binding["model"]
        self.reasoning_effort = binding.get("reasoning_effort", "")

    def snapshot(self):
        return {"provider": self.provider, "model": self.model, "vision": self.binding["vision"],
                "reasoning_effort": self.reasoning_effort, "endpoint": self.http.base_url}

    def propose(self, request):
        timeout = self.http.timeout
        if request.deadline_monotonic is not None:
            remaining = request.deadline_monotonic - time.monotonic()
            if remaining <= 0: raise TimeoutError("brain request deadline has expired")
            timeout = min(timeout, remaining)
        if request.screenshot and not self.binding["vision"]:
            raise ValueError("selected model binding does not support vision")
        payload = chat_payload(self.binding, messages(request), self.extension)
        return decode_decision(self.http.json("chat/completions", payload=payload, limit=128*1024, timeout=timeout), request)


class TranscriptionAdapter:
    def __init__(self, http, binding, *, extension=""):
        self.http, self.binding, self.extension = http, binding, extension

    def transcribe_pcm(self, pcm_s16le):
        fields = transcription_fields(self.binding, self.extension)
        boundary = uuid.uuid4().hex
        body = _multipart_body(boundary, fields=fields, wav=_pcm_to_wav(pcm_s16le))
        payload = self.http.json("audio/transcriptions", body=body, content_type=f"multipart/form-data; boundary={boundary}")
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str) or not payload["text"].strip():
            raise RuntimeError("transcription provider returned an empty transcript")
        return payload["text"].strip()


def pcm_wav(raw, rate=24000):
    if not raw or len(raw) % 2: raise RuntimeError("speech provider returned invalid PCM samples")
    stream = BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(rate); writer.writeframes(raw)
    return stream.getvalue()


def validate_wav(raw):
    try:
        with wave.open(BytesIO(raw)) as reader:
            frames = reader.getnframes()
            if reader.getsampwidth() != 2 or reader.getnchannels() not in (1, 2) or frames <= 0:
                raise ValueError("expected non-empty 16-bit PCM mono or stereo WAV")
            if len(reader.readframes(frames)) != frames * reader.getnchannels() * reader.getsampwidth():
                raise ValueError("truncated WAV")
    except (wave.Error, EOFError, ValueError) as exc: raise RuntimeError(f"invalid speech WAV: {exc}") from exc
    return raw


class SpeechAdapter:
    def __init__(self, http, binding, *, protocol, streaming_wav=False):
        self.http, self.binding, self.protocol = http, binding, protocol
        self.streaming_wav = streaming_wav

    def synthesize(self, text):
        if not isinstance(text, str) or not text.strip(): raise ValueError("speech text must not be empty")
        if self.protocol == "elevenlabs":
            path = f"text-to-speech/{quote(self.binding['voice'], safe='')}?output_format=pcm_24000"
            payload = {"text": text, "model_id": self.binding["model"]}
            raw, mime = self.http.request(path, payload=payload, accept="audio/pcm", limit=MAX_WAV_PAYLOAD_BYTES-44)
            if mime not in ("audio/pcm", "audio/x-pcm", "application/octet-stream"):
                raise RuntimeError(f"unexpected speech content type: {mime}")
            return pcm_wav(raw)
        payload = {"model": self.binding["model"], "input": text, "voice": self.binding["voice"], "response_format": "wav"}
        raw, mime = self.http.request("audio/speech", payload=payload, accept="audio/wav", limit=MAX_WAV_PAYLOAD_BYTES)
        if mime not in ("audio/wav", "audio/x-wav", "application/octet-stream"):
            raise RuntimeError(f"unexpected speech content type: {mime}")
        return validate_wav(complete_streaming_wav(raw) if self.streaming_wav else raw)


class LocalWhisperAdapter:
    """Load once on the execution host. Missing models never trigger a download."""
    def __init__(self, config, binding):
        self.config, self.binding = config, binding
        self._model = None
        self._lock = threading.Lock()

    def transcribe_pcm(self, pcm_s16le):
        audio = BytesIO(_pcm_to_wav(pcm_s16le))
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.binding["model"], device=self.config["device"],
                                          compute_type=self.config["compute_type"], local_files_only=True)
            segments, _ = self._model.transcribe(audio, language=self.binding.get("language") or None)
            result = " ".join(segment.text.strip() for segment in segments).strip()
        if not result: raise RuntimeError("local transcription returned an empty transcript")
        return result
