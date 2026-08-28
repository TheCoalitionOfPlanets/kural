"""ElevenLabs HTTP client — the international half of STT and TTS.

The local stack is Indic by construction (see `languages.route_for`), so a
turn in Spanish or Russian or Japanese has neither a local ear nor a local
voice. Those turns are served here: Scribe transcribes them, and ElevenLabs'
multilingual TTS speaks the reply.

This module is imported by workers running in three different venvs, so it is
**stdlib only** — `urllib.request` rather than `requests`, and the multipart
body below is hand-rolled for the same reason. That is also why it deals in
bytes: the caller owns numpy, this owns the wire.

Audio is requested as raw PCM (`output_format: pcm_*`) rather than MP3, so
nothing here has to decode a compressed stream. The bytes come back ready to
be wrapped in a WAV header and handed to the same normalization and playback
path the local voice uses — which is what keeps barge-in, the echo guard and
the VAD gate working identically on an international reply.
"""
import io
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
import wave

API_ROOT = "https://api.elevenlabs.io/v1"

# Scribe is the only speech-to-text model ElevenLabs offers, and it detects the
# language itself — which is why LID upstream only has to decide *whether* to
# come here, not exactly what was spoken.
DEFAULT_STT_MODEL = "scribe_v1"

# eleven_multilingual_v2: 29 languages, higher quality than the flash/turbo
# tiers at the cost of some latency. `languages.ELEVEN_TTS_LANGUAGES` is that
# list; a reply outside it is left unspoken rather than mispronounced.
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"

# Rachel — a stock voice on every account, so the pipeline runs before anyone
# has picked one. Override with elevenlabs.voice_id.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# 24kHz PCM is available on every paid tier; pcm_44100 needs Pro or above, and
# silently 401s below it. Higher rates buy nothing here — the reply is speech
# played through one speaker, and it is loudness-normalized immediately after.
DEFAULT_OUTPUT_FORMAT = "pcm_24000"

_PCM_RATES = {
    "pcm_8000": 8000,
    "pcm_16000": 16000,
    "pcm_22050": 22050,
    "pcm_24000": 24000,
    "pcm_44100": 44100,
}


class ElevenLabsError(RuntimeError):
    """A call failed. `status` is the HTTP code, or None for a transport error."""

    def __init__(self, message, status=None, body=""):
        super().__init__(message)
        self.status = status
        self.body = body


def output_format_rate(fmt):
    """Sample rate implied by an `output_format`, rejecting compressed ones.

    Nothing downstream can decode MP3 — playback reads a WAV and the loudness
    meter needs samples — so an unusable format is refused here, at config
    time, instead of producing a file that fails to open one reply later.
    """
    fmt = (fmt or DEFAULT_OUTPUT_FORMAT).strip().lower()
    if fmt not in _PCM_RATES:
        raise ElevenLabsError(
            f"output_format {fmt!r} is not raw PCM. This pipeline writes WAV "
            f"and normalizes loudness, so it cannot use a compressed format. "
            f"Use one of: {', '.join(sorted(_PCM_RATES))}."
        )
    return _PCM_RATES[fmt]


def pcm16_to_wav(pcm_bytes, sample_rate, channels=1):
    """Wrap raw 16-bit little-endian PCM in a WAV container, in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm_bytes)
    return buf.getvalue()


def _multipart(fields, files):
    """Encode a multipart/form-data body.

    `requests` would do this in one line, but it is not a dependency of any of
    the three venvs and adding it to all of them to post one form is a poor
    trade. `files` is [(field, filename, content_type, data)].
    """
    boundary = f"----kural{uuid.uuid4().hex}"
    sep = f"--{boundary}\r\n".encode()
    out = bytearray()
    for name, value in fields.items():
        if value is None:
            continue
        out += sep
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    for name, filename, ctype, data in files:
        out += sep
        out += (f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n').encode()
        out += f"Content-Type: {ctype or 'application/octet-stream'}\r\n\r\n".encode()
        out += data
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _request(url, api_key, data, content_type, timeout, retries, accept):
    """POST once, retrying only what is worth retrying.

    A 4xx is a bad key or a bad request — retrying it just spends the latency
    budget on the same answer. Rate limits, 5xx and transport failures are
    transient, so those get one more attempt by default. In a realtime loop
    every retry is silence the user is sitting through, so the count is low
    and configurable rather than a library default.
    """
    last = None
    for attempt in range(max(1, int(retries) + 1)):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("xi-api-key", api_key)
        req.add_header("Content-Type", content_type)
        req.add_header("Accept", accept)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            last = ElevenLabsError(
                f"HTTP {exc.code} from {url}: {body or exc.reason}",
                status=exc.code, body=body,
            )
            if exc.code == 401:
                last = ElevenLabsError(
                    "ElevenLabs rejected the API key (HTTP 401). Check the "
                    "environment variable named by elevenlabs.api_key_env.",
                    status=401, body=body,
                )
            if exc.code < 500 and exc.code != 429:
                raise last
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = ElevenLabsError(f"could not reach {url}: {exc!r}")
        if attempt < retries:
            # Brief, fixed backoff. Exponential would be correct for a batch
            # job and wrong here: past a second or so the turn is stale and
            # failing fast is the better answer.
            time.sleep(0.4 * (attempt + 1))
    raise last


class ElevenLabs:
    """Config-bound client. One per worker; requests are made on its thread."""

    def __init__(self, api_key, *, stt_model=DEFAULT_STT_MODEL,
                 tts_model=DEFAULT_TTS_MODEL, voice_id=DEFAULT_VOICE_ID,
                 output_format=DEFAULT_OUTPUT_FORMAT, voice_settings=None,
                 timeout_s=20.0, retries=1, api_root=API_ROOT):
        if not api_key:
            raise ElevenLabsError("no ElevenLabs API key")
        self.api_key = api_key
        self.stt_model = stt_model or DEFAULT_STT_MODEL
        self.tts_model = tts_model or DEFAULT_TTS_MODEL
        self.voice_id = voice_id or DEFAULT_VOICE_ID
        self.output_format = (output_format or DEFAULT_OUTPUT_FORMAT).lower()
        self.sample_rate = output_format_rate(self.output_format)
        self.voice_settings = voice_settings or {}
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.api_root = api_root.rstrip("/")

    # -- speech to text ----------------------------------------------------

    def transcribe(self, wav_bytes, language_code=None, filename="utterance.wav"):
        """One utterance -> {"text", "language", "language_code", "probability"}.

        `language_code` is a *hint*, passed only when the caller is confident.
        Scribe's own detection is better than the local LID gate's, so the
        language it reports is the one the rest of the turn runs on — which is
        also how a misrouted turn corrects itself: LID says Spanish, Scribe
        says Tamil, and TTS routes back to the local voice.
        """
        fields = {
            "model_id": self.stt_model,
            # Both add latency and neither is used: there is one speaker and
            # the transcript is fed to an LLM, not rendered as subtitles.
            "diarize": "false",
            "tag_audio_events": "false",
        }
        if language_code:
            fields["language_code"] = language_code
        ctype = mimetypes.guess_type(filename)[0] or "audio/wav"
        body, content_type = _multipart(
            fields, [("file", filename, ctype, wav_bytes)]
        )
        raw, _ = _request(
            f"{self.api_root}/speech-to-text", self.api_key, body, content_type,
            self.timeout_s, self.retries, "application/json",
        )
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except ValueError as exc:
            raise ElevenLabsError(f"speech-to-text returned non-JSON: {exc}")
        return {
            "text": (payload.get("text") or "").strip(),
            "language_code": payload.get("language_code"),
            "probability": payload.get("language_probability"),
        }

    # -- text to speech ----------------------------------------------------

    def synthesize(self, text):
        """One reply -> (raw PCM16 little-endian bytes, sample rate).

        No `language_code` is sent: eleven_multilingual_v2 infers the language
        from the text, and the parameter is only accepted by the turbo/flash
        v2.5 models. The language has already been decided upstream anyway —
        it is what routed this reply here.
        """
        payload = {"text": text, "model_id": self.tts_model}
        if self.voice_settings:
            payload["voice_settings"] = self.voice_settings
        url = (f"{self.api_root}/text-to-speech/{self.voice_id}"
               f"?output_format={self.output_format}")
        raw, content_type = _request(
            url, self.api_key, json.dumps(payload).encode("utf-8"),
            "application/json", self.timeout_s, self.retries, "audio/*",
        )
        if "json" in (content_type or "").lower():
            # An error body slipping through with a 200 would otherwise be
            # written to a WAV and played as a burst of noise.
            raise ElevenLabsError(
                f"text-to-speech returned JSON, not audio: "
                f"{raw.decode('utf-8', 'replace')[:300]}"
            )
        if not raw:
            raise ElevenLabsError("text-to-speech returned no audio")
        return raw, self.sample_rate
