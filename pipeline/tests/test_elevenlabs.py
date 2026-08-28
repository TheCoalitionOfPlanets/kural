"""Tests for the ElevenLabs client — the international ear and voice.

    venv\\Scripts\\python.exe pipeline\\tests\\test_elevenlabs.py

No network. `urlopen` is replaced, so what is under test is the wire format and
the retry policy — the two things that fail silently in production and cannot
be checked by reading the code.
"""
import io
import json
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.realtime.elevenlabs import (  # noqa: E402
    ElevenLabs,
    ElevenLabsError,
    _multipart,
    _request,
    output_format_rate,
    pcm16_to_wav,
)

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


class _FakeResponse:
    def __init__(self, body, content_type="application/json"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patch_urlopen(fn):
    """Swap urlopen for the duration of one test, then put it back."""
    original = urllib.request.urlopen
    urllib.request.urlopen = fn
    return original


print("output_format — only raw PCM can reach the WAV writer")
check("pcm_24000 -> 24000", output_format_rate("pcm_24000") == 24000)
check("pcm_16000 -> 16000", output_format_rate("pcm_16000") == 16000)
check("default when unset", output_format_rate(None) == 24000)
try:
    output_format_rate("mp3_44100_128")
    check("mp3 refused", False)
except ElevenLabsError as exc:
    # Nothing downstream can decode MP3, so this has to fail at config time
    # rather than produce a WAV that will not open one reply later.
    check("mp3 refused", "not raw PCM" in str(exc))

print("\npcm16_to_wav — the bytes must survive the wave module")
pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in range(-100, 100))
blob = pcm16_to_wav(pcm, 24000)
check("has a RIFF header", blob[:4] == b"RIFF")
with wave.open(io.BytesIO(blob), "rb") as fh:
    check("mono", fh.getnchannels() == 1)
    check("16-bit", fh.getsampwidth() == 2)
    check("rate preserved", fh.getframerate() == 24000)
    check("samples round-trip", fh.readframes(fh.getnframes()) == pcm)

print("\nmultipart — hand-rolled, so the framing is worth asserting")
body, ctype = _multipart(
    {"model_id": "scribe_v1", "language_code": None, "diarize": "false"},
    [("file", "utterance.wav", "audio/wav", b"RIFFxxxx")],
)
boundary = ctype.split("boundary=")[1]
check("content-type carries the boundary", boundary in ctype)
check("body opens with the boundary", body.startswith(f"--{boundary}\r\n".encode()))
check("body closes with the terminator", body.endswith(f"--{boundary}--\r\n".encode()))
check("CRLF line endings", b"\r\n" in body and b"\n\n" not in body)
check("field included", b'name="model_id"' in body and b"scribe_v1" in body)
# A None field must be omitted entirely — sending an empty language_code would
# pin Scribe to no language at all rather than letting it detect one.
check("None field omitted", b'name="language_code"' not in body)
check("file part carries filename", b'filename="utterance.wav"' in body)
check("file bytes survive", b"RIFFxxxx" in body)

print("\napi key — a client with no key must not be constructible")
try:
    ElevenLabs("")
    check("empty key refused", False)
except ElevenLabsError:
    check("empty key refused", True)
try:
    ElevenLabs(None)
    check("missing key refused", False)
except ElevenLabsError:
    check("missing key refused", True)

print("\nretry policy — every retry is silence the user sits through")
calls = []


def _http_error(code):
    def _fn(req, timeout=None):
        calls.append(code)
        raise urllib.error.HTTPError(req.full_url, code, "err", {},
                                     io.BytesIO(b'{"detail":"nope"}'))
    return _fn


original = patch_urlopen(_http_error(400))
try:
    calls.clear()
    try:
        _request("https://x/y", "k", b"", "application/json", 5, 2, "application/json")
    except ElevenLabsError:
        pass
    # A bad request returns the same answer however many times it is asked.
    check("4xx is not retried", len(calls) == 1)

    urllib.request.urlopen = _http_error(401)
    calls.clear()
    try:
        _request("https://x/y", "k", b"", "application/json", 5, 2, "application/json")
        check("401 explains the key", False)
    except ElevenLabsError as exc:
        check("401 is not retried", len(calls) == 1)
        # The fix is an environment variable, so the message has to say so.
        check("401 explains the key",
              exc.status == 401 and "api_key_env" in str(exc))

    urllib.request.urlopen = _http_error(500)
    calls.clear()
    try:
        _request("https://x/y", "k", b"", "application/json", 5, 1, "application/json")
    except ElevenLabsError:
        pass
    check("5xx is retried once", len(calls) == 2)

    urllib.request.urlopen = _http_error(429)
    calls.clear()
    try:
        _request("https://x/y", "k", b"", "application/json", 5, 1, "application/json")
    except ElevenLabsError:
        pass
    check("429 is retried", len(calls) == 2)
finally:
    urllib.request.urlopen = original

print("\ntranscribe — Scribe's answer is what the turn runs on")
payload = json.dumps({
    "text": "  ¿Cómo estás?  ",
    "language_code": "spa",
    "language_probability": 0.98,
}).encode()
original = patch_urlopen(lambda req, timeout=None: _FakeResponse(payload))
try:
    client = ElevenLabs("k")
    got = client.transcribe(pcm16_to_wav(pcm, 16000), language_code="spa")
    check("text stripped", got["text"] == "¿Cómo estás?")
    check("language reported", got["language_code"] == "spa")
    check("probability reported", got["probability"] == 0.98)
finally:
    urllib.request.urlopen = original

print("\nsynthesize — an error body must never be played as audio")
audio = b"\x01\x02" * 64
original = patch_urlopen(lambda req, timeout=None: _FakeResponse(audio, "audio/mpeg"))
try:
    client = ElevenLabs("k", output_format="pcm_24000")
    raw, rate = client.synthesize("hola")
    check("audio returned", raw == audio)
    check("rate matches output_format", rate == 24000)
finally:
    urllib.request.urlopen = original

original = patch_urlopen(
    lambda req, timeout=None: _FakeResponse(b'{"detail":"quota"}', "application/json")
)
try:
    try:
        ElevenLabs("k").synthesize("hola")
        check("JSON body rejected as audio", False)
    except ElevenLabsError as exc:
        # Written to a WAV and played, this would be a burst of noise.
        check("JSON body rejected as audio", "not audio" in str(exc))
finally:
    urllib.request.urlopen = original

original = patch_urlopen(lambda req, timeout=None: _FakeResponse(b"", "audio/mpeg"))
try:
    try:
        ElevenLabs("k").synthesize("hola")
        check("empty audio rejected", False)
    except ElevenLabsError:
        check("empty audio rejected", True)
finally:
    urllib.request.urlopen = original

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all elevenlabs tests passed")
