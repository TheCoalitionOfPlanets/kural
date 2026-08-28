"""Tests that the identified language survives the whole pipeline.

    venv\\Scripts\\python.exe pipeline\\tests\\test_international_flow.py

The routing decision is made once, at STT, from the waveform. Everything after
that depends on it arriving intact: the LLM needs it because a Spanish
transcript is Latin script that detect_language() reads as English, and TTS
needs it because it is what chooses between the local voice and ElevenLabs. So
what is tested here is the plumbing between the stages, with the models
replaced by scripted fakes.
"""
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.realtime.messages import Utterance  # noqa: E402
from pipeline.realtime.workers import LLMStage, STTStage, TTSStage  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


class FakeWorker:
    """Stands in for a model subprocess, recording what it was asked."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def run(self, payload):
        self.calls.append(payload)
        return dict(self.response)


def run_pipeline(stt_response, llm_response, tts_response, pcm_seconds=2.0):
    """Push one utterance through STT -> LLM -> TTS with fake models.

    Returns (wav_job_or_None, events, workers).
    """
    audio_q, transcript_q = queue.Queue(), queue.Queue()
    reply_q, wav_q = queue.Queue(), queue.Queue()
    stop = threading.Event()
    events = []
    lock = threading.Lock()

    def on_event(kind, **kw):
        with lock:
            events.append((kind, kw))

    stt_w = FakeWorker(stt_response)
    llm_w = FakeWorker(llm_response)
    tts_w = FakeWorker(tts_response)

    with tempfile.TemporaryDirectory() as spill:
        stages = [
            STTStage(stt_w, spill, audio_q, transcript_q, stop, on_event),
            LLMStage(llm_w, transcript_q, reply_q, stop, on_event),
            TTSStage(tts_w, spill, reply_q, wav_q, stop, on_event),
        ]
        for s in stages:
            s.start()

        audio_q.put(Utterance(
            utt_id="u1",
            pcm=np.zeros(int(16000 * pcm_seconds), dtype=np.float32),
            sample_rate=16000, duration_s=pcm_seconds,
            t_captured=time.perf_counter(),
        ))

        job = None
        try:
            job = wav_q.get(timeout=5)
        except queue.Empty:
            # Not a failure on its own: the no-voice and no-ear cases are
            # supposed to end here, and the events say why.
            pass
        # Let a terminal event land before the threads are torn down.
        time.sleep(0.05)
        stop.set()
        for s in stages:
            s.join(timeout=2)
    return job, events, (stt_w, llm_w, tts_w)


def kinds(events):
    return [k for k, _ in events]


def payload(events, kind):
    for k, kw in events:
        if k == kind:
            return kw
    return None


print("international turn — Scribe's language reaches both the model and TTS")
job, events, (stt_w, llm_w, tts_w) = run_pipeline(
    {"ok": True, "text": "¿Cómo estás?", "lang": "spanish", "lang_code": "spa",
     "route": "international", "backend": "elevenlabs", "confidence": 0.98},
    {"ok": True, "text": "Estoy bien, gracias.", "lang": "spanish"},
    {"ok": True, "wav_path": "/tmp/u1.wav", "sample_rate": 24000,
     "lang": "spanish", "backend": "elevenlabs", "audio_s": 1.4},
)
check("a wav job came out the far end", job is not None)
# The rate travels with the audio because LID and SraVaani are both 16kHz-only
# and a mismatch there is silent rather than loud.
check("stt worker was told the sample rate",
      stt_w.calls and stt_w.calls[0].get("sample_rate") == 16000)
# The whole point: without this the LLM re-derives the language from Latin-script
# Spanish and calls it English.
check("llm worker was told the language",
      llm_w.calls and llm_w.calls[0].get("lang") == "spanish")
check("tts worker was told the language",
      tts_w.calls and tts_w.calls[0].get("lang") == "spanish")
check("stt event names the backend",
      payload(events, "stt").get("backend") == "elevenlabs")
check("tts event names the backend",
      payload(events, "tts").get("backend") == "elevenlabs")
check("wav job carries the international sample rate",
      job is not None and job.sample_rate == 24000)

print("\nlocal turn — nothing about it changed")
job, events, (stt_w, llm_w, tts_w) = run_pipeline(
    {"ok": True, "text": "எப்படி இருக்கீங்க", "lang": "tamil",
     "route": "local", "backend": "sravaani", "confidence": 0.94},
    {"ok": True, "text": "நான் நன்றாக இருக்கிறேன்", "lang": "tamil"},
    {"ok": True, "wav_path": "/tmp/u1.wav", "sample_rate": 44100,
     "lang": "tamil", "backend": "indic-mio", "audio_s": 1.1},
)
check("local turn produces a wav job", job is not None)
check("llm worker got tamil", llm_w.calls[0].get("lang") == "tamil")
check("tts worker got tamil", tts_w.calls[0].get("lang") == "tamil")
check("no backend tag on a local stt event",
      payload(events, "stt").get("backend") == "sravaani")

print("\nLID abstained — the old transcript-based path still works")
job, events, (stt_w, llm_w, tts_w) = run_pipeline(
    {"ok": True, "text": "enna panra", "lang": None, "backend": "sravaani"},
    # No language from STT, so the LLM worker detects it and reports back.
    {"ok": True, "text": "நல்லா இருக்கேன்", "lang": "tamil"},
    {"ok": True, "wav_path": "/tmp/u1.wav", "sample_rate": 44100,
     "lang": "tamil", "backend": "indic-mio", "audio_s": 1.0},
)
check("llm worker receives no language", llm_w.calls[0].get("lang") is None)
# The LLM's own detection has to reach TTS, or the reply is synthesized with
# no language at all.
check("the language the llm found reaches tts",
      tts_w.calls[0].get("lang") == "tamil")

print("\nno ear — identified, and nowhere to send it")
job, events, _ = run_pipeline(
    {"ok": False, "error": "no_international_stt", "lang": "spanish",
     "confidence": 0.97},
    {"ok": True, "text": "unused"},
    {"ok": True, "wav_path": "/tmp/u1.wav", "sample_rate": 24000},
)
check("nothing is spoken", job is None)
# The fix is an API key, not something wrong with the audio, so it must not be
# reported as an STT failure.
check("reported as its own event", "stt_no_international" in kinds(events))
check("not reported as a generic failure", "stt_failed" not in kinds(events))
check("the event names the language",
      payload(events, "stt_no_international").get("lang") == "spanish")

print("\nno voice — heard perfectly, cannot be spoken")
job, events, _ = run_pipeline(
    {"ok": True, "text": "ආයුබෝවන්", "lang": "sinhala",
     "route": "international", "backend": "elevenlabs", "confidence": 0.91},
    {"ok": True, "text": "ඔබට කෙසේද", "lang": "sinhala"},
    {"ok": False, "error": "no_voice", "lang": "sinhala",
     "reason": "not one of the multilingual model's languages"},
)
check("nothing is spoken", job is None)
check("reported as a missing voice", "tts_no_voice" in kinds(events))
check("not reported as a crash", "tts_failed" not in kinds(events))
# The answer is still correct; the console prints it as text, so it has to
# carry both the reply and why it was not read aloud.
check("the reply text survives for printing",
      payload(events, "tts_no_voice").get("text") == "ඔබට කෙසේද")
check("the reason survives for printing",
      "multilingual" in (payload(events, "tts_no_voice").get("reason") or ""))

print("\nnetwork failure — a real failure, not a missing voice")
job, events, _ = run_pipeline(
    {"ok": True, "text": "¿Cómo estás?", "lang": "spanish",
     "backend": "elevenlabs", "confidence": 0.98},
    {"ok": True, "text": "Estoy bien.", "lang": "spanish"},
    {"ok": False, "error": "elevenlabs_tts: could not reach api", "lang": "spanish"},
)
check("nothing is spoken", job is None)
check("reported as a failure", "tts_failed" in kinds(events))
check("not reported as a missing voice", "tts_no_voice" not in kinds(events))

print("\nvoice coverage — the gate must not name a language TTS then refuses")
# Before audio-level LID, the script detector bucketed every Devanagari
# language under "hindi", so names like "marathi" and "konkani" never reached
# the TTS worker. Now they do, and each one it does not recognize becomes a
# reply that is silently printed instead of spoken.
import importlib.util  # noqa: E402

from pipeline.realtime.languages import (  # noqa: E402
    ELEVEN_TTS_LANGUAGES,
    LOCAL,
    ROUTE_INTERNATIONAL,
    route_for,
)
from pipeline.realtime.languages import _ISO639_1 as _iso639_1  # noqa: E402
from pipeline.realtime.languages import _ISO639_3 as _iso639_3  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_worker_tts", Path(__file__).resolve().parents[1] / "workers" / "worker_tts.py"
)
_worker_tts = importlib.util.module_from_spec(_spec)
_stdout = sys.stdout          # the worker repoints stdout at import time
try:
    _spec.loader.exec_module(_worker_tts)
finally:
    sys.stdout = _stdout

_gap = sorted(LOCAL - _worker_tts.SUPPORTED_LANGS)
check("every locally-routed language has a local voice", not _gap)
if _gap:
    print(f"        no Indic-Mio voice for: {_gap}")

# The mirror of the same mistake, swept across every language the code tables
# can produce: each one must have a voice on whichever stack it routes to, or
# be a gap someone decided to accept. Scribe hears far more languages than the
# multilingual voice speaks, so the gaps are real — pinning them here means a
# new one cannot appear quietly, only deliberately.
KNOWN_TEXT_ONLY = {
    "sinhala", "vietnamese", "thai", "hebrew", "hungarian", "norwegian",
    "persian", "swahili", "afrikaans",
}
_all_names = set(_iso639_3.values()) | set(_iso639_1.values())
_voiceless = set()
for _lang in _all_names:
    if route_for(_lang) == ROUTE_INTERNATIONAL:
        if _lang not in ELEVEN_TTS_LANGUAGES:
            _voiceless.add(_lang)
    elif _lang not in _worker_tts.SUPPORTED_LANGS:
        _voiceless.add(_lang)

check("the text-only gap is exactly what is documented",
      _voiceless == KNOWN_TEXT_ONLY)
if _voiceless != KNOWN_TEXT_ONLY:
    print(f"        newly voiceless: {sorted(_voiceless - KNOWN_TEXT_ONLY)}")
    print(f"        no longer voiceless: {sorted(KNOWN_TEXT_ONLY - _voiceless)}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all international-flow tests passed")
