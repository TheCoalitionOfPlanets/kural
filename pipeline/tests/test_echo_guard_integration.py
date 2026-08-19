"""Worker-path tests for the echo guard and mute layer.

Drives the real stage classes with fake model workers, so no GPU or model
download is needed.

    venv\\Scripts\\python.exe pipeline\\tests\\test_echo_guard_integration.py
"""
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from pipeline.realtime.echo_guard import RecentSpeech  # noqa: E402
from pipeline.realtime.messages import Reply, Utterance  # noqa: E402
from pipeline.realtime.workers import STTStage, TTSStage  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


class FakeWorker:
    """Stands in for a model subprocess."""

    def __init__(self, reply_fn):
        self.reply_fn = reply_fn
        self.calls = []

    def run(self, payload):
        self.calls.append(payload)
        return self.reply_fn(payload)


def drain(q, timeout=2.0):
    items = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            items.append(q.get(timeout=0.05))
        except queue.Empty:
            if items:
                break
    return items


def run_stt(transcript, recent_speech, threshold=0.6):
    """Push one utterance through STTStage, return what reached the next queue."""
    tmp = Path(tempfile.mkdtemp())
    in_q, out_q = queue.Queue(8), queue.Queue(8)
    stop = threading.Event()
    events = []
    echoes = []

    worker = FakeWorker(lambda p: {"ok": True, "utt_id": p["utt_id"],
                                   "text": transcript, "elapsed_s": 0.1})
    stage = STTStage(worker, tmp, in_q, out_q, stop, lambda k, **kw: events.append((k, kw)),
                     recent_speech=recent_speech, echo_threshold=threshold,
                     on_echo=lambda u, t: echoes.append((u, t)))
    stage.start()
    in_q.put(Utterance("u1", np.zeros(1600, dtype=np.float32), 16000, 0.1,
                       time.perf_counter()))
    passed = drain(out_q)
    stop.set()
    stage.join(timeout=2)
    return passed, events, echoes


print("STT stage — echo suppression")
rs = RecentSpeech(6)
rs.add("Paris is the capital of France. It is a beautiful and historic city.")

passed, events, echoes = run_stt("paris is the capital of france", rs)
check("echoing transcript never reaches the model", passed == [])
check("echo is reported", len(echoes) == 1)
check("echo event carries the text",
      any(k == "echo_dropped" for k, _ in events))

passed, events, echoes = run_stt("what time is my meeting tomorrow", rs)
check("real speech passes through while assistant talks", len(passed) == 1)
check("real speech is not flagged", echoes == [])
check("passed item carries the transcript",
      passed and passed[0].text == "what time is my meeting tomorrow")

print("\nSTT stage — guard disabled")
passed, events, echoes = run_stt("paris is the capital of france", None)
check("echo passes when guard is off", len(passed) == 1)
check("no echo callback when off", echoes == [])

print("\nSTT stage — empty transcript still dropped")
passed, events, echoes = run_stt("   ", rs)
check("blank transcript dropped", passed == [])
check("reported as empty not echo",
      any(k == "stt_empty" for k, _ in events))

print("\nTTS stage — records what is spoken")
tmp = Path(tempfile.mkdtemp())
in_q, out_q = queue.Queue(8), queue.Queue(8)
stop = threading.Event()
window = RecentSpeech(6)


def fake_tts(p):
    Path(p["wav_path"]).write_bytes(b"")  # stage only checks the path back
    return {"ok": True, "utt_id": p["utt_id"], "wav_path": p["wav_path"],
            "sample_rate": 44100, "audio_s": 1.0, "elapsed_s": 0.5}


tts = TTSStage(FakeWorker(fake_tts), tmp, in_q, out_q, stop, lambda k, **kw: None,
               recent_speech=window)
tts.start()
in_q.put(Reply("u9", "The meeting is at four in the afternoon.", "when is it",
               time.perf_counter()))
jobs = drain(out_q)
stop.set()
tts.join(timeout=2)

check("synthesis produces a wav job", len(jobs) == 1)
check("spoken text recorded in window", len(window) == 1)
check("window content matches reply",
      "The meeting is at four in the afternoon." in window.snapshot())

print("\nTTS -> STT round trip")
# What the TTS stage recorded must be what the STT stage suppresses.
passed, _, echoes = run_stt("the meeting is at four in the afternoon", window)
check("assistant's own line is suppressed on the way back", passed == [])
check("round-trip echo reported", len(echoes) == 1)

print("\nTTS stage — failed synthesis")
tmp2 = Path(tempfile.mkdtemp())
in_q2, out_q2 = queue.Queue(8), queue.Queue(8)
stop2 = threading.Event()
window2 = RecentSpeech(6)
tts2 = TTSStage(FakeWorker(lambda p: {"ok": False, "utt_id": p["utt_id"],
                                      "error": "boom"}),
                tmp2, in_q2, out_q2, stop2, lambda k, **kw: None,
                recent_speech=window2)
tts2.start()
in_q2.put(Reply("u10", "This never gets spoken.", "x", time.perf_counter()))
jobs2 = drain(out_q2, timeout=1.0)
stop2.set()
tts2.join(timeout=2)
check("failed synthesis yields no job", jobs2 == [])

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all integration tests passed")
