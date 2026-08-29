"""Drives the WebSocket server end to end with stubbed model subprocesses.

    venv\\Scripts\\python.exe pipeline\\tests\\test_server.py

Everything except the three models is real here: the socket, the frame
re-chunking, the VAD endpointing, the four stage threads, the WebPlayer's
handshake with the client. What is faked is only what needs a GPU.

The point is the seam. `run_realtime.py` and the server share `Session`, so the
pipeline itself is already covered elsewhere; what is new and worth asserting
is that browser frames reach the VAD as whole frames, that events come back as
JSON in the right order, that reply audio arrives as bytes, and that a browser
going away does not wedge a playback stage waiting for a report.
"""
import json
import struct
import sys
import threading
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pipeline.realtime.session as session_mod  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


REPLY_TEXT = "I am doing well, thank you."
HEARD_TEXT = "how are you doing"


class FakeWorker:
    """Stands in for one model subprocess."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def start(self, timeout_s=300):
        return {"event": "ready", "load_s": 0.0, "lid": True,
                "international_stt": True, "international_tts": True,
                "vram_gb": 0.0}

    def run(self, payload):
        self.calls.append(payload)
        if self.name == "stt":
            return {"ok": True, "utt_id": payload["utt_id"], "text": HEARD_TEXT,
                    "lang": "english", "route": "local", "backend": "sravaani",
                    "confidence": 0.95, "elapsed_s": 0.01}
        if self.name == "llm":
            return {"ok": True, "utt_id": payload["utt_id"], "text": REPLY_TEXT,
                    "lang": "english", "elapsed_s": 0.01}
        # tts: the player reads this file, so it has to be a real WAV.
        path = Path(payload["wav_path"])
        rate, seconds = 24000, 0.25
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes(struct.pack("<%dh" % int(rate * seconds),
                                       *([0] * int(rate * seconds))))
        return {"ok": True, "utt_id": payload["utt_id"], "wav_path": str(path),
                "sample_rate": rate, "lang": "english", "backend": "indic-mio",
                "audio_s": seconds, "elapsed_s": 0.01}

    def stop(self, timeout=10):
        pass


def fake_worker_process(**kwargs):
    """Matches WorkerProcess's keyword-only construction in session.py."""
    return FakeWorker(kwargs["name"])


def test_config():
    """A copy of the real config, made deterministic and fast."""
    import yaml

    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "pipeline/config/realtime.yaml").read_text("utf-8"))
    vad = cfg["capture"]["vad"]
    # The energy gate is deterministic; webrtc's voicedness classifier is not,
    # and synthetic noise is not speech to it.
    vad["backend"] = "energy"
    vad["calibration_s"] = 0.15
    vad["silence_ms"] = 200
    vad["min_utterance_ms"] = 200
    cfg["runtime"]["spill_dir"] = tempfile.mkdtemp()
    path = Path(tempfile.mkdtemp()) / "test.yaml"
    path.write_text(yaml.safe_dump(cfg), "utf-8")
    return path


def pcm(seconds, amplitude, rate=16000):
    """int16 little-endian mono, the wire format the browser sends."""
    n = int(rate * seconds)
    if amplitude:
        rng = np.random.default_rng(0)
        sig = rng.normal(0, amplitude, n)
    else:
        sig = np.zeros(n)
    return (np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes()


class Browser:
    """A test client that behaves like the real one.

    Starlette's test websocket has no receive timeout, so reading happens on
    its own thread and everything else polls what that thread has collected.
    All *sending* stays on the main thread — two threads writing to one test
    socket is a flake waiting to happen.
    """

    def __init__(self, ws, autoplay=True):
        self.ws = ws
        self.autoplay = autoplay
        self.events, self.blobs = [], []
        self.played = []
        self._answered = set()
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        while not self._stop.is_set():
            try:
                packet = self.ws.receive()
            except Exception:
                return
            if packet.get("type") == "websocket.close":
                return
            if (raw := packet.get("text")) is not None:
                self.events.append(json.loads(raw))
            elif (data := packet.get("bytes")) is not None:
                self.blobs.append(data)

    def close(self):
        self._stop.set()

    def stream(self, data, chunk=256):
        """Send as a browser would: many small blocks, not frame-aligned."""
        for i in range(0, len(data), chunk * 2):
            self.ws.send_bytes(data[i:i + chunk * 2])

    def _pump(self):
        """Answer anything the server is waiting on. Main thread only."""
        if not self.autoplay:
            return
        for e in list(self.events):
            if e.get("type") != "audio" or e["utt_id"] in self._answered:
                continue
            self._answered.add(e["utt_id"])
            self.played.append(e["utt_id"])
            # Stand in for the browser's AudioBufferSourceNode.
            self.ws.send_text(json.dumps({"type": "playback_started",
                                          "utt_id": e["utt_id"]}))
            self.ws.send_text(json.dumps({"type": "playback_finished",
                                          "utt_id": e["utt_id"]}))

    def wait_for(self, kind, timeout=25.0, keep_streaming=True):
        """Poll until an event of `kind` arrives, feeding silence meanwhile.

        A real browser never stops sending. Keeping frames flowing is what
        makes this realistic *and* what keeps the capture thread producing the
        level events the orb runs on.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._pump()
            if any(e.get("type") == kind for e in self.events):
                return True
            if keep_streaming:
                self.stream(pcm(0.06, 0.0))
            time.sleep(0.02)
        self._pump()
        return any(e.get("type") == kind for e in self.events)

    def kinds(self):
        return [e.get("type") for e in self.events]

    def first(self, kind):
        return next((e for e in self.events if e.get("type") == kind), None)


def wait_idle(client, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline and client.get("/health").json()["busy"]:
        time.sleep(0.05)
    return client.get("/health").json()["busy"] is False


# --------------------------------------------------------------------------

session_mod.WorkerProcess = fake_worker_process
from fastapi.testclient import TestClient  # noqa: E402

from pipeline.server.app import create_app  # noqa: E402

config_path = test_config()
app = create_app(config_path)

with TestClient(app) as client:
    deadline = time.time() + 15
    while time.time() < deadline and client.get("/health").json()["status"] == "loading":
        time.sleep(0.05)

    health = client.get("/health").json()
    print("health")
    check("models report ready", health["status"] == "ready")
    check("capture rate is advertised", health["sample_rate"] == 16000)
    check("frame size is advertised", health["frame_ms"] == 20)
    check("nothing is holding the pipeline", health["busy"] is False)

    print("\na full turn — browser audio in, reply audio out")
    with client.websocket_connect("/ws") as ws:
        b = Browser(ws)
        check("hello arrives first", b.wait_for("hello", timeout=5,
                                                keep_streaming=False))
        hello = b.first("hello")
        check("hello carries the frame contract",
              hello["sample_rate"] == 16000 and hello["frame_ms"] == 20)

        # A real browser streams in real time, so the noise floor is measured
        # from a second of actual room silence. Dumping the whole clip at once
        # instead lets calibration swallow the speech burst and take its median
        # from it — the floor ends up above the speech it was meant to sit
        # under, and nothing is ever detected again.
        check("noise floor was calibrated from browser frames",
              b.wait_for("calibrated", timeout=10))
        b.stream(pcm(0.6, 0.35))
        got = b.wait_for("latency")
        b.close()

    check("the turn completed", got)
    seen = b.kinds()
    check("capture reported listening", "listening" in seen)
    check("speech was detected", "speech_start" in seen)
    check("an utterance was closed", "utterance" in seen)
    check("the transcript came back", (b.first("stt") or {}).get("text") == HEARD_TEXT)
    check("the reply came back", (b.first("llm") or {}).get("text") == REPLY_TEXT)
    check("synthesis was reported", b.first("tts") is not None)
    check("audio was announced", b.first("audio") is not None)
    check("audio arrived as bytes", len(b.blobs) == 1)
    check("the bytes are a WAV", b.blobs and b.blobs[0][:4] == b"RIFF")
    check("the announced size matches the payload",
          b.blobs and (b.first("audio") or {}).get("bytes") == len(b.blobs[0]))
    check("the browser was asked to play it", len(b.played) == 1)

    lat = b.first("latency") or {}
    check("latency covers every stage",
          all(k in lat for k in ("stt_ms", "llm_ms", "tts_ms", "total_ms")))

    # The orb is driven by these; at 50/s they would swamp the socket, so the
    # server thins them to ~20/s.
    levels = [e for e in b.events if e.get("type") == "level"]
    check("level events are sent", len(levels) > 0)
    check("level events are thinned, not one per frame",
          len(levels) < sum(1 for e in b.events) )

    print("\nthe pipeline is released when the tab closes")
    check("no longer busy", wait_idle(client))

    print("\na second browser is refused, not interleaved")
    with client.websocket_connect("/ws") as ws1:
        b1 = Browser(ws1)
        b1.wait_for("hello", timeout=5, keep_streaming=False)
        with client.websocket_connect("/ws") as ws2:
            b2 = Browser(ws2)
            b2.wait_for("fatal", timeout=5, keep_streaming=False)
            msg = b2.first("fatal")
            check("second connection is told why",
                  msg is not None and "already" in msg["message"])
            b2.close()
        b1.close()
    check("released again", wait_idle(client))

    print("\na browser that never plays must not wedge the pipeline")
    # No playback_started is ever sent, so WebPlayer falls through its start
    # timeout. The turn is lost; the session must not be.
    import pipeline.realtime.web_player as wp_mod
    original = wp_mod.START_TIMEOUT_S
    wp_mod.START_TIMEOUT_S = 0.4
    try:
        with client.websocket_connect("/ws") as ws:
            b = Browser(ws, autoplay=False)
            b.wait_for("hello", timeout=5, keep_streaming=False)
            b.wait_for("calibrated", timeout=10)
            b.stream(pcm(0.6, 0.35))
            stalled = b.wait_for("notice", timeout=20)
            b.close()
        notice = b.first("notice")
        check("the stall is reported, not swallowed",
              stalled and notice.get("event") == "playback_never_started")
    finally:
        wp_mod.START_TIMEOUT_S = original

    check("released after the stall", wait_idle(client))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all server tests passed")
