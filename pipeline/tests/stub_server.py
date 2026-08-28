"""The real WebSocket server with the three models faked out.

    venv/bin/python pipeline/tests/stub_server.py        # port 8123

For working on the UI, and for `web/e2e/voice.mjs`. Everything except the
models is the production path: the same Session, the same VAD, the same
WebPlayer handshake — so the socket protocol, the frame contract and the event
stream are exactly what the real server produces.

Needs no GPU, no model weights, and no ElevenLabs key.
"""
import math
import pathlib
import struct
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPLY = "I am doing well, thank you for asking."
HEARD = "how are you"


class FakeWorker:
    """One model subprocess, minus the model."""

    def __init__(self, name):
        self.name = name

    def start(self, timeout_s=300):
        return {"event": "ready", "load_s": 0.0, "lid": True,
                "elevenlabs": True, "vram_gb": 0.0}

    def run(self, payload):
        if self.name == "stt":
            return {"ok": True, "utt_id": payload["utt_id"], "text": HEARD,
                    "lang": "english", "route": "local", "backend": "sravaani",
                    "confidence": 0.95}
        if self.name == "llm":
            return {"ok": True, "utt_id": payload["utt_id"], "text": REPLY,
                    "lang": "english"}
        # The player reads this off disk and the browser decodes it, so it has
        # to be a real, well-formed WAV rather than a stub path.
        path = pathlib.Path(payload["wav_path"])
        rate, seconds = 24000, 0.6
        n = int(rate * seconds)
        pcm = [int(math.sin(2 * math.pi * 220 * i / rate) * 0.3 * 32767)
               for i in range(n)]
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes(struct.pack("<%dh" % n, *pcm))
        return {"ok": True, "utt_id": payload["utt_id"], "wav_path": str(path),
                "sample_rate": rate, "lang": "english", "backend": "indic-mio",
                "audio_s": seconds}

    def stop(self, timeout=10):
        pass


def main():
    import yaml

    import pipeline.realtime.session as session_mod

    session_mod.WorkerProcess = lambda **kw: FakeWorker(kw["name"])

    cfg = yaml.safe_load(
        (ROOT / "pipeline/config/realtime.yaml").read_text("utf-8"))
    # The energy gate is deterministic. webrtcvad classifies *voicedness*, and
    # a synthetic burst is not speech to it however loud it is.
    cfg["capture"]["vad"]["backend"] = "energy"
    cfg["runtime"]["spill_dir"] = tempfile.mkdtemp()
    cfg.setdefault("server", {})["port"] = 8123

    path = pathlib.Path(tempfile.mkdtemp()) / "stub.yaml"
    path.write_text(yaml.safe_dump(cfg), "utf-8")

    import uvicorn

    from pipeline.server.app import create_app

    print("stub pipeline server on http://127.0.0.1:8123", flush=True)
    print("  models are faked; the socket and the VAD are real", flush=True)
    uvicorn.run(create_app(str(path)), host="127.0.0.1", port=8123,
                log_level="warning")


if __name__ == "__main__":
    main()
