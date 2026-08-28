"""Drives the real STT worker over its JSON protocol, with stubbed models.

    venv\\Scripts\\python.exe pipeline\\tests\\test_stt_routing.py

The routing decision, the Scribe call and the protocol replies are glued
together inside `worker_stt.main()`, which the other tests reach around: the
flow test fakes the worker, and the client test fakes the network. This one
runs the actual worker loop with `torch`, `transformers` and `urlopen` replaced,
so the glue itself is what is under test.
"""
import io
import json
import os
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


# -- the smallest torch that worker_stt actually uses ----------------------

class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class _Logits(list):
    def float(self):
        return self


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _make_torch():
    t = types.ModuleType("torch")
    t.float32, t.float16, t.bfloat16 = "f32", "f16", "bf16"
    t.cuda = types.SimpleNamespace(is_available=lambda: False,
                                   memory_allocated=lambda: 0)
    t.no_grad = _Ctx
    t.inference_mode = _Ctx
    t.softmax = lambda x, dim=-1: x
    t.argmax = lambda p: _Scalar(max(range(len(p)), key=lambda i: p[i].item()))
    return t


class _Hyp:
    def __init__(self, text):
        self.text = text


class _StubASR:
    """Stands in for SraVaani, which maps any audio onto Indic output."""
    transcript = "எப்படி இருக்கீங்க"

    def to(self, device):
        return self

    def eval(self):
        return self

    def _ensure_loaded(self):
        return None

    def transcribe(self, wavs, return_hypotheses=False):
        return [_Hyp(self.transcript)]


class _StubLID:
    """Returns whatever the test set on the class, as one-hot-ish probs."""
    predicted = ("spa", 0.97)
    device = "cpu"
    config = types.SimpleNamespace(id2label={0: "spa", 1: "tam", 2: "eng"})

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        code, conf = _StubLID.predicted
        idx = {v: k for k, v in self.config.id2label.items()}[code]
        rest = (1.0 - conf) / (len(self.config.id2label) - 1)
        probs = [_Scalar(conf if i == idx else rest)
                 for i in range(len(self.config.id2label))]
        return types.SimpleNamespace(logits=_Logits([probs]))


class _StubExtractor:
    def __call__(self, clip, sampling_rate=None, return_tensors=None):
        return {"input_values": types.SimpleNamespace(to=lambda *a, **k: None)}


def _make_transformers():
    m = types.ModuleType("transformers")
    m.AutoModel = types.SimpleNamespace(from_pretrained=lambda *a, **k: _StubASR())
    m.AutoFeatureExtractor = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _StubExtractor())
    m.Wav2Vec2ForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _StubLID())
    return m


class _FakeResponse:
    def __init__(self, body):
        self._body = body
        self.headers = {"Content-Type": "application/json"}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_worker(commands, scribe=None, config_overrides=None, lid_dir=True):
    """Feed the real worker a script of commands; return its emitted lines."""
    sys.modules["torch"] = _make_torch()
    sys.modules["transformers"] = _make_transformers()
    sys.modules.pop("worker_stt", None)
    import worker_stt  # noqa: E402  (imported after the stubs are in place)

    scribe_calls = []

    def fake_urlopen(req, timeout=None):
        scribe_calls.append(req)
        return _FakeResponse(json.dumps(
            scribe or {"text": "¿Cómo estás?", "language_code": "spa",
                       "language_probability": 0.99}).encode())

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "model_dir": tmp,
            "device": "cpu",
            "require_cuda": False,
            "lid": {"enabled": True, "model_dir": tmp if lid_dir else "/nope",
                    "min_audio_s": 0.7, "min_confidence": 0.55,
                    "sticky_ttl_s": 60, "hint_confidence": 0.85},
            "elevenlabs": {"enabled": True, "api_key_env": "KURAL_TEST_KEY"},
        }
        cfg.update(config_overrides or {})

        script = [{"cmd": "init", "config": cfg}] + commands + [{"cmd": "shutdown"}]
        stdin = io.StringIO("\n".join(json.dumps(c) for c in script) + "\n")
        stdout = io.StringIO()

        real = (sys.stdin, sys.stdout, sys.stderr, urllib.request.urlopen)
        urllib.request.urlopen = fake_urlopen
        sys.stdin, sys.stdout = stdin, stdout
        sys.stderr = io.StringIO()
        try:
            worker_stt.main()
        finally:
            sys.stdin, sys.stdout, sys.stderr, urllib.request.urlopen = real

    lines = [json.loads(x) for x in stdout.getvalue().splitlines() if x.strip()]
    return lines, scribe_calls


def utterance(seconds, tmpdir):
    path = Path(tmpdir) / "u.npy"
    np.save(path, np.zeros(int(16000 * seconds), dtype=np.float32))
    return {"cmd": "run", "utt_id": "u1", "pcm_path": str(path),
            "sample_rate": 16000}


os.environ["KURAL_TEST_KEY"] = "sk_test"
scratch = tempfile.mkdtemp()

print("confident spanish — the local ear is never asked")
_StubLID.predicted = ("spa", 0.97)
lines, calls = run_worker([utterance(3.0, scratch)])
ready, result = lines[0], lines[1]
check("worker reports ready", ready.get("event") == "ready")
check("ready reports the gate is up", ready.get("lid") is True)
check("ready reports elevenlabs is up", ready.get("elevenlabs") is True)
check("routed international", result.get("route") == "international")
check("backend named", result.get("backend") == "elevenlabs")
check("scribe was called", len(calls) == 1)
check("scribe's text is returned", result.get("text") == "¿Cómo estás?")
check("scribe's language is returned", result.get("lang") == "spanish")
# Above hint_confidence the LID answer is worth passing along.
check("a confident prediction is sent as a hint",
      calls and b'name="language_code"' in calls[0].data)

print("\nconfident tamil — nothing is sent anywhere")
_StubLID.predicted = ("tam", 0.94)
lines, calls = run_worker([utterance(3.0, scratch)])
result = lines[1]
check("routed local", result.get("route") == "local")
check("backend named", result.get("backend") == "sravaani")
check("no api call was made", not calls)
check("the local transcript is returned",
      result.get("text") == _StubASR.transcript)
check("the language is carried", result.get("lang") == "tamil")

print("\nweak prediction — the gate declines, and declining is local")
_StubLID.predicted = ("spa", 0.40)
lines, calls = run_worker([utterance(3.0, scratch)])
check("routed local", lines[1].get("route") == "local")
check("no api call was made", not calls)
# Nothing was established, so the LLM falls back to reading the transcript.
check("no language is claimed", lines[1].get("lang") is None)

print("\nshort utterance — not worth classifying however confident")
_StubLID.predicted = ("spa", 0.99)
lines, calls = run_worker([utterance(0.4, scratch)])
check("routed local", lines[1].get("route") == "local")
check("no api call was made", not calls)

print("\nhysteresis across turns — the worker holds the window")
_StubLID.predicted = ("spa", 0.97)
first = utterance(3.0, scratch)
lines, calls = run_worker([first, dict(first, utt_id="u2")])
check("both turns routed international",
      [l.get("route") for l in lines[1:3]] == ["international"] * 2)
check("two api calls", len(calls) == 2)

print("\nscribe overrules the gate — a misroute repairs itself")
_StubLID.predicted = ("spa", 0.97)
lines, calls = run_worker(
    [utterance(3.0, scratch)],
    scribe={"text": "எப்படி இருக்கீங்க", "language_code": "tam",
            "language_probability": 0.95},
)
result = lines[1]
# LID sent it abroad; Scribe says Tamil, so the turn is Tamil from here on and
# TTS will route it back to the local voice.
check("the turn becomes tamil", result.get("lang") == "tamil")
check("and is reported as locally routed", result.get("route") == "local")

print("\nno api key — identified, and honestly refused")
os.environ.pop("KURAL_TEST_KEY", None)
_StubLID.predicted = ("spa", 0.97)
lines, calls = run_worker([utterance(3.0, scratch)])
ready, result = lines[0], lines[1]
check("ready reports elevenlabs is down", ready.get("elevenlabs") is False)
check("the turn is refused", result.get("ok") is False)
# The alternative is confident Indic gibberish, which reads as a working
# pipeline giving a strange answer.
check("refused for the right reason",
      result.get("error") == "no_international_stt")
check("the language is named", result.get("lang") == "spanish")
check("no api call was attempted", not calls)
os.environ["KURAL_TEST_KEY"] = "sk_test"

print("\nno gate weights — everything routes locally, and says so")
_StubLID.predicted = ("spa", 0.97)
lines, calls = run_worker([utterance(3.0, scratch)], lid_dir=False)
ready, result = lines[0], lines[1]
check("ready reports the gate is down", ready.get("lid") is False)
check("and explains why", "not found" in (ready.get("lid_error") or ""))
check("the turn routes locally", result.get("route") == "local")
check("no api call was made", not calls)
check("the local transcript is still returned",
      result.get("text") == _StubASR.transcript)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all stt-routing tests passed")
