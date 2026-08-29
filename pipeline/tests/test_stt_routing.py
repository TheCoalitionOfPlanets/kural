"""Drives the real STT worker over its JSON protocol, with stubbed models.

    venv\\Scripts\\python.exe pipeline\\tests\\test_stt_routing.py

The routing decision, the Whisper call and the protocol replies are glued
together inside `worker_stt.main()`, which the flow test reaches around by
faking the worker entirely. This one runs the actual worker loop with `torch`
and `transformers` replaced, so the glue itself is what is under test —
including that Whisper is loaded lazily and only when a turn routes to it.
"""
import io
import json
import os
import sys
import tempfile
import types
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


class _Probs(list):
    """A probability row that supports the two things the worker does to one.

    `probs[idx]` for the argmax, and `probs[index_tensor].sum()` for the local
    mass — the latter is fancy indexing, which a plain list does not do, so it
    is modelled here rather than by pulling real torch into the test.
    """

    def __getitem__(self, key):
        if isinstance(key, _IndexTensor):
            return _Probs(list.__getitem__(self, i) for i in key.indices)
        return list.__getitem__(self, key)

    def sum(self):
        return _Scalar(sum(p.item() for p in self))


class _IndexTensor:
    """What `torch.tensor(sorted(ids))` returns in the stubbed world."""

    def __init__(self, indices):
        self.indices = list(indices)


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
    # The worker builds an index tensor of the LID labels that route locally,
    # then sums the probability sitting on them — the local-mass check that
    # keeps English at home. `device=` is accepted and ignored.
    t.tensor = lambda values, **kw: _IndexTensor(values)
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
        probs = _Probs(_Scalar(conf if i == idx else rest)
                       for i in range(len(self.config.id2label)))
        return types.SimpleNamespace(logits=_Logits([probs]))


class _StubExtractor:
    def __call__(self, clip, sampling_rate=None, return_tensors=None):
        return {"input_values": types.SimpleNamespace(to=lambda *a, **k: None)}


class _StubWhisper:
    """Stands in for Whisper large-v3.

    `heard` is what the test wants it to have transcribed; `loads` counts how
    many times it was constructed, which is what proves the lazy load happens
    once and only when a turn actually routes here.
    """
    heard = {"text": "¿Cómo estás?", "language_code": "es"}
    loads = 0
    device = "cpu"
    last_kwargs = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, features, **kwargs):
        _StubWhisper.last_kwargs = kwargs
        return types.SimpleNamespace(sequences=[["<|startoftranscript|>",
                                                 f"<|{self.heard['language_code']}|>"]])


class _StubWhisperProcessor:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(
            convert_ids_to_tokens=lambda seq: list(seq))

    def __call__(self, wav, sampling_rate=None, return_tensors=None):
        return types.SimpleNamespace(
            input_features=types.SimpleNamespace(to=lambda *a, **k: None))

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [_StubWhisper.heard["text"]]


def _new_whisper(*a, **k):
    _StubWhisper.loads += 1
    return _StubWhisper()


def _make_transformers():
    m = types.ModuleType("transformers")
    m.AutoModel = types.SimpleNamespace(from_pretrained=lambda *a, **k: _StubASR())
    m.AutoFeatureExtractor = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _StubExtractor())
    m.Wav2Vec2ForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _StubLID())
    m.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _StubWhisperProcessor())
    m.AutoModelForSpeechSeq2Seq = types.SimpleNamespace(
        from_pretrained=_new_whisper)
    return m


def run_worker(commands, whisper=None, config_overrides=None, lid_dir=True,
               whisper_dir=True):
    """Feed the real worker a script of commands; return its emitted lines.

    Returns (lines, whisper_loads) — the load count standing in for the "was
    the remote called" assertion the network-backed version used to make.
    """
    sys.modules["torch"] = _make_torch()
    sys.modules["transformers"] = _make_transformers()
    sys.modules.pop("worker_stt", None)
    import worker_stt  # noqa: E402  (imported after the stubs are in place)

    _StubWhisper.heard = whisper or {"text": "¿Cómo estás?",
                                     "language_code": "es"}
    _StubWhisper.loads = 0
    _StubWhisper.last_kwargs = None

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "model_dir": tmp,
            "device": "cpu",
            "require_cuda": False,
            "lid": {"enabled": True, "model_dir": tmp if lid_dir else "/nope",
                    "min_audio_s": 0.7, "min_confidence": 0.55,
                    "sticky_ttl_s": 60, "hint_confidence": 0.85},
            "whisper": {"enabled": True,
                        "model_dir": tmp if whisper_dir else "/nope"},
        }
        cfg.update(config_overrides or {})

        script = [{"cmd": "init", "config": cfg}] + commands + [{"cmd": "shutdown"}]
        stdin = io.StringIO("\n".join(json.dumps(c) for c in script) + "\n")
        stdout = io.StringIO()

        real = (sys.stdin, sys.stdout, sys.stderr)
        sys.stdin, sys.stdout = stdin, stdout
        sys.stderr = io.StringIO()
        try:
            worker_stt.main()
        finally:
            sys.stdin, sys.stdout, sys.stderr = real

    lines = [json.loads(x) for x in stdout.getvalue().splitlines() if x.strip()]
    return lines, _StubWhisper.loads


def utterance(seconds, tmpdir):
    path = Path(tmpdir) / "u.npy"
    np.save(path, np.zeros(int(16000 * seconds), dtype=np.float32))
    return {"cmd": "run", "utt_id": "u1", "pcm_path": str(path),
            "sample_rate": 16000}


scratch = tempfile.mkdtemp()

print("confident spanish — the local ear is never asked")
_StubLID.predicted = ("spa", 0.97)
lines, loads = run_worker([utterance(3.0, scratch)])
ready, result = lines[0], lines[1]
check("worker reports ready", ready.get("event") == "ready")
check("ready reports the gate is up", ready.get("lid") is True)
check("ready reports the set B ear is available",
      ready.get("international_stt") is True)
# Available, but not yet loaded — that is the whole point of the lazy path.
check("routed international", result.get("route") == "international")
check("backend named", result.get("backend") == "whisper")
check("whisper was loaded once", loads == 1)
check("whisper's text is returned", result.get("text") == "¿Cómo estás?")
check("whisper's language is returned", result.get("lang") == "spanish")
# Above hint_confidence the LID answer is worth passing along, translated from
# the 639-3 code LID emits to the 639-1 tag Whisper takes.
check("a confident prediction is sent as a hint",
      (_StubWhisper.last_kwargs or {}).get("language") == "es")

print("\nconfident tamil — the set B ear is never loaded")
_StubLID.predicted = ("tam", 0.94)
lines, loads = run_worker([utterance(3.0, scratch)])
result = lines[1]
check("routed local", result.get("route") == "local")
check("backend named", result.get("backend") == "sravaani")
check("whisper was never loaded", loads == 0)
check("the local transcript is returned",
      result.get("text") == _StubASR.transcript)
check("the language is carried", result.get("lang") == "tamil")

print("\nweak prediction — the gate declines, and declining is local")
_StubLID.predicted = ("spa", 0.40)
lines, loads = run_worker([utterance(3.0, scratch)])
check("routed local", lines[1].get("route") == "local")
check("whisper was never loaded", loads == 0)
# Nothing was established, so the LLM falls back to reading the transcript.
check("no language is claimed", lines[1].get("lang") is None)

print("\nbelow hint_confidence — routed, but with no language forced")
# Between min_confidence and hint_confidence: confident enough to route on, not
# confident enough to tell Whisper what it is hearing. Forcing a wrong language
# makes Whisper *translate* rather than refuse, which returns a plausible and
# entirely wrong transcript — so the hint is withheld and Whisper detects.
#
# 0.80 rather than something nearer min_confidence because the local-mass rule
# gets there first: the residual probability is spread over the stub's `tam`
# and `eng` labels, and below ~0.7 that residual alone clears min_local_mass
# and keeps the turn home. Which is the rule working — it just means this case
# has to be built above it to be about the hint at all.
_StubLID.predicted = ("spa", 0.80)
lines, loads = run_worker([utterance(3.0, scratch)])
check("routed international", lines[1].get("route") == "international")
check("no language was forced",
      "language" not in (_StubWhisper.last_kwargs or {}))

print("\nshort utterance — not worth classifying however confident")
_StubLID.predicted = ("spa", 0.99)
lines, loads = run_worker([utterance(0.4, scratch)])
check("routed local", lines[1].get("route") == "local")
check("whisper was never loaded", loads == 0)

print("\nhysteresis across turns — the worker holds the window")
_StubLID.predicted = ("spa", 0.97)
first = utterance(3.0, scratch)
lines, loads = run_worker([first, dict(first, utt_id="u2")])
check("both turns routed international",
      [l.get("route") for l in lines[1:3]] == ["international"] * 2)
# Two turns, one load: the model is cached after the first international turn,
# which is what makes the lazy path cheap rather than merely deferred.
check("whisper was loaded once, not per turn", loads == 1)

print("\nwhisper overrules the gate — a misroute repairs itself")
_StubLID.predicted = ("spa", 0.97)
lines, loads = run_worker(
    [utterance(3.0, scratch)],
    whisper={"text": "எப்படி இருக்கீங்க", "language_code": "ta"},
)
result = lines[1]
# LID sent it abroad; Whisper says Tamil, so the turn is Tamil from here on and
# TTS will route it back to the local voice.
check("the turn becomes tamil", result.get("lang") == "tamil")
check("and is reported as locally routed", result.get("route") == "local")

print("\nthe transcript outranks whisper's own label")
# Whisper misnames the language of short or noisy clips. The script of the text
# it produced cannot disagree with itself, so it wins when the two scripts are
# genuinely incompatible.
_StubLID.predicted = ("spa", 0.97)
lines, loads = run_worker(
    [utterance(3.0, scratch)],
    whisper={"text": "how are you doing today", "language_code": "ko"},
)
check("latin text is not called korean", lines[1].get("lang") == "english")

print("\nno whisper weights — identified, and honestly refused")
_StubLID.predicted = ("spa", 0.97)
lines, loads = run_worker([utterance(3.0, scratch)], whisper_dir=False)
ready, result = lines[0], lines[1]
check("ready reports the set B ear is unavailable",
      ready.get("international_stt") is False)
check("and explains why", "not found" in (ready.get("international_stt_error") or ""))
check("the turn is refused", result.get("ok") is False)
# The alternative is confident Indic gibberish, which reads as a working
# pipeline giving a strange answer.
check("refused for the right reason",
      result.get("error") == "no_international_stt")
check("the language is named", result.get("lang") == "spanish")
check("nothing was loaded", loads == 0)

print("\nno gate weights — everything routes locally, and says so")
_StubLID.predicted = ("spa", 0.97)
lines, loads = run_worker([utterance(3.0, scratch)], lid_dir=False)
ready, result = lines[0], lines[1]
check("ready reports the gate is down", ready.get("lid") is False)
check("and explains why", "not found" in (ready.get("lid_error") or ""))
check("the turn routes locally", result.get("route") == "local")
check("whisper was never loaded", loads == 0)
check("the local transcript is still returned",
      result.get("text") == _StubASR.transcript)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all stt-routing tests passed")
