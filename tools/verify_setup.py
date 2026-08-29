"""Smoke-test the installation without loading a model onto the GPU.

    venv/bin/python tools/verify_setup.py

Called by setup.sh and setup.bat as their last step. Runs in the root
environment and reaches the other two by subprocess, so one script checks all
three rather than each shell re-implementing the same checks.

What it proves:

  * each environment imports what its worker imports
  * the interpreters named in realtime.yaml actually resolve on this platform
  * the weights are where the config says they are
  * the reference clip is a shape the TTS worker will accept
  * CUDA is present, which every stage requires

What it deliberately does not do is load a model. That takes minutes and
several GB of VRAM, and every failure it would catch that this does not is a
failure the first real run reports just as clearly.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fetch_models import SET_A, SET_B  # noqa: E402  (sibling, lazy imports)

# What each environment has to be able to import, taken from what the workers
# and the orchestrator actually import rather than from the requirements files
# — a package can be installed and still be broken, and it is the import that
# the pipeline depends on.
ENVIRONMENTS = [
    ("stt", "root       orchestrator + STT",
     "numpy torch transformers sounddevice soundfile yaml webrtcvad "
     "fastapi uvicorn"),
    ("llm", "reasoning  Gemma 3 4B",
     "numpy torch transformers bitsandbytes accelerate"),
    ("tts", "tts        Indic-Mio + MioCodec",
     "numpy torch transformers miocodec safetensors pyloudnorm"),
]

# The Set B model directories, and the config flag that decides whether they
# are required. Enabled-but-missing is a real failure; disabled-and-missing is
# the state the repo ships in.
OPTIONAL = [
    (("stt", "lid"), "language router"),
    (("stt", "whisper"), "Set B ear (Whisper)"),
    (("tts", "mms_tts"), "Set B voice (MMS-TTS)"),
]

failures = []


def ok(text):
    print(f"    ok   {text}", flush=True)


def note(text):
    print(f"    ..   {text}", flush=True)


def bad(text, detail=""):
    # Everything goes to stdout, failures included. Splitting them across two
    # streams reorders the report the moment either one is redirected, and a
    # setup log that reads out of order is worse than one that is not colour
    # coded. The exit status is what the caller branches on.
    print(f"    FAIL {text}", flush=True)
    if detail:
        for line in detail.strip().splitlines()[-3:]:
            print(f"         {line}", flush=True)
    failures.append(text)


def main():
    import yaml

    from pipeline.realtime.session import venv_python

    cfg = yaml.safe_load((ROOT / "pipeline/config/realtime.yaml").read_text())
    print("\n==> Verifying", flush=True)

    # -- the three environments -------------------------------------------
    interpreters = {}
    for key, label, modules in ENVIRONMENTS:
        # realtime.yaml names the Windows layout; venv_python resolves it to
        # whichever exists here. Checking it now means a missing environment is
        # reported as itself, rather than as a Popen failure a minute into the
        # first run.
        python = venv_python(ROOT, cfg[key]["python"])
        if not python.exists():
            bad(f"{label}: no interpreter at {python}")
            continue
        interpreters[key] = python

        proc = subprocess.run(
            [str(python), "-c", "import " + ", ".join(modules.split())],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            ok(label)
        else:
            bad(f"{label}: import failed", proc.stderr)

    # -- weights -----------------------------------------------------------
    for _repo, _dest, sentinel, label, _allow in SET_A:
        name = label.split(" (")[0]
        if (ROOT / sentinel).is_file():
            ok(f"{name} weights")
        else:
            bad(f"{name} weights missing ({sentinel})")

    # -- the optional international stack ----------------------------------
    set_b_sentinels = {dest: sentinel for _r, dest, sentinel, _l, _a in SET_B}
    for (section, sub), label in OPTIONAL:
        block = (cfg.get(section) or {}).get(sub) or {}
        model_dir = block.get("model_dir", "")
        enabled = bool(block.get("enabled", False))
        sentinel = set_b_sentinels.get(model_dir)

        if not enabled:
            note(f"{label}: suspended in realtime.yaml")
            continue
        # mms_tts points at a directory of per-language checkpoints rather than
        # one model, so its presence is the directory being non-empty.
        present = ((ROOT / sentinel).is_file() if sentinel
                   else any((ROOT / model_dir).glob("*/config.json")))
        if present:
            ok(f"{label}: enabled, weights present")
        else:
            bad(f"{label}: enabled in realtime.yaml but {model_dir} is empty",
                "Re-run setup with --with-set-b, or set enabled: false.")

    # -- the reference voice ----------------------------------------------
    # Without one the TTS worker raises on every utterance; with a wrong-shaped
    # one it logs and carries on, and the voice drifts between replies.
    reference = ROOT / (cfg["tts"].get("reference_wav") or "")
    if not reference.is_file():
        bad(f"reference voice missing ({reference.relative_to(ROOT)})",
            "The TTS worker refuses to synthesize without it.")
    else:
        import wave

        with wave.open(str(reference), "rb") as fh:
            width, channels = fh.getsampwidth(), fh.getnchannels()
            seconds = fh.getnframes() / float(fh.getframerate() or 1)
        if width != 2 or channels != 1:
            bad(f"reference voice is {width * 8}-bit {channels}ch",
                "The worker needs 16-bit mono and will ignore this clip.")
        elif seconds < 3:
            bad(f"reference voice is only {seconds:.1f}s",
                "3s or more clones far more stably.")
        else:
            ok(f"reference voice ({seconds:.1f}s, 16-bit mono)")

    # -- CUDA --------------------------------------------------------------
    # Every stage sets require_cuda, so this is reported here rather than left
    # to fail the first run.
    if "stt" in interpreters:
        proc = subprocess.run(
            [str(interpreters["stt"]), "-c",
             "import torch; print(torch.cuda.get_device_name(0)) "
             "if torch.cuda.is_available() else exit(1)"],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            ok(f"CUDA: {proc.stdout.strip()}")
        else:
            bad("CUDA unavailable",
                "Every stage sets require_cuda and will refuse to start.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
