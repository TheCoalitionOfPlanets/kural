"""Download the model weights the pipeline needs, and set the reference voice.

    venv/bin/python tools/fetch_models.py [--set-b] [--langs spa,fra,...]

Called by setup.sh and setup.bat so the two do not each carry their own copy of
this logic — batch and bash disagree about quoting badly enough that two copies
would drift, and the thing they would drift about is which weights land where.

Runs in the root environment, which is where `huggingface_hub` lives.

Every step checks for its own result first, so an interrupted download resumes
rather than starting over and a finished one is a no-op. Nothing retries: a
failure prints what to do and stops, because a setup that silently loops is
worse than one that stops.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Set A — what the pipeline needs to run as it ships. Each entry names the
# sentinel file that proves the download finished, so a half-written directory
# is re-fetched rather than trusted.
#
# (repo, destination, sentinel, label, allow_patterns)
SET_A = [
    (
        "ARTPARK-IISc/SraVaani-1.0",
        "stt/models",
        "stt/models/model-asr.fp16.ts",
        "SraVaani STT (~900 MB, gated)",
        None,
    ),
    (
        "google/gemma-3-4b-it",
        "reasoning/models/gemma-3-4b-it",
        "reasoning/models/gemma-3-4b-it/model.safetensors.index.json",
        "Gemma 3 4B (~8.6 GB, gated)",
        # Only what transformers loads — this skips any GGUF or ONNX variants
        # the repo carries, which the worker never touches.
        ["*.json", "*.safetensors", "*.model", "*.txt"],
    ),
    (
        "SPRINGLab/Indic-Mio",
        "tts/models/Indic-Mio",
        "tts/models/Indic-Mio/model.safetensors",
        "Indic-Mio TTS (~1.2 GB)",
        None,
    ),
    (
        # MUST be the 24kHz checkpoint. Indic-Mio's audio tokens index this
        # codec's codebook; the 44.1kHz one decodes them into fluent,
        # correctly-timed, completely wrong speech rather than failing.
        "Aratako/MioCodec-25Hz-24kHz",
        "tts/models/MioCodec-25Hz-24kHz",
        "tts/models/MioCodec-25Hz-24kHz/config.yaml",
        "MioCodec 25Hz-24kHz",
        None,
    ),
]

# Set B — the international stack. Suspended in realtime.yaml (stt.lid,
# stt.whisper and tts.mms_tts are all false), so it is opt-in.
SET_B = [
    (
        "facebook/mms-lid-126",
        "stt/models/mms-lid-126",
        "stt/models/mms-lid-126/config.json",
        "MMS language router (~1.2 GB)",
        ["*.json", "*.safetensors", "*.bin"],
    ),
    (
        "openai/whisper-large-v3",
        "stt/models/whisper-large-v3",
        "stt/models/whisper-large-v3/config.json",
        "Whisper large-v3 (~3.1 GB)",
        ["*.json", "*.safetensors", "*.txt"],
    ),
]

# MMS-TTS ships one checkpoint per language, so only the ones in use are worth
# fetching. These are ISO 639-3 codes from MMS_TTS_VOICES in
# pipeline/realtime/languages.py — note `cmn` for Mandarin (there is no `zho`
# checkpoint) and `tgl` for Filipino.
DEFAULT_LANGS = ["spa", "fra", "deu", "rus", "jpn", "cmn", "ara", "por"]


def step(text):
    print(f"\n==> {text}", flush=True)


def info(text):
    print(f"    {text}", flush=True)


def ok(text):
    print(f"    ok   {text}", flush=True)


def skip(text):
    print(f"    skip {text}", flush=True)


def warn(text):
    print(f"    warn {text}", file=sys.stderr, flush=True)


def download(repo, dest, sentinel, label, allow):
    """Fetch one repo unless its sentinel is already there."""
    step(label)
    if (ROOT / sentinel).is_file():
        skip(dest)
        return

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    info(repo)
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(ROOT / dest),
            allow_patterns=allow,
            token=token,
            max_workers=4,
        )
    except GatedRepoError:
        sys.exit(
            f"\n  {repo} is gated.\n"
            f"    1. Open https://huggingface.co/{repo} and accept the "
            f"license.\n"
            f"    2. Authenticate:  hf auth login   (or set HF_TOKEN)\n"
            f"    3. Re-run setup — finished steps are skipped.\n"
        )
    except RepositoryNotFoundError:
        sys.exit(
            f"\n  {repo} not found. If it is private, authenticate first:\n"
            f"    hf auth login   (or set HF_TOKEN)\n"
        )
    ok(dest)


def set_reference_voice():
    """Copy a sample clip out of Indic-Mio to become the assistant's voice.

    Indic-Mio is zero-shot and has no built-in speaker: with no reference clip
    it invents one per utterance at temperature 0.9, so consecutive replies
    come back in different voices — and the worker refuses to synthesize at
    all. So this is required rather than cosmetic.

    The clip is copied rather than referenced in place, so that re-downloading
    the model cannot silently change the assistant's voice.
    """
    step("Reference voice — tts/voice/reference.wav")
    target = ROOT / "tts/voice/reference.wav"
    if target.is_file():
        skip("already set")
        return

    model = ROOT / "tts/models/Indic-Mio"
    if not model.is_dir():
        warn("Indic-Mio is not downloaded yet; re-run without --skip-models.")
        return

    # sample1 is neutral Hindi/English code-mixed speech. sample3 and sample4
    # are emotion-tagged, and cloning is faithful enough that the tone carries
    # into every reply.
    preferred = model / "samples" / "sample1.wav"
    if preferred.is_file():
        source = preferred
    else:
        candidates = sorted(model.rglob("*.wav"))
        if not candidates:
            warn("No sample clip in the model. Supply your own 16-bit mono WAV:")
            warn("  cp your-voice.wav tts/voice/reference.wav")
            return
        source = candidates[0]

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    ok(f"copied {source.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Download the pipeline's model weights.",
    )
    ap.add_argument(
        "--set-b", action="store_true",
        help="also fetch the international stack (router, Whisper, MMS-TTS)",
    )
    ap.add_argument(
        "--langs", default=",".join(DEFAULT_LANGS),
        help="comma-separated ISO 639-3 codes for the MMS-TTS voices",
    )
    args = ap.parse_args()

    for entry in SET_A:
        download(*entry)

    if args.set_b:
        for entry in SET_B:
            download(*entry)

        langs = [c.strip() for c in args.langs.split(",") if c.strip()]
        step(f"Set B voices — facebook/mms-tts-* ({len(langs)}, ~145 MB each)")
        for code in langs:
            download(
                f"facebook/mms-tts-{code}",
                f"tts/models/mms-tts/{code}",
                f"tts/models/mms-tts/{code}/config.json",
                f"mms-tts-{code}",
                None,
            )

    set_reference_voice()


if __name__ == "__main__":
    main()
