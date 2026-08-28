"""Clone the local reference voice into ElevenLabs, so both stacks match.

The local and international halves of this pipeline use different TTS models,
and nothing makes them agree by itself: Indic-Mio clones zero-shot from
`tts.reference_wav`, while ElevenLabs speaks whichever `voice_id` it is given —
by default a stock voice that sounds like a different person entirely. Switching
languages mid-conversation therefore switches speaker, which is exactly what the
routing was built to make seamless.

This closes that gap. It uploads the *same* reference clip the local model is
conditioned on to ElevenLabs' Instant Voice Cloning, and prints the resulting
voice id to paste into `elevenlabs.voice_id`. One reference, two stacks, one
recognizable speaker.

Run it once, after setting the reference clip and an API key:

    venv/Scripts/python.exe pipeline/tools/clone_voice.py

What it cannot do is make them identical. IVC reproduces timbre, not the prosody
of another model: ElevenLabs speaking Spanish applies its own rhythm and stress.
Expect "the same person speaking another language", which is the ceiling for two
independent models — and is what a listener actually notices.

Requires a paid ElevenLabs tier (Starter and up); IVC is not available on the
free tier, and the API says so with an HTTP 401/403 rather than a clear message.
"""
import argparse
import json
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime.elevenlabs import (  # noqa: E402
    API_ROOT,
    ElevenLabsError,
    _multipart,
    _request,
)

# ElevenLabs wants at least a few seconds to model a speaker; below this the
# clone is audibly unstable, which is worse than not cloning at all.
MIN_SECONDS = 3.0
# Past a minute or so IVC gains nothing, and the upload just gets slower.
GOOD_SECONDS = 30.0


def _load_config(path):
    """Read the reference clip and key env var out of realtime.yaml.

    Deliberately a hand-rolled scan rather than a yaml import: this script runs
    from whichever venv the user happens to have activated, and pyyaml is not
    guaranteed in all three. Only two flat keys are needed.
    """
    reference, key_env = None, "ELEVENLABS_API_KEY"
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("reference_wav:"):
                    reference = stripped.split(":", 1)[1].strip().strip("'\"")
                elif stripped.startswith("api_key_env:"):
                    key_env = stripped.split(":", 1)[1].strip().strip("'\"")
    except FileNotFoundError:
        pass
    return reference, key_env


def _load_dotenv(path):
    """Make the key readable without a shell that has already exported it."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip())
    except FileNotFoundError:
        pass


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_cfg = os.path.join(root, "pipeline", "config", "realtime.yaml")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=default_cfg)
    ap.add_argument("--reference", help="override tts.reference_wav")
    ap.add_argument("--name", default="kural-local-voice",
                    help="name the cloned voice appears under in ElevenLabs")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="uploads are slower than a synthesis call")
    args = ap.parse_args()

    _load_dotenv(os.path.join(root, ".env"))
    cfg_reference, key_env = _load_config(args.config)

    reference = args.reference or cfg_reference
    if not reference:
        sys.exit("no reference clip: set tts.reference_wav or pass --reference")
    if not os.path.isabs(reference):
        reference = os.path.join(root, reference)
    if not os.path.isfile(reference):
        sys.exit(f"reference clip not found: {reference}")

    with wave.open(reference, "rb") as fh:
        seconds = fh.getnframes() / float(fh.getframerate())
    if seconds < MIN_SECONDS:
        sys.exit(f"{reference} is only {seconds:.1f}s; IVC needs at least "
                 f"{MIN_SECONDS:.0f}s of speech to model a speaker")
    if seconds < GOOD_SECONDS:
        print(f"note: {seconds:.1f}s is usable but short — {GOOD_SECONDS:.0f}s "
              f"of varied speech clones more faithfully", file=sys.stderr)

    api_key = os.environ.get(key_env, "")
    if not api_key or api_key.startswith("sk_123"):
        sys.exit(f"{key_env} is unset or still the placeholder from "
                 f".env.example — put a real key there first")

    with open(reference, "rb") as fh:
        audio = fh.read()

    body, content_type = _multipart(
        {"name": args.name,
         "description": "Local Indic-Mio reference voice, cloned so the "
                        "international path matches the local one."},
        [("files", os.path.basename(reference), "audio/wav", audio)],
    )

    print(f"uploading {seconds:.1f}s from {reference} ...", file=sys.stderr)
    try:
        raw, _ = _request(f"{API_ROOT}/voices/add", api_key, body, content_type,
                          args.timeout, retries=0, accept="application/json")
    except ElevenLabsError as exc:
        if exc.status in (401, 403):
            sys.exit(f"{exc}\n\nInstant Voice Cloning requires a paid tier "
                     f"(Starter and up). The rest of the pipeline still works "
                     f"on the free tier — international replies just use the "
                     f"stock voice in elevenlabs.voice_id.")
        sys.exit(str(exc))

    voice_id = (json.loads(raw.decode("utf-8")) or {}).get("voice_id")
    if not voice_id:
        sys.exit(f"upload succeeded but no voice_id came back: {raw[:400]!r}")

    print(f"\ncloned as {args.name!r}\n\nvoice_id: {voice_id}\n")
    print("Paste it into pipeline/config/realtime.yaml:\n")
    print(f"  elevenlabs:\n    voice_id: {voice_id}\n")


if __name__ == "__main__":
    main()
