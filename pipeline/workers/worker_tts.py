"""TTS worker — Piper, runs in tts/venv.

Piper is a small VITS-based synthesizer that runs on CPU through onnxruntime.
One voice is one .onnx file plus a .onnx.json describing its phoneme
inventory and sample rate, so switching language means switching files rather
than reconditioning a single model.

Unlike a voice-cloning model there is no reference clip: the voice IS the
checkpoint. Language selection therefore happens here, by picking the voice
whose language matches the reply, and a language with no installed voice
cannot be spoken at all.

Voices load lazily and stay cached, because a cold load is ~0.3s while
synthesis of a short reply is ~0.1s — loading every voice up front would cost
more than most sessions ever use.

Protocol (JSON lines on stdin/stdout):
    <- {"cmd": "init", "config": {...}}
    -> {"event": "ready", ...}
    <- {"cmd": "run", "utt_id": ..., "text": ..., "lang": ..., "wav_path": ...}
    -> {"ok": true, "utt_id": ..., "wav_path": ..., "sample_rate": ...}
    -> {"ok": false, "utt_id": ..., "error": "no_voice", "lang": ...}
"""
import json
import os
import sys
import time
import wave

# stdout is the JSON protocol channel, so nothing else may write to it. Model
# and phonemizer libraries print freely, and one stray line makes the parent's
# json.loads fail with "Expecting value: line 1 column 1", masking the real
# error. Keep the real stdout here and point sys.stdout at stderr so library
# chatter is logged instead of corrupting the stream.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def emit(obj):
    _PROTOCOL.write(json.dumps(obj) + "\n")
    _PROTOCOL.flush()


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def main():
    line = sys.stdin.readline()
    if not line:
        return
    init = json.loads(line)
    cfg = init["config"]

    try:
        from piper import PiperVoice
    except ImportError:
        log("piper-tts is not installed in this venv. Install it with:")
        log(r"  tts\venv\Scripts\python.exe -m pip install piper-tts")
        emit({"event": "error", "error": "piper_missing"})
        return

    voices_dir = cfg["voices_dir"]
    if not os.path.isdir(voices_dir):
        log(f"voices dir not found: {voices_dir!r}")
        log("Download the Piper voices first:")
        log("  bash docs/piperInstallDocs.md")
        emit({"event": "error", "error": "voices_dir_missing"})
        return

    # language -> export dir name, from config. The mapping is data rather than
    # convention because Piper names voices by speaker ("lessac", "meera"),
    # which says nothing about the language they speak.
    voice_map = cfg.get("voices") or {}
    default_lang = cfg.get("default_language", "english")

    def voice_path(lang):
        """Resolve a language to its voice.onnx, or None if not installed."""
        name = voice_map.get(lang)
        if not name:
            return None
        path = os.path.join(voices_dir, name, "voice.onnx")
        return path if os.path.isfile(path) else None

    installed = sorted(l for l in voice_map if voice_path(l))
    if not installed:
        log(f"no voices found under {voices_dir}")
        log("Expected <voices_dir>/<name>/voice.onnx for each entry in "
            "tts.voices.")
        emit({"event": "error", "error": "no_voices_installed"})
        return

    missing = sorted(set(voice_map) - set(installed))
    if missing:
        # Not fatal — those languages simply cannot be spoken. Say so once at
        # startup rather than only when someone speaks one.
        log(f"voices configured but not on disk: {', '.join(missing)}")

    if default_lang not in installed:
        log(f"default_language {default_lang!r} has no installed voice; "
            f"falling back to {installed[0]!r}")
        default_lang = installed[0]

    t0 = time.time()
    cache = {}

    def load_voice(lang):
        if lang not in cache:
            path = voice_path(lang)
            if not path:
                return None
            t = time.time()
            cache[lang] = PiperVoice.load(path)
            log(f"loaded {lang} voice in {time.time() - t:.2f}s")
        return cache[lang]

    # Load the default eagerly so "ready" means genuinely ready, and so a
    # broken install fails at startup instead of on the first reply.
    try:
        default_voice = load_voice(default_lang)
    except Exception as exc:
        log(f"voice load failed: {exc!r}")
        emit({"event": "error", "error": f"load_failed: {exc}"})
        return
    load_s = time.time() - t0

    # Piper's sample rate is a property of each voice, not a global setting;
    # 22050 for the medium voices used here. Report the default's rate, and
    # send the actual rate with every utterance since it can differ per voice.
    sample_rate = default_voice.config.sample_rate

    # Generation settings are passed as a SynthesisConfig object rather than
    # kwargs. Leaving a field None makes Piper fall back to the value baked
    # into the voice, which differs per checkpoint — so only set what is
    # configured here.
    from piper.config import SynthesisConfig

    syn_kwargs = {}
    for key in ("length_scale", "noise_scale", "noise_w_scale"):
        if cfg.get(key) is not None:
            syn_kwargs[key] = float(cfg[key])
    if cfg.get("volume") is not None:
        syn_kwargs["volume"] = float(cfg["volume"])
    # normalize_audio would rescale every utterance to full scale, undoing the
    # loudness targeting below and re-introducing the echo it exists to avoid.
    syn_cfg = SynthesisConfig(normalize_audio=False, **syn_kwargs)

    import numpy as np

    # Echo reduction layer 1: the feedback loop is acoustic, so the most
    # effective software lever is amplitude. Quieter output that the user
    # amplifies at the system level keeps a better speaker-to-mic ratio than
    # hot output does.
    norm_cfg = cfg.get("normalize") or {}
    target_lufs = norm_cfg.get("target_lufs")
    peak_target = float(norm_cfg.get("peak", 0.0) or 0.0)
    meters = {}

    def normalize(audio, rate):
        """Bring output down to the configured loudness."""
        if target_lufs is not None:
            try:
                import pyloudnorm

                if rate not in meters:
                    meters[rate] = pyloudnorm.Meter(rate)
                loudness = meters[rate].integrated_loudness(audio)
                if np.isfinite(loudness):
                    audio = pyloudnorm.normalize.loudness(
                        audio, loudness, float(target_lufs)
                    )
            except Exception as exc:
                log(f"loudness normalization failed ({exc!r})")
        if peak_target:
            peak = float(np.abs(audio).max())
            if peak > peak_target:
                audio = audio * (peak_target / peak)
        # Normalization can push past full scale; clipping there would be
        # audible distortion, which is worse than being slightly quiet.
        return np.clip(audio, -1.0, 1.0)

    def synthesize(voice, text):
        """One utterance -> (float32 mono waveform, sample rate).

        synthesize() yields chunks so long replies can stream; this pipeline
        writes a whole file before playback, so concatenate them.
        """
        chunks = []
        rate = voice.config.sample_rate
        for chunk in voice.synthesize(text, syn_config=syn_cfg):
            chunks.append(np.asarray(chunk.audio_float_array, dtype=np.float32))
            rate = chunk.sample_rate
        if not chunks:
            return np.zeros(0, dtype=np.float32), rate
        return np.concatenate(chunks).squeeze(), rate

    emit({
        "event": "ready",
        "load_s": round(load_s, 2),
        "sample_rate": sample_rate,
        "languages": installed,
        "vram_gb": 0.0,   # Piper runs on CPU
    })

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        req = json.loads(line)
        cmd = req.get("cmd")

        if cmd == "shutdown":
            break
        if cmd != "run":
            continue

        lang = req.get("lang") or default_lang
        try:
            voice = load_voice(lang)
            if voice is None:
                # No voice for this language. Say so explicitly rather than
                # reading the reply aloud in the wrong language, which sounds
                # like a bug and mispronounces every word.
                log(f"no Piper voice for {lang!r}; utterance not spoken")
                emit({"ok": False, "utt_id": req.get("utt_id"),
                      "error": "no_voice", "lang": lang})
                continue

            t = time.time()
            audio, rate = synthesize(voice, req["text"])
            if audio.size == 0:
                emit({"ok": False, "utt_id": req.get("utt_id"),
                      "error": "empty_audio", "lang": lang})
                continue
            audio = normalize(audio, rate)

            # Write int16 via the stdlib rather than pulling in soundfile;
            # Piper is natively int16 and this is the only writer here.
            pcm = (audio * 32767.0).astype(np.int16)
            with wave.open(req["wav_path"], "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(rate)
                fh.writeframes(pcm.tobytes())

            emit({
                "ok": True,
                "utt_id": req["utt_id"],
                "wav_path": req["wav_path"],
                "sample_rate": rate,
                "lang": lang,
                "audio_s": round(len(audio) / rate, 2),
                "elapsed_s": round(time.time() - t, 3),
            })
        except Exception as exc:
            log(f"synthesis failed: {exc!r}")
            emit({"ok": False, "utt_id": req.get("utt_id"), "error": str(exc)})


if __name__ == "__main__":
    main()
