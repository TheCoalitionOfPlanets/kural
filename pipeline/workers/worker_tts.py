"""TTS worker — SPRINGLab/Indic-Mio, runs in tts/venv.

Indic-Mio is a 0.6B causal LM (fine-tuned from Aratako/MioTTS-0.6B) that
speaks all 22 scheduled Indian languages plus English from a single set of
weights. Unlike Piper there is no per-language voice file: language is
inferred by the model directly from the script of the input text, so
synthesis is just chat-template -> generate -> decode regardless of which
language the reply is in.

Generation produces ordinary text tokens interleaved with audio tokens in a
reserved id range; the audio tokens are extracted, shifted back to codec
codes, and decoded to a waveform by MioCodec.

Protocol (JSON lines on stdin/stdout):
    <- {"cmd": "init", "config": {...}}
    -> {"event": "ready", ...}
    <- {"cmd": "run", "utt_id": ..., "text": ..., "lang": ..., "wav_path": ...}
    -> {"ok": true, "utt_id": ..., "wav_path": ..., "sample_rate": ...}
    -> {"ok": false, "utt_id": ..., "error": "no_voice", "lang": ...}
"""
import json
import sys
import time
import wave

# stdout is the JSON protocol channel, so nothing else may write to it. torch
# and transformers print freely, and one stray line makes the parent's
# json.loads fail with "Expecting value: line 1 column 1", masking the real
# error. Keep the real stdout here and point sys.stdout at stderr so library
# chatter is logged instead of corrupting the stream.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr

# Indic-Mio's supported languages (frontmatter of the model card), mapped to
# this pipeline's language names from realtime/languages.py. A reply in a
# language outside this set has no voice, same as Piper's missing-voice gap.
SUPPORTED_LANGS = frozenset({
    "english", "hindi", "bengali", "marathi", "telugu", "kannada", "tamil",
    "malayalam", "gujarati", "punjabi", "odia", "urdu", "nepali",
    # Devanagari/Bengali-script languages the pipeline's script-range
    # detector already buckets under "hindi"/"bengali" (see the comment on
    # _SCRIPT_RANGES in realtime/languages.py) but that Indic-Mio also
    # covers natively: Maithili, Assamese, Sanskrit, Konkani, Dogri, Bodo,
    # Santali, Sindhi, Manipuri, Kashmiri.
    "assamese",
})

# Audio tokens occupy a reserved id range above the text vocabulary; the
# model interleaves them with ordinary text tokens during generation.
SPEECH_OFFSET = 151669
SPEECH_RANGE = 12800

# Fixed by the model/codec pairing (Aratako/MioCodec-25Hz-24kHz decodes to
# 44.1kHz), not a per-voice property to read off a config like Piper's.
SAMPLE_RATE = 44100


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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.get("require_cuda", True) and not torch.cuda.is_available():
        log("CUDA required but unavailable")
        emit({"event": "error", "error": "cuda_unavailable"})
        return

    try:
        from miocodec import MioCodec
    except ImportError:
        log("miocodec is not installed in this venv. Install it with:")
        log(r"  tts\venv\Scripts\python.exe -m pip install "
            "git+https://github.com/Aratako/MioCodec")
        emit({"event": "error", "error": "miocodec_missing"})
        return

    device = cfg.get("device", "cuda:0")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_dir"], dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    codec = MioCodec.from_pretrained(cfg["codec_dir"])
    load_s = time.time() - t0

    max_new_tokens = int(cfg.get("max_new_tokens", 1024))
    temperature = float(cfg.get("temperature", 0.9))
    top_p = float(cfg.get("top_p", 0.9))

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

    def synthesize(text):
        """One utterance -> (float32 mono waveform, sample rate)."""
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        audio_codes = [
            t.item() - SPEECH_OFFSET for t in generated
            if SPEECH_OFFSET <= t.item() < SPEECH_OFFSET + SPEECH_RANGE
        ]
        if not audio_codes:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE

        codes_tensor = torch.tensor([audio_codes], dtype=torch.long).unsqueeze(0)
        wav = codec.decode(codes_tensor)
        audio = wav.squeeze().to(torch.float32).cpu().numpy()
        return audio, SAMPLE_RATE

    emit({
        "event": "ready",
        "load_s": round(load_s, 2),
        "sample_rate": SAMPLE_RATE,
        "languages": sorted(SUPPORTED_LANGS),
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
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

        lang = req.get("lang") or "english"
        try:
            if lang not in SUPPORTED_LANGS:
                # No voice for this language. Say so explicitly rather than
                # reading the reply aloud in the wrong language, which sounds
                # like a bug and mispronounces every word.
                log(f"Indic-Mio has no voice for {lang!r}; utterance not spoken")
                emit({"ok": False, "utt_id": req.get("utt_id"),
                      "error": "no_voice", "lang": lang})
                continue

            t = time.time()
            audio, rate = synthesize(req["text"])
            if audio.size == 0:
                emit({"ok": False, "utt_id": req.get("utt_id"),
                      "error": "empty_audio", "lang": lang})
                continue
            audio = normalize(audio, rate)

            # Write int16 via the stdlib rather than pulling in soundfile;
            # this is the only writer here.
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
