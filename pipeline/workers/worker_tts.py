"""TTS worker — two voices behind one protocol, runs in tts/venv.

Indic-Mio is a 0.6B causal LM (fine-tuned from Aratako/MioTTS-0.6B) that
speaks all 22 scheduled Indian languages plus English from a single set of
weights. Unlike Piper there is no per-language voice file: language is
inferred by the model directly from the script of the input text, so
synthesis is just chat-template -> generate -> decode regardless of which
language the reply is in.

Generation produces ordinary text tokens interleaved with audio tokens in a
reserved id range; the audio tokens are extracted, shifted back to codec
codes, and decoded to a waveform by MioCodec.

What Indic-Mio cannot do is speak Spanish, Russian or Japanese — its language
set stops at India's border. Replies in those languages are synthesized by
ElevenLabs instead, chosen per utterance by `languages.route_for()` on the
reply's language. The two backends are deliberately interchangeable below the
waist: both produce a float32 waveform, both go through the same loudness
normalization, and both are written to the same WAV. That matters more than it
looks — `normalize()` is echo-reduction layer 1, and an un-normalized
international reply would come back hot enough to trip the barge-in gate and
cut itself off mid-sentence.

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

# stdout is the JSON protocol channel, so nothing else may write to it. torch
# and transformers print freely, and one stray line makes the parent's
# json.loads fail with "Expecting value: line 1 column 1", masking the real
# error. Keep the real stdout here and point sys.stdout at stderr so library
# chatter is logged instead of corrupting the stream.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr

# Runs in tts/venv and cannot import the orchestrator package by name, so the
# shared routing tables are loaded by path — the same trick the LLM and STT
# workers use. Both modules are stdlib-only for exactly this reason.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime.elevenlabs import ElevenLabs, ElevenLabsError  # noqa: E402
from realtime.languages import ROUTE_INTERNATIONAL, has_eleven_voice, route_for  # noqa: E402

# Indic-Mio's supported languages (frontmatter of the model card), mapped to
# this pipeline's language names from realtime/languages.py. A reply in a
# language outside this set has no voice, same as Piper's missing-voice gap.
SUPPORTED_LANGS = frozenset({
    "english", "hindi", "bengali", "marathi", "telugu", "kannada", "tamil",
    "malayalam", "gujarati", "punjabi", "odia", "urdu", "nepali",
    # The script-range detector buckets every Devanagari language under
    # "hindi" and every Bengali-script one under "bengali", so these names
    # never used to reach this worker. Audio-level LID names them directly
    # now, and Indic-Mio speaks all of them natively — so they have to be
    # listed, or a Marathi reply would be refused as having no voice.
    "assamese", "sanskrit", "konkani", "maithili", "dogri", "bodo",
    "santali", "sindhi", "manipuri", "kashmiri",
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

    # -- the international voice -------------------------------------------
    el_cfg = cfg.get("elevenlabs") or {}
    client = None
    if el_cfg.get("enabled", True):
        try:
            client = ElevenLabs(
                os.environ.get(el_cfg.get("api_key_env", "ELEVENLABS_API_KEY"), ""),
                tts_model=el_cfg.get("tts_model"),
                voice_id=el_cfg.get("voice_id"),
                output_format=el_cfg.get("output_format"),
                voice_settings=el_cfg.get("voice_settings"),
                timeout_s=el_cfg.get("tts_timeout_s", el_cfg.get("timeout_s", 20)),
                retries=el_cfg.get("retries", 1),
            )
        except ElevenLabsError as exc:
            log(f"ElevenLabs voice unavailable ({exc}); replies in languages "
                f"Indic-Mio does not speak will be text-only")

    def synthesize_eleven(text):
        """Same contract as synthesize(): (float32 mono waveform, sample rate).

        The response is raw little-endian PCM16 rather than MP3, so there is
        nothing to decode — just a reinterpretation of the same bytes into the
        float32 the normalizer and the WAV writer already expect.
        """
        raw, rate = client.synthesize(text)
        pcm = np.frombuffer(raw, dtype="<i2")
        return pcm.astype(np.float32) / 32768.0, rate

    emit({
        "event": "ready",
        "load_s": round(load_s, 2),
        "sample_rate": SAMPLE_RATE,
        "languages": sorted(SUPPORTED_LANGS),
        "elevenlabs": client is not None,
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
        # The reply's own language picks the voice, not the route the *user's*
        # audio took. They usually agree, but when they do not the reply is
        # what is about to be spoken — so a turn misrouted to Scribe and then
        # correctly identified as Tamil still comes back in the local voice.
        route = route_for(lang)
        backend = "elevenlabs" if route == ROUTE_INTERNATIONAL else "indic-mio"
        try:
            # Same gap, two voices: a language neither backend speaks is
            # reported rather than read aloud in the wrong one, which sounds
            # like a bug and mispronounces every word. The orchestrator prints
            # the reply as text instead.
            no_voice = None
            if route == ROUTE_INTERNATIONAL:
                if client is None:
                    no_voice = "elevenlabs is not configured"
                elif not has_eleven_voice(lang):
                    no_voice = "not one of the multilingual model's languages"
            elif lang not in SUPPORTED_LANGS:
                no_voice = "not one of Indic-Mio's languages"

            if no_voice:
                log(f"no {backend} voice for {lang!r} ({no_voice}); "
                    f"utterance not spoken")
                emit({"ok": False, "utt_id": req.get("utt_id"),
                      "error": "no_voice", "lang": lang, "reason": no_voice})
                continue

            t = time.time()
            if route == ROUTE_INTERNATIONAL:
                audio, rate = synthesize_eleven(req["text"])
            else:
                audio, rate = synthesize(req["text"])
            if audio.size == 0:
                emit({"ok": False, "utt_id": req.get("utt_id"),
                      "error": "empty_audio", "lang": lang})
                continue
            # Both backends normalize. Skipping it for ElevenLabs would put a
            # full-scale reply into a room whose barge-in gate was tuned
            # against a -23 LUFS one, and the reply would interrupt itself.
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
                "backend": backend,
                "audio_s": round(len(audio) / rate, 2),
                "elapsed_s": round(time.time() - t, 3),
            })
        except ElevenLabsError as exc:
            # A network failure is not a missing voice: the reply is speakable,
            # it just could not be fetched. Reported as a plain failure so it
            # does not read as an unsupported language.
            log(f"ElevenLabs synthesis failed: {exc}")
            emit({"ok": False, "utt_id": req.get("utt_id"),
                  "error": f"elevenlabs_tts: {exc}", "lang": lang})
        except Exception as exc:
            log(f"synthesis failed: {exc!r}")
            emit({"ok": False, "utt_id": req.get("utt_id"), "error": str(exc)})


if __name__ == "__main__":
    main()
