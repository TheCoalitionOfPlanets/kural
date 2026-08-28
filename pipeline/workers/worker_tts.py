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

# Fallback only. The real rate comes from the codec's own config at load time
# (`_codec_sample_rate`), because the two MioCodec checkpoints differ: the
# wave-decoder one emits 24kHz, the mel one 44.1kHz.
SAMPLE_RATE = 24000


def _codec_sample_rate(config_path, default):
    """The codec's output rate, read from its own config.

    A tiny scan rather than a yaml import: this worker's venv is not guaranteed
    to have pyyaml, and only one scalar is needed. The first `sample_rate:` in
    the file is the codec's, and getting it wrong plays every reply at the
    wrong speed.
    """
    try:
        with open(config_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("sample_rate:"):
                    value = stripped.split(":", 1)[1].split("#")[0].strip()
                    if value:
                        return int(value)
    except (OSError, ValueError):
        pass
    return default


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

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.get("require_cuda", True) and not torch.cuda.is_available():
        log("CUDA required but unavailable")
        emit({"event": "error", "error": "cuda_unavailable"})
        return

    try:
        from miocodec import MioCodec  # noqa: F401  (kept for the error message)
        from miocodec.model import MioCodecModel
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
    # Indic-Mio's audio tokens are codec-specific: they index *this* codec's
    # quantizer codebook and mean nothing to another one. The model card names
    # MioCodec-25Hz-24kHz, so that is the only checkpoint whose decoder turns
    # them back into the words that were asked for. Decoding them with the
    # 44.1kHz codec does not fail — it produces fluent, correctly-timed,
    # completely wrong speech, which sounds like the model answering in some
    # other language.
    #
    # The two checkpoints are different architectures, not two sizes of one:
    #
    #   25Hz-24kHz     wave decoder + iSTFT head -> waveform directly.
    #                  `mel_decoder: null`, and no vocoder weights, because it
    #                  does not need one.
    #   25Hz-44.1kHz   mel decoder + a bundled `vocoder.`-prefixed vocoder.
    #
    # `MioCodec.from_pretrained` only knows the second shape and rejects the
    # first for having "no vocoder weights". So the wave-decoder checkpoint is
    # loaded through MioCodecModel directly, whose `decode()` returns a
    # waveform when the config sets `use_wave_decoder`.
    codec_dir = cfg["codec_dir"]
    codec_config = os.path.join(codec_dir, "config.yaml")
    codec_weights = os.path.join(codec_dir, "model.safetensors")
    vocoder_config = os.path.join(codec_dir, "vocoder_config.json")

    from safetensors.torch import load_file as _load_safetensors

    _state = _load_safetensors(codec_weights, device="cpu")
    uses_vocoder = any(k.startswith("vocoder.") for k in _state)
    if uses_vocoder:
        codec = MioCodec.from_pretrained(
            config_path=codec_config,
            weights_path=codec_weights,
            vocoder_config_path=(vocoder_config
                                 if os.path.isfile(vocoder_config) else None),
        )
        codec_model = codec.model
        log(f"codec {os.path.basename(codec_dir)}: mel decoder + vocoder")
    else:
        codec = None
        codec_model = MioCodecModel.from_hparams(codec_config)
        # strict=False: the SSL feature extractor is an *encoder*-side module
        # and is simply absent from this checkpoint. Nothing on the synthesis
        # path reads it, so its absence is expected rather than a partial load.
        codec_model.load_state_dict(_state, strict=False)
        codec_model = codec_model.to(device).eval()
        log(f"codec {os.path.basename(codec_dir)}: wave decoder (no vocoder)")
    del _state

    # The codec's own rate, not a constant: the wave-decoder checkpoint emits
    # 24kHz and the mel one 44.1kHz, and writing the wrong number into the WAV
    # header plays the reply at the wrong speed and pitch.
    sample_rate = _codec_sample_rate(codec_config, SAMPLE_RATE)

    # -- the voice ---------------------------------------------------------
    # Indic-Mio is zero-shot: with no reference speaker it invents one per
    # utterance, sampled at temperature 0.9, so consecutive replies come back
    # in different voices. The reference WAV below is encoded once at startup
    # and conditions every synthesis, which is what makes the assistant sound
    # like one person.
    reference_wav = cfg.get("reference_wav")
    reference = None
    speaker = None
    if reference_wav and os.path.isfile(reference_wav):
        with wave.open(reference_wav, "rb") as fh:
            if fh.getsampwidth() != 2 or fh.getnchannels() != 1:
                log(f"reference_wav must be 16-bit mono; {reference_wav} is "
                    f"{fh.getsampwidth() * 8}-bit {fh.getnchannels()}ch — ignoring")
            else:
                raw = fh.readframes(fh.getnframes())
                ref_rate = fh.getframerate()
                samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
                if ref_rate != sample_rate:
                    # The codec encodes at its own rate; a reference at another
                    # rate yields a speaker embedding for a voice pitched wrong.
                    idx = (np.arange(int(len(samples) * sample_rate / ref_rate))
                           * ref_rate / sample_rate).astype(int)
                    samples = samples[idx[idx < len(samples)]]
                reference = torch.from_numpy(samples).unsqueeze(0).to(device)
                # Encoded once here rather than per utterance: the embedding is
                # a property of the clip, and re-deriving it every reply would
                # pay for the same forward pass on every turn.
                speaker = codec_model.encode(
                    reference, return_content=False, return_global=True
                ).global_embedding
    elif reference_wav:
        log(f"reference_wav {reference_wav!r} not found; the local voice will "
            f"vary between utterances")

    load_s = time.time() - t0

    max_new_tokens = int(cfg.get("max_new_tokens", 1024))
    temperature = float(cfg.get("temperature", 0.9))
    top_p = float(cfg.get("top_p", 0.9))

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
            return np.zeros(0, dtype=np.float32), sample_rate

        codes_tensor = torch.tensor(audio_codes, dtype=torch.long, device=device)
        if speaker is None:
            # Without a reference there is no speaker embedding, and the codec
            # requires one — there is no "just decode the content" path. Say so
            # plainly rather than raising from inside the codec.
            raise RuntimeError(
                "no reference voice: set tts.reference_wav (see docs/voice.md)"
            )
        # Content tokens carry *what* was said; the speaker embedding supplies
        # who says it. Both codec shapes take the same two arguments here — the
        # wave decoder returns a waveform, the mel decoder a spectrogram that
        # still needs the vocoder run over it.
        out = codec_model.decode(
            global_embedding=speaker, content_token_indices=codes_tensor,
        )
        if codec is not None:
            from miocodec.util import vocode
            out = vocode(codec.vocoder, out.unsqueeze(0)).squeeze(0)
        audio = out.squeeze().to(torch.float32).cpu().numpy()
        return audio, sample_rate

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
            log(f"international voice unavailable ({exc}); replies in "
                f"languages Indic-Mio does not speak will be text-only")

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
        "sample_rate": sample_rate,
        "languages": sorted(SUPPORTED_LANGS),
        # False means the local voice is re-invented every utterance, which
        # sounds like a bug long before anyone suspects the config.
        "reference_voice": reference is not None,
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
        # stage="tts" because this is the *voice* decision, and Indic-Mio
        # speaks two languages SraVaani cannot hear (Urdu, Kashmiri). Asking
        # the combined set would send an Urdu reply to ElevenLabs even though
        # the local voice handles it.
        route = route_for(lang, stage="tts")
        backend = "elevenlabs" if route == ROUTE_INTERNATIONAL else "indic-mio"
        try:
            # Same gap, two voices: a language neither backend speaks is
            # reported rather than read aloud in the wrong one, which sounds
            # like a bug and mispronounces every word. The orchestrator prints
            # the reply as text instead.
            no_voice = None
            if route == ROUTE_INTERNATIONAL:
                if client is None:
                    no_voice = "the international voice is not configured"
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
            log(f"international synthesis failed: {exc}")
            emit({"ok": False, "utt_id": req.get("utt_id"),
                  "error": f"international_tts: {exc}", "lang": lang})
        except Exception as exc:
            log(f"synthesis failed: {exc!r}")
            emit({"ok": False, "utt_id": req.get("utt_id"), "error": str(exc)})


if __name__ == "__main__":
    main()
