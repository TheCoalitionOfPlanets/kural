"""STT worker — SraVaani plus the language gate, runs in the root venv.

SraVaani is an Indic ASR: it hears the scheduled Indian languages and English
and nothing else. Hand it Spanish and it does not report failure — it emits
confident-looking Devanagari or Latin gibberish. So the transcript cannot be
used to decide whether the transcript should have been made here at all, and
the routing decision has to be made *before* transcription, from the waveform.

That is what the LID gate is. `facebook/mms-lid-126` classifies the language of
the audio directly, in one small forward pass, and its answer chooses the ear:

    indic / english  ->  SraVaani, locally, free
    anything else    ->  ElevenLabs Scribe

LID only decides the *route*. On the international path Scribe's own detected
language is what the rest of the turn runs on, because it is the better
detector — which is also how a misroute repairs itself: if LID says Spanish and
Scribe says Tamil, the turn is Tamil from here on and TTS goes back to the
local voice.

Protocol (JSON lines on stdin/stdout):
    <- {"cmd": "init", "config": {...}}
    -> {"event": "ready", ...}
    <- {"cmd": "run", "utt_id": ..., "pcm_path": "...npy", "sample_rate": 16000}
    -> {"ok": true, "utt_id": ..., "text": ..., "lang": ..., "route": ...}
    -> {"ok": false, "utt_id": ..., "error": "no_international_stt", ...}
"""
import json
import os
import sys
import time

import numpy as np

# This worker runs in the root venv and cannot import the orchestrator package
# by name, so the shared routing tables are loaded by path — same as the LLM
# worker does with the reply-language policy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime.elevenlabs import ElevenLabs, ElevenLabsError, pcm16_to_wav  # noqa: E402
from realtime.languages import (  # noqa: E402
    ROUTE_INTERNATIONAL,
    ROUTE_LOCAL,
    RouteGate,
    detect_language,
    language_from_code,
    route_for,
    script_of,
)

# MMS-LID was trained on 16kHz speech, which is also SraVaani's rate and the
# capture rate in realtime.yaml. Anything else would need resampling, and a
# silently wrong rate makes LID confidently wrong rather than obviously broken.
LID_SAMPLE_RATE = 16000


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _load_lid(cfg, torch):
    """Load the language-ID model, or return None with a loud explanation.

    A missing LID model is not fatal. The pipeline still runs — it just runs
    local-only, which is exactly what it did before this path existed. Failing
    startup over an optional model would be a worse trade than saying so and
    carrying on, but it must be *said*, or international turns silently become
    Indic gibberish and look like an ASR bug.
    """
    lid_cfg = cfg.get("lid") or {}
    if not lid_cfg.get("enabled", True):
        return None, "disabled in config"

    model_dir = lid_cfg.get("model_dir")
    if not model_dir or not os.path.isdir(model_dir):
        return None, (f"model_dir {model_dir!r} not found — run "
                      f"`bash download.sh --skip-venvs` to fetch it")

    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(lid_cfg.get("dtype", "float32"),
                                           torch.float32)
    extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir, dtype=dtype)
    model = model.to(cfg.get("device", "cuda"))
    model.eval()
    return (extractor, model, dtype), None


def main():
    line = sys.stdin.readline()
    if not line:
        return
    init = json.loads(line)
    cfg = init["config"]

    import torch
    from transformers import AutoModel

    if cfg.get("require_cuda", True) and not torch.cuda.is_available():
        log("CUDA required but unavailable")
        emit({"event": "error", "error": "cuda_unavailable"})
        return

    device = cfg.get("device", "cuda")
    t0 = time.time()
    model = AutoModel.from_pretrained(cfg["model_dir"], trust_remote_code=True)
    model = model.to(device)
    model.eval()
    # The TorchScript graphs load lazily on first use; force it now so the
    # first utterance is not charged for model load.
    model._ensure_loaded()

    lid, lid_error = _load_lid(cfg, torch)
    if lid_error:
        log(f"language ID unavailable ({lid_error}); every turn will be "
            f"routed to the local Indic models")
    load_s = time.time() - t0

    lid_cfg = cfg.get("lid") or {}
    # The routing policy lives in realtime/languages.py so it can be tuned and
    # tested without a GPU; what stays here is the forward pass that feeds it.
    gate = RouteGate(
        min_confidence=lid_cfg.get("min_confidence", 0.55),
        min_audio_s=lid_cfg.get("min_audio_s", 0.7),
        sticky_ttl_s=lid_cfg.get("sticky_ttl_s", 60),
        min_local_mass=lid_cfg.get("min_local_mass", 0.30),
    )
    # LID needs a few seconds, not the whole turn; capping bounds the forward
    # pass so a 15s utterance costs the same as a 3s one.
    max_audio_s = float(lid_cfg.get("max_audio_s", 5.0))
    # Above this the LID answer is passed to Scribe as a hint. Below it Scribe
    # is left to detect on its own, since a confident wrong hint is worse than
    # no hint at all.
    hint_confidence = float(lid_cfg.get("hint_confidence", 0.85))

    # -- ElevenLabs -------------------------------------------------------
    el_cfg = cfg.get("elevenlabs") or {}
    api_key = os.environ.get(el_cfg.get("api_key_env", "ELEVENLABS_API_KEY"), "")
    client = None
    if el_cfg.get("enabled", True):
        try:
            client = ElevenLabs(
                api_key,
                stt_model=el_cfg.get("stt_model"),
                timeout_s=el_cfg.get("stt_timeout_s", el_cfg.get("timeout_s", 20)),
                retries=el_cfg.get("retries", 1),
            )
        except ElevenLabsError as exc:
            log(f"international speech-to-text unavailable ({exc}); "
                f"international speech will be reported, not transcribed")

    # Which of MMS-LID's 126 labels the local ear can handle. Built once, from
    # the model's own label list, so it cannot drift from LOCAL_STT.
    local_label_ids = None

    def _local_label_ids(lid_model):
        """Indices of labels that route to the local stack."""
        ids = [
            i for i, code in lid_model.config.id2label.items()
            if route_for(language_from_code(code), stage="stt") == ROUTE_LOCAL
        ]
        return torch.tensor(sorted(ids), device=lid_model.device)

    def identify(wav):
        """(language, code, confidence, local_mass) for this waveform.

        `local_mass` is the summed probability of every label the local ear
        handles. It matters more than the top-1 label: only 15 of 126 labels
        are local, so an English turn whose mass is spread over `eng`, `cym`
        and `nno` can lose the argmax to a foreign label while local remains
        the right route by a wide margin.
        """
        nonlocal local_label_ids
        if lid is None:
            return None, None, 0.0, None
        extractor, lid_model, dtype = lid
        clip = wav[:int(max_audio_s * LID_SAMPLE_RATE)]
        inputs = extractor(clip, sampling_rate=LID_SAMPLE_RATE,
                           return_tensors="pt")
        inputs = {k: v.to(lid_model.device) for k, v in inputs.items()}
        if dtype != torch.float32:
            inputs["input_values"] = inputs["input_values"].to(dtype)
        with torch.inference_mode():
            logits = lid_model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1)[0]
        if local_label_ids is None:
            local_label_ids = _local_label_ids(lid_model)
        local_mass = float(probs[local_label_ids].sum().item())
        idx = int(torch.argmax(probs).item())
        code = lid_model.config.id2label[idx]
        return (language_from_code(code), code, float(probs[idx].item()),
                local_mass)

    def decide(wav, duration_s):
        """Route this utterance. Returns (route, lang, code, confidence)."""
        # Asking the gate first skips a forward pass whose answer it would
        # discard anyway — every short "mm" would otherwise pay for one.
        if lid is None or not gate.should_identify(duration_s):
            return ROUTE_LOCAL, None, None, 0.0
        predicted, code, confidence, local_mass = identify(wav)
        route, lang = gate.decide(predicted, confidence, duration_s,
                                  local_mass=local_mass)
        return route, lang, code, confidence

    emit({
        "event": "ready",
        "load_s": round(load_s, 2),
        "lid": lid is not None,
        "lid_error": lid_error,
        "elevenlabs": client is not None,
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2)
        if torch.cuda.is_available() else 0.0,
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

        try:
            wav = np.load(req["pcm_path"])
            rate = int(req.get("sample_rate") or LID_SAMPLE_RATE)
            duration_s = len(wav) / float(rate)
            t = time.time()

            if rate != LID_SAMPLE_RATE:
                # Both models are 16kHz-only. Say so once per utterance rather
                # than returning quietly wrong answers for the whole session.
                log(f"capture is {rate}Hz but the models expect "
                    f"{LID_SAMPLE_RATE}Hz; set capture.sample_rate: 16000")

            t_lid = time.time()
            route, lang, code, confidence = decide(wav, duration_s)
            lid_ms = int((time.time() - t_lid) * 1000)

            if route == ROUTE_INTERNATIONAL:
                if client is None:
                    # Transcribing it locally would produce Indic gibberish
                    # that reads as a working pipeline giving a strange answer.
                    # Naming the problem is the only honest option.
                    emit({"ok": False, "utt_id": req["utt_id"],
                          "error": "no_international_stt", "lang": lang,
                          "confidence": round(confidence, 3)})
                    continue

                wav_bytes = pcm16_to_wav(
                    (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes(),
                    rate,
                )
                result = client.transcribe(
                    wav_bytes,
                    language_code=code if confidence >= hint_confidence else None,
                )
                text = result["text"]
                # Scribe outranks LID: it heard the words, not just the accent.
                heard = language_from_code(result.get("language_code")) or lang

                # ...but not the transcript it just produced. Scribe misnames
                # the language of short or noisy clips — reporting Korean or
                # Chinese for English speech — and that name is not cosmetic:
                # it becomes "Reply ONLY in Korean" in the LLM's directive and
                # a Korean voice at synthesis. The script of the text is the
                # one piece of evidence that cannot disagree with itself, so
                # when the two conflict the text wins.
                written = detect_language(text) if text.strip() else None
                if written and written != heard:
                    # Only override when the scripts genuinely differ. Scribe
                    # saying "hindi" for Marathi is a distinction the script
                    # cannot make and should not be second-guessed on; Scribe
                    # saying "korean" over Latin text is.
                    if script_of(written) != script_of(heard):
                        log(f"scribe said {heard!r} but the transcript is "
                            f"{written!r}; trusting the script")
                        heard = written
                gate.observe(heard)
                emit({
                    "ok": True,
                    "utt_id": req["utt_id"],
                    "text": text,
                    "lang": heard,
                    "lang_code": result.get("language_code") or code,
                    "route": route_for(heard, stage="stt"),
                    "backend": "elevenlabs",
                    "confidence": round(
                        float(result.get("probability") or confidence), 3),
                    "lid_ms": lid_ms,
                    "elapsed_s": round(time.time() - t, 3),
                })
                continue

            with torch.no_grad():
                hyps = model.transcribe([wav], return_hypotheses=True)
            text = hyps[0].text.strip()
            gate.observe(lang)
            emit({
                "ok": True,
                "utt_id": req["utt_id"],
                "text": text,
                # None when LID abstained, so the LLM falls back to detecting
                # the language from the transcript as it always has.
                "lang": lang,
                "lang_code": code,
                "route": ROUTE_LOCAL,
                "backend": "sravaani",
                "confidence": round(confidence, 3),
                "lid_ms": lid_ms,
                "elapsed_s": round(time.time() - t, 3),
            })
        except ElevenLabsError as exc:
            log(f"international speech-to-text failed: {exc}")
            emit({"ok": False, "utt_id": req.get("utt_id"),
                  "error": f"international_stt: {exc}"})
        except Exception as exc:  # keep the worker alive across bad utterances
            log(f"transcribe failed: {exc!r}")
            emit({"ok": False, "utt_id": req.get("utt_id"), "error": str(exc)})


if __name__ == "__main__":
    main()
