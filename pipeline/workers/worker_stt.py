"""STT worker — SraVaani plus the language gate, runs in the root venv.

SraVaani is an Indic ASR: it hears the scheduled Indian languages and English
and nothing else. Hand it Spanish and it does not report failure — it emits
confident-looking Devanagari or Latin gibberish. So the transcript cannot be
used to decide whether the transcript should have been made here at all, and
the routing decision has to be made *before* transcription, from the waveform.

That is what the LID gate is. `facebook/mms-lid-126` classifies the language of
the audio directly, in one small forward pass, and its answer chooses the ear:

    indic / english  ->  SraVaani      (Set A, resident)
    anything else    ->  Whisper large-v3  (Set B, loaded on demand)

Both ears are local — there is no network on this path. Whisper is *not* held
resident, though: Set A and Set B are never both needed for the same turn, and
at 3.1 GB it does not belong on the card next to SraVaani, Gemma and the local
voice. It is loaded the first time a turn actually routes to it and kept from
then on, so the cost is paid once rather than per utterance.

LID only decides the *route*. On the international path Whisper's own detected
language is what the rest of the turn runs on, because it is the better
detector — it heard the words, not just the accent. That is also how a misroute
repairs itself: if LID says Spanish and Whisper hears Tamil, the turn is Tamil
from here on and TTS goes back to the local voice.

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
from realtime.languages import (  # noqa: E402
    ROUTE_INTERNATIONAL,
    ROUTE_LOCAL,
    RouteGate,
    detect_language,
    is_iso639_1,
    iso639_1_for,
    language_from_code,
    route_for,
    script_of,
)

# MMS-LID was trained on 16kHz speech, which is also SraVaani's rate and the
# capture rate in realtime.yaml. Anything else would need resampling, and a
# silently wrong rate makes LID confidently wrong rather than obviously broken.
LID_SAMPLE_RATE = 16000

# Whisper's feature extractor is also 16kHz-only, so the whole STT path shares
# one rate and nothing on it ever resamples.
WHISPER_SAMPLE_RATE = 16000


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
                      f"`hf download facebook/mms-lid-126 --local-dir "
                      f"{model_dir}` to fetch it")

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
    # Above this the LID answer is passed to Whisper as a decoding hint. Below
    # it Whisper is left to detect on its own, since a confident wrong hint is
    # worse than no hint at all — Whisper forced into the wrong language does
    # not fail, it translates.
    hint_confidence = float(lid_cfg.get("hint_confidence", 0.85))

    # -- Whisper large-v3, the Set B ear -----------------------------------
    #
    # Loaded on demand rather than at startup. It is 3.1 GB and Set A and Set B
    # are never both needed for the same turn, so holding it beside SraVaani,
    # Gemma and the local voice would cost a third of the card for a model most
    # sessions never reach. The first international turn pays the load; every
    # one after it is free.
    #
    # `enabled: false` (the suspended state) means never load it at all, and
    # international turns are reported rather than transcribed — same contract
    # the missing-key case had before.
    wh_cfg = cfg.get("whisper") or {}
    whisper_enabled = wh_cfg.get("enabled", True)
    whisper_dir = wh_cfg.get("model_dir")
    # The tuple is (processor, model, dtype) once loaded; the string is why it
    # could not be. Both start empty — nothing is decided until a turn asks.
    whisper = None
    whisper_error = None
    if not whisper_enabled:
        whisper_error = "disabled in config"
    elif not whisper_dir or not os.path.isdir(whisper_dir):
        whisper_error = (f"model_dir {whisper_dir!r} not found — fetch "
                         f"openai/whisper-large-v3 into it")

    def _load_whisper():
        """Load Whisper on first use. Returns (bundle, error); one is None.

        Failures are sticky: a missing directory or an OOM will not fix itself
        between utterances, and retrying the load on every foreign turn would
        pay several seconds each time to reach the same answer.
        """
        nonlocal whisper, whisper_error
        if whisper is not None or whisper_error is not None:
            return whisper, whisper_error

        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}.get(wh_cfg.get("dtype", "float16"),
                                               torch.float16)
        t = time.time()
        try:
            processor = AutoProcessor.from_pretrained(whisper_dir)
            wmodel = AutoModelForSpeechSeq2Seq.from_pretrained(
                whisper_dir, dtype=dtype, low_cpu_mem_usage=True,
            )
            wmodel = wmodel.to(cfg.get("device", "cuda"))
            wmodel.eval()
        except Exception as exc:
            whisper_error = repr(exc)
            log(f"Whisper failed to load ({whisper_error}); international "
                f"speech will be reported, not transcribed")
            return None, whisper_error
        whisper = (processor, wmodel, dtype)
        log(f"Whisper large-v3 loaded in {time.time() - t:.1f}s "
            f"({wh_cfg.get('dtype', 'float16')})")
        return whisper, None

    # Whisper's own language tags are ISO 639-1 ("es", "ta"), while LID speaks
    # 639-3 ("spa", "tam"). Rather than carry a second table, the hint is
    # resolved through the shared one in realtime/languages.py: code ->
    # language name -> the 639-1 tag Whisper wants. A name with no 639-1 tag
    # simply yields no hint, which is the safe direction.

    def transcribe_whisper(wav, hint_code=None):
        """One utterance -> {"text", "language"} using Whisper large-v3.

        Whisper detects the language as part of decoding and reports it, which
        is what the rest of the turn runs on. The LID hint is passed only when
        LID was confident: forcing the wrong language makes Whisper *translate*
        into it rather than refuse, which reads as a plausible but wrong
        transcript.
        """
        bundle, err = _load_whisper()
        if bundle is None:
            raise RuntimeError(err or "whisper unavailable")
        processor, wmodel, dtype = bundle

        features = processor(
            wav, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt",
        ).input_features.to(wmodel.device, dtype)

        kwargs = {"task": "transcribe"}
        hint_lang = language_from_code(hint_code) if hint_code else None
        iso1 = iso639_1_for(hint_lang) if hint_lang else None
        if iso1:
            kwargs["language"] = iso1

        with torch.inference_mode():
            ids = wmodel.generate(
                features,
                max_new_tokens=int(wh_cfg.get("max_new_tokens", 220)),
                num_beams=int(wh_cfg.get("num_beams", 1)),
                return_dict_in_generate=True,
                **kwargs,
            )
        sequences = ids.sequences
        text = processor.batch_decode(sequences, skip_special_tokens=True)[0]

        # The language token is the one Whisper itself chose, emitted near the
        # start of the sequence as `<|es|>`. Read it back rather than trusting
        # the hint, since the whole point of this path is that Whisper's answer
        # outranks LID's.
        detected = None
        for token in processor.tokenizer.convert_ids_to_tokens(sequences[0][:4]):
            if token.startswith("<|") and token.endswith("|>"):
                tag = token[2:-2]
                if is_iso639_1(tag):
                    detected = tag
                    break
        return {"text": text.strip(), "language_code": detected or iso1}

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
        # Whether the Set B ear *can* be loaded, not whether it is: the weights
        # are on disk and enabled, so an international turn will get one. It is
        # loaded on first use, so nothing is resident yet.
        "international_stt": whisper_error is None,
        "international_stt_error": whisper_error,
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
                if whisper_error is not None:
                    # Transcribing it locally would produce Indic gibberish
                    # that reads as a working pipeline giving a strange answer.
                    # Naming the problem is the only honest option.
                    emit({"ok": False, "utt_id": req["utt_id"],
                          "error": "no_international_stt", "lang": lang,
                          "reason": whisper_error,
                          "confidence": round(confidence, 3)})
                    continue

                result = transcribe_whisper(
                    wav,
                    hint_code=code if confidence >= hint_confidence else None,
                )
                text = result["text"]
                # Whisper outranks LID: it heard the words, not just the accent.
                heard = language_from_code(result.get("language_code")) or lang

                # ...but not the transcript it just produced. Whisper misnames
                # the language of short or noisy clips — reporting Welsh or
                # Chinese for English speech — and that name is not cosmetic:
                # it becomes "Reply ONLY in Welsh" in the LLM's directive and
                # a Welsh voice at synthesis. The script of the text is the
                # one piece of evidence that cannot disagree with itself, so
                # when the two conflict the text wins.
                written = detect_language(text) if text.strip() else None
                if written and written != heard:
                    # Only override when the scripts genuinely differ. Whisper
                    # saying "hindi" for Marathi is a distinction the script
                    # cannot make and should not be second-guessed on; Whisper
                    # saying "korean" over Latin text is.
                    if script_of(written) != script_of(heard):
                        log(f"whisper said {heard!r} but the transcript is "
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
                    "backend": "whisper",
                    # Whisper reports no per-utterance language probability, so
                    # LID's confidence is what there is. It described the route,
                    # not the transcript, which is the honest reading of it.
                    "confidence": round(float(confidence), 3),
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
        except Exception as exc:  # keep the worker alive across bad utterances
            log(f"transcribe failed: {exc!r}")
            emit({"ok": False, "utt_id": req.get("utt_id"), "error": str(exc)})


if __name__ == "__main__":
    main()
