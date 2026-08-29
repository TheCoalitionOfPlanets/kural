"""Reasoning worker — Gemma 3 4B IT, runs in reasoning/venv.

Loaded in 4-bit so SraVaani also fits on the same 12GB card (Piper TTS is
CPU-only and uses no VRAM).

Protocol (JSON lines on stdin/stdout):
    <- {"cmd": "init", "config": {...}}
    -> {"event": "ready", ...}
    <- {"cmd": "run", "utt_id": ..., "text": ..., "lang": ...}
    -> {"ok": true, "utt_id": ..., "text": ..., "lang": ...}
"""
import json
import os
import sys
import time

# This worker runs in its own venv and cannot import the orchestrator package,
# so the speakable-text backstop is loaded by path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime.languages import ENGLISH, detect_language, language_directive  # noqa: E402
from realtime.speakable import strip_unspeakable  # noqa: E402


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


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
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Gemma3ForConditionalGeneration,
    )

    if cfg.get("require_cuda", True) and not torch.cuda.is_available():
        log("CUDA required but unavailable")
        emit({"event": "error", "error": "cuda_unavailable"})
        return

    load_kwargs = {"dtype": torch.bfloat16, "device_map": cfg.get("device", "cuda:0")}
    if cfg.get("load_in_4bit", True):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(cfg["model_dir"])
    model = Gemma3ForConditionalGeneration.from_pretrained(cfg["model_dir"], **load_kwargs)
    model.eval()
    load_s = time.time() - t0

    # The prompt is a config file, not code — edit it and restart, no rebuild.
    # An inline system_prompt still works as a fallback.
    system_prompt = (cfg.get("system_prompt") or "").strip()
    prompt_file = cfg.get("prompt_file")
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as fh:
                system_prompt = fh.read().strip()
        except OSError as exc:
            log(f"could not read prompt_file {prompt_file!r}: {exc}; "
                f"falling back to inline system_prompt")
    max_new_tokens = int(cfg.get("max_new_tokens", 160))
    temperature = float(cfg.get("temperature", 1.0))
    enforce_language = bool(cfg.get("enforce_language", True))
    history_turns = int(cfg.get("history_turns", 4))
    history_ttl_s = float(cfg.get("history_ttl_s", 0) or 0)
    history = []  # [(user_text, reply_text), ...]
    last_exchange = 0.0
    last_lang = None

    emit({
        "event": "ready",
        "load_s": round(load_s, 2),
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
        if cmd == "reset":
            history.clear()
            emit({"ok": True, "reset": True})
            continue
        if cmd != "run":
            continue

        try:
            # A stale thread is worse than no thread: the model answers as if
            # a conversation from an hour ago were still in progress.
            if history_ttl_s and history and (
                time.time() - last_exchange > history_ttl_s
            ):
                history.clear()

            # The language is decided in code, not left to the model. Stating
            # it explicitly is far more reliable than a 4B model re-inferring
            # it from a short, possibly garbled transcript — it was observed
            # naming the language correctly and still replying in English.
            #
            # Upstream wins when it has an answer. The STT stage identifies the
            # language from the *audio*, and on the international path gets it
            # from Whisper, which actually heard the words. Re-deriving it from
            # the transcript here would be strictly worse and, for a
            # Latin-script language, plainly wrong: detect_language() has no
            # markers for Spanish or German, so it calls them English and the
            # reply comes back in the wrong language with a matching voice.
            lang = None
            if enforce_language:
                lang = req.get("lang") or detect_language(req["text"])

            messages = []
            if system_prompt:
                system = system_prompt
                if lang:
                    system = f"{system_prompt}\n\n{language_directive(lang)}"
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system}],
                })
            # History in the previous language is the strongest pull back
            # toward it — several turns of English right before a Tamil
            # question outweigh any instruction. Drop it on a switch.
            if lang and last_lang and lang != last_lang:
                history.clear()

            for prev_user, prev_reply in history[-history_turns:]:
                messages.append({"role": "user",
                                 "content": [{"type": "text", "text": prev_user}]})
                messages.append({"role": "assistant",
                                 "content": [{"type": "text", "text": prev_reply}]})

            # Restated immediately before the user turn: position beats
            # emphasis, and the system slot is far from where generation
            # begins once history sits in between.
            user_text = req["text"]
            if lang and lang != ENGLISH:
                user_text = f"{req['text']}\n\n[{language_directive(lang)}]"
            messages.append({"role": "user",
                             "content": [{"type": "text", "text": user_text}]})

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]

            t = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=64,
                )
            new_tokens = generated[0][prompt_len:]
            raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
            # The prompt asks for speech-shaped output; this enforces it.
            reply = strip_unspeakable(raw)

            if lang:
                last_lang = lang
            if history_turns and reply:
                history.append((req["text"], reply))
                del history[:-history_turns]
                last_exchange = time.time()

            emit({
                "ok": True,
                "utt_id": req["utt_id"],
                "text": reply,
                "lang": lang,
                "n_tokens": int(len(new_tokens)),
                "elapsed_s": round(time.time() - t, 3),
            })
        except Exception as exc:
            log(f"generate failed: {exc!r}")
            emit({"ok": False, "utt_id": req.get("utt_id"), "error": str(exc)})


if __name__ == "__main__":
    main()
