"""STT worker — SraVaani, runs in the root venv.

Protocol (JSON lines on stdin/stdout):
    <- {"cmd": "init", "config": {...}}
    -> {"event": "ready", ...}
    <- {"cmd": "run", "utt_id": ..., "pcm_path": "...npy"}
    -> {"ok": true, "utt_id": ..., "text": ...}
"""
import json
import sys
import time

import numpy as np


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
    load_s = time.time() - t0

    emit({
        "event": "ready",
        "load_s": round(load_s, 2),
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
            t = time.time()
            with torch.no_grad():
                hyps = model.transcribe([wav], return_hypotheses=True)
            text = hyps[0].text.strip()
            emit({
                "ok": True,
                "utt_id": req["utt_id"],
                "text": text,
                "elapsed_s": round(time.time() - t, 3),
            })
        except Exception as exc:  # keep the worker alive across bad utterances
            log(f"transcribe failed: {exc!r}")
            emit({"ok": False, "utt_id": req.get("utt_id"), "error": str(exc)})


if __name__ == "__main__":
    main()
