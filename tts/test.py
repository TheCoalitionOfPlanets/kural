"""Manual smoke test for the Indic-Mio TTS worker's synthesis path.

Run with tts/venv's interpreter, from the repo root:

    tts/venv/bin/python tts/test.py

Loads the model and codec directly (not through the JSON-lines subprocess
protocol) and writes a handful of short utterances to tts/out/ so you can
listen to them.
"""
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "workers"))

import worker_tts  # noqa: E402

SAMPLES = [
    ("english", "Hello! This is a test of the new text to speech voice."),
    ("hindi", "नमस्ते, आप कैसे हैं?"),
    ("tamil", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"),
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from miocodec import MioCodec

    model_dir = ROOT / "tts" / "models" / "Indic-Mio"
    codec_dir = ROOT / "tts" / "models" / "MioCodec-25Hz-24kHz"

    print(f"loading {model_dir} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    codec = MioCodec.from_pretrained(str(codec_dir))
    print(f"loaded in {time.time() - t0:.2f}s")

    out_dir = ROOT / "tts" / "out"
    out_dir.mkdir(exist_ok=True)

    for lang, text in SAMPLES:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t = time.time()
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=1024, temperature=0.9, top_p=0.9,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        audio_codes = [
            tok.item() - worker_tts.SPEECH_OFFSET for tok in generated
            if worker_tts.SPEECH_OFFSET <= tok.item()
            < worker_tts.SPEECH_OFFSET + worker_tts.SPEECH_RANGE
        ]
        if not audio_codes:
            print(f"[{lang}] no audio tokens generated")
            continue

        codes_tensor = torch.tensor([audio_codes], dtype=torch.long).unsqueeze(0)
        wav = codec.decode(codes_tensor)
        audio = wav.squeeze().to(torch.float32).cpu().numpy()

        out_path = out_dir / f"{lang}.wav"
        pcm = (audio.clip(-1.0, 1.0) * 32767.0).astype("int16")
        with wave.open(str(out_path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(worker_tts.SAMPLE_RATE)
            fh.writeframes(pcm.tobytes())
        print(f"[{lang}] {len(audio) / worker_tts.SAMPLE_RATE:.2f}s audio -> "
              f"{out_path} ({time.time() - t:.2f}s)")


if __name__ == "__main__":
    main()
