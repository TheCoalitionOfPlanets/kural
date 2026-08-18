"""
Gemma 3 4B IT test — runs fully on CUDA via transformers.

At bf16 the model is ~8.6GB, which fits natively in 12GB VRAM, so no
quantization is needed. Set USE_4BIT = True to load it in ~3GB instead
(frees VRAM for longer contexts, slight quality cost).
"""
import os
import sys
import time

import torch

# Gemma emits em-dashes, arrows, etc. — the Windows console defaults to cp1252
# and would raise UnicodeEncodeError on them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "gemma-3-4b-it"
)

USE_4BIT = False

print("=" * 60)
print("GEMMA-3-4B-IT TEST")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this machine.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")

print(f"\nLoading model ({'4-bit' if USE_4BIT else 'bf16'})...")
t0 = time.time()

load_kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
if USE_4BIT:
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Gemma3ForConditionalGeneration.from_pretrained(MODEL_PATH, **load_kwargs)
model.eval()

print(f"Model loaded in {time.time() - t0:.1f}s")
print("VRAM allocated:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")


def ask(question, max_new_tokens=512, temperature=1.0):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": question}]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]

    print("\n" + "-" * 60)
    print("Q:", question)

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
    elapsed = time.time() - t

    # Strip the prompt, keeping only newly generated tokens.
    new_tokens = generated[0][prompt_len:]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n" + answer)

    n = len(new_tokens)
    print(f"\n({n} tokens in {elapsed:.1f}s = {n / elapsed:.1f} tok/s)")


ask("A farmer has 17 sheep. All but 9 run away. How many sheep does he have left? Explain your reasoning.")
ask("Write a single haiku about GPUs.", max_new_tokens=64)

print("\n" + "=" * 60)
print("Done!")
print("Peak VRAM:", round(torch.cuda.max_memory_allocated() / 1024**3, 2), "GB")
