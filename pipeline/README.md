# Real-time speech pipeline

Always-listening voice loop:

```
mic → VAD → SraVaani STT → Gemma 3 4B → Piper TTS → speaker
```

## Running

```bat
venv\Scripts\python.exe pipeline\run_realtime.py
```

Speak, pause, and the reply is spoken back. `Ctrl+C` shuts down cleanly and
releases the mic.

Verify capture and VAD alone first — it starts instantly and writes each
detected utterance to `spill/` so you can listen and confirm segmentation:

```bat
venv\Scripts\python.exe pipeline\run_realtime.py --capture-only
```

## Why subprocesses

The three model stacks pin incompatible dependencies and cannot share one
interpreter:

| Stage | venv | Key pin |
|---|---|---|
| STT (SraVaani) | `venv/` | Python 3.14, transformers 5.15 |
| Reasoning (Gemma 3 4B) | `reasoning/venv/` | Python 3.12, transformers 5.15 |
| TTS (Piper) | `tts/venv/` | Python 3.12, piper-tts (onnxruntime, CPU) |

So each model is hosted in a child process in its own venv, speaking JSON-lines
over stdin/stdout (`realtime/proc.py`). The orchestrator owns capture, queues,
scheduling, and playback.

## VRAM

All three models stay resident on one 12 GB card:

| Model | VRAM |
|---|---|
| SraVaani STT | 1.7 GB |
| Gemma 3 4B (4-bit NF4) | 3.0 GB |
| Piper TTS | 0 GB* |
| **Total** | **~4.7 GB** |

\* Piper runs on CPU through onnxruntime, so it uses no VRAM at all. Each
voice is a ~60 MB ONNX file held in host RAM, loaded on first use and cached.

Gemma is quantized specifically so all three fit; at bf16 it alone takes 8 GB.
Set `llm.load_in_4bit: false` in the config to trade VRAM for quality.

## Measured latency

For a short utterance, warm:

| Stage | Time |
|---|---|
| STT | ~0.7 s |
| Reasoning | ~1.4 s |
| TTS | not yet re-measured |
| **Total** | — |

The TTS row needs re-measuring after the Piper swap. Piper is a small VITS
model synthesizing roughly an order of magnitude faster than real time on CPU,
so TTS is no longer the dominant cost and reply length is once again the main
lever. `llm.max_new_tokens` and a brief system prompt are therefore the tuning
knobs, and both remain tuned for brevity in `config/realtime.yaml`. A voice's
first use also pays a one-off ~0.3 s load, after which it stays cached.

## Echo reduction

With speakers open the mic hears the assistant's own reply, STT transcribes it,
and the model answers itself. Three layers guard against that:

1. **Quieter output** — `tts.normalize.target_lufs: -23.0`. The loop is
   acoustic, so amplitude is the most effective software lever. If replies are
   too quiet, raise the *system* volume: a quiet signal amplified downstream
   keeps a better speaker-to-mic ratio than a hot one.
2. **Mic muted while replying** — `echo.mute_capture_while_replying: true`.
   Capture drops frames entirely while audio is playing, so bleed is never
   recorded. Airtight, at the cost of not being able to interrupt mid-reply.
   `echo.mute_tail_ms` keeps the mute a moment past the last sample, since
   speakers and room reverb ring on.
3. **Echo guard** — `echo.guard: true`. The only layer that *identifies* echo
   rather than reducing its odds: if a transcript's words overlap what the
   assistant just said, it is bleed, not a human, and it never reaches the
   model.

The genuinely airtight fix is physical: **headphones**, or a directional mic
pointed away from the speakers. Everything above is mitigation.

## The system prompt

[config/prompts/voice_assistant.md](config/prompts/voice_assistant.md) is a
config file, not code — edit it and restart, no rebuild.

The rules that matter acoustically are not trusted to the prompt. A 4B
quantized model ignores instructions under load, so each has a backstop in
[realtime/speakable.py](realtime/speakable.py):

| Prompt rule | Backstop |
|---|---|
| "No emoji" | stripped from every reply |
| "No markdown" | headings, bullets, emphasis, code fences stripped |
| "Two to four sentences" | `llm.max_new_tokens: 96` hard cut |
| (no reasoning spans read aloud) | `<think>` blocks stripped unconditionally |

The prompt asks; the decode budget and the sanitizer decide.

## Reply language

**You are answered in the language you spoke.** Speak English, get English;
speak Tamil, get Tamil in Tamil script.

The prompt asks for this, but a 4B model does not reliably obey — it was
observed identifying "epdi irukka" as Tamil and then answering in English. So
the language is decided in code:

1. [languages.py](realtime/languages.py) detects the language from the
   transcript. Native script is unambiguous (a Tamil character proves Tamil).
   Romanized input ("enna panra", "kya kar rahe ho") is scored against
   per-language function words — the grammatical machinery a speaker cannot
   avoid.
2. A directive naming that language is appended to the system prompt *and*
   restated just before the user turn. Position beats emphasis: a rule next to
   where generation starts is followed far more reliably than one in the middle
   of a long prompt.
3. Switching language clears history, since several turns in the old language
   outweigh any instruction.

A stray borrowed word does not count as a switch — "send it to my machan" stays
English. Detection covers English, Spanish, Tamil, Hindi, Telugu, Kannada,
Malayalam, Bengali, Gujarati, Punjabi, Odia, Urdu. Set `llm.enforce_language:
false` to fall back to prompt-only behavior.

The TTS voice switches with the language: a Piper voice *is* a language, so
the detected reply language selects the checkpoint (`tts.voices` in the
config).

Voices exist for English, Tamil, Hindi, Malayalam, Telugu, Urdu, Bengali,
Marathi and Nepali, plus a core international set. Tamil is not in the official
`rhasspy/piper-voices` repo — it comes from the community
[Jeyaram-K/piper-tamil-voices](https://huggingface.co/Jeyaram-K/piper-tamil-voices)
(Apache-2.0), in standard Piper format. Two are installed: `ta_valluvar`
(default, trained far longer) and `ta_hemalatha`.

Kannada has no Piper voice in any repo. It is built instead by converting the
[SYSPIN](https://huggingface.co/SYSPIN/vits_Kannada_Female) (IISc Bangalore,
MIT) Coqui VITS checkpoint with `tts/convert_coqui_to_piper.py` — Piper is VITS
too, and that model is character-based, so the generator weights transfer
directly and the Coqui charset becomes the phoneme_id_map. `download_voices.sh`
runs the conversion for you.

**Gujarati, Punjabi, Odia and Assamese still have no usable voice.** Replies in
those languages are not spoken — the worker returns `no_voice` and the
orchestrator prints the reply as text instead. That is deliberate: reading them
aloud with a wrong-language voice mispronounces every word, which is worse than
staying silent.

Run `bash tts/download_voices.sh` to fetch the voices.

## Tuning

Everything lives in [config/realtime.yaml](config/realtime.yaml).

**Assistant replies to itself** — confirm `echo.guard: true`; lower
`echo.threshold` toward `0.5`; lower `tts.normalize.target_lufs`; raise
`echo.mute_tail_ms`.

**Real speech gets swallowed as echo** — raise `echo.threshold` toward `0.75`.

**False triggers on room noise** — raise `capture.vad.noise_margin` (energy
gate multiplier over the calibrated floor) or `aggressiveness` (0–3). The
webrtc backend requires *both* a voiced classification and energy above the
gate, which is what keeps fan noise out.

**Clipped first syllable** — raise `pre_roll_ms`. Audio from before speech
onset is prepended so the leading phoneme survives.

**Sentences chopped mid-thought** — raise `silence_ms`.

**Replies too long/slow** — lower `llm.max_new_tokens`.

## Files

```
pipeline/
├── run_realtime.py        entrypoint: config → queues → workers → shutdown
├── config/realtime.yaml   all tunables
├── config/prompts/        system prompt (edit and restart)
├── realtime/
│   ├── messages.py        Utterance, Sentence, Reply, WavJob
│   ├── capture.py         mic reader + VAD endpointing state machine
│   ├── proc.py            subprocess model host (JSON-lines protocol)
│   ├── workers.py         stage threads: STT → LLM → TTS → playback
│   ├── echo_guard.py      text-level self-hearing detection
│   ├── languages.py       reply-language detection and per-turn directive
│   ├── speakable.py       strips what a TTS voice cannot say
│   └── audio_out.py       blocking, strictly serial playback
├── workers/
│   ├── worker_stt.py      runs in venv/
│   ├── worker_llm.py      runs in reasoning/venv/
│   └── worker_tts.py      runs in tts/venv/
├── tests/                 run each file directly with venv's python
└── spill/                 scratch audio, deleted after playback
```

## Backpressure

Queues are bounded. `audio_queue` **drops its oldest utterance** when full —
blocking there would stall the mic callback and corrupt the stream. Every drop
is logged; a steady stream of them means the GPU cannot keep up. The other
queues block, which throttles the stage upstream naturally.

## Notes

- `HF_HOME` pointing at an unmounted drive is detected and replaced with an
  in-repo cache, so a disconnected drive cannot break a run.
- Playback is strictly serial — one utterance at a time, never overlapping.
- Capture keeps running during playback, so you can speak over a reply.
