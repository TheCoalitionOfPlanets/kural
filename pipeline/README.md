# Real-time speech pipeline

Always-listening voice loop:

```
mic → VAD → SraVaani STT → Gemma 3 4B → Indic-Mio TTS → speaker
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
| TTS (Indic-Mio) | `tts/venv/` | Python 3.12, transformers 5.15 (cu130 torch) |

So each model is hosted in a child process in its own venv, speaking JSON-lines
over stdin/stdout (`realtime/proc.py`). The orchestrator owns capture, queues,
scheduling, and playback.

## VRAM

All three models stay resident on one 12 GB card:

| Model | VRAM |
|---|---|
| SraVaani STT | 1.7 GB |
| Gemma 3 4B (4-bit NF4) | 3.0 GB |
| Indic-Mio TTS (bf16) | ~1.5 GB* |
| **Total** | **~6.2 GB** |

\* Indic-Mio is a 0.6B causal LM held resident on the GPU like STT and the
LLM, not a per-voice file loaded lazily on CPU — the exact figure is reported
in the `tts` worker's `ready` event (`vram_gb`) at startup.

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

The TTS row needs re-measuring after the Indic-Mio swap. Unlike Piper's
non-autoregressive VITS, Indic-Mio generates audio tokens autoregressively —
each reply pays a `model.generate()` cost proportional to the audio produced,
reported per-utterance as `elapsed_s` in the `tts` event. `llm.max_new_tokens`
and a brief system prompt remain the tuning knobs for reply length, and both
stay tuned for brevity in `config/realtime.yaml`; `tts.max_new_tokens` bounds
how much audio a single reply can generate.

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
   speakers and room reverb ring on. Ignored when `barge_in.enabled` — an
   interruptible mic cannot be a muted one; a stricter VAD gate
   (`capture.vad.barge_in_*`) takes over as the acoustic defence.
3. **Echo guard** — `echo.guard: true`. The only layer that *identifies* echo
   rather than reducing its odds: if a transcript's words overlap what the
   assistant just said, it is bleed, not a human, and it never reaches the
   model.

The genuinely airtight fix is physical: **headphones**, or a directional mic
pointed away from the speakers. Everything above is mitigation.

See [docs/echoReduction.md](../docs/echoReduction.md) for the full layer-by-layer
account.

## Barge-in

`barge_in.enabled: true` lets the user cut a reply short mid-playback. The
difficulty is that stopping has to happen in ~250 ms, while telling the user's
voice apart from the assistant's own needs the transcript, ~1–2 s later — so the
decision is split and the fast half is made reversible:

- **Tier 1** — sustained speech over a strict VAD gate stops playback
  immediately. Fast, and cannot tell you from the speakers.
- **Tier 2** — the echo guard rules on that utterance: real speech abandons the
  reply and flushes the queues; the assistant's own bleed replays it from the
  start.

A false interrupt therefore costs a restarted sentence, not a swallowed reply,
which is what makes the tier-1 gate safe to tune aggressively. Requires
`echo.guard`. Full design in [docs/bargeIn.md](../docs/bargeIn.md).

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
English. Detection covers English, Tamil, Hindi, Telugu, Kannada, Malayalam,
Bengali, Gujarati, Punjabi, Odia, Urdu. Set `llm.enforce_language: false` to
fall back to prompt-only behavior.

Unlike Piper, TTS is not a per-language voice file: Indic-Mio is a single
0.6B model (fine-tuned from
[Aratako/MioTTS-0.6B](https://huggingface.co/Aratako/MioTTS-0.6B)) that
infers the language directly from the script of the reply text and speaks
all 22 scheduled Indian languages plus English from one set of weights — no
`tts.voices` map, no per-language checkpoint to install. Generated audio
tokens are decoded to a waveform by
[Aratako/MioCodec-25Hz-24kHz](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz).

A language outside that set (e.g. Sinhala, which the detector recognizes but
Indic-Mio does not speak) still gets the same `no_voice` handling as before:
the worker returns `{"ok": false, "error": "no_voice"}` and the orchestrator
prints the reply as text instead, rather than reading it aloud in the wrong
language.

Run `bash download.sh --skip-venvs` (or a full `bash download.sh`) to fetch
the model and codec weights into `tts/models/`.

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
