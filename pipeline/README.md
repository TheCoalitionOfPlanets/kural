# Real-time speech pipeline

Always-listening voice loop:

```
                        ┌ SraVaani STT ┐                ┌ Indic-Mio TTS ┐
mic → VAD → language ID ┤              ├→ Gemma 3 4B →  ┤               ├→ speaker
                        └ Whisper lg-v3┘                └ MMS-TTS       ┘
```

The Set A models are Indic: they hear and speak the 22 scheduled Indian
languages plus English. Anything else — Spanish, Russian, Japanese — is routed
to Set B (Whisper large-v3 and MMS-TTS), per utterance, decided from the
waveform before transcription. Every model is local; the Set B pair is loaded
on demand rather than held resident.
See [Reply language](#reply-language) and
[docs/internationalLanguages.md](../docs/internationalLanguages.md).

## Running

Build the environments and fetch the weights first — `bash setup.sh` from the
repo root, or `setup.bat` on Windows. See
[Running it](../README.md#running-it).

Two front ends over the same pipeline.

**Terminal** — the machine's own microphone and speakers:

```bash
venv/bin/python pipeline/run_realtime.py            # Linux, macOS
venv\Scripts\python.exe pipeline\run_realtime.py     # Windows
```

**Browser** — a ChatGPT-voice-style interface, where the tab is the microphone
and the speakers:

```bash
venv/bin/python -m pipeline.server
cd web && npm install && npm run dev
```

They share everything but the two ends: `realtime/session.py` builds one graph,
and only the frame source and the player differ (`MicSource`/`Player` vs
`StreamSource`/`WebPlayer`). The VAD, barge-in, echo guard and language gate are
the same code either way. See [web/README.md](../web/README.md) and
[docs/webInterface.md](../docs/webInterface.md).

Speak, pause, and the reply is spoken back. `Ctrl+C` shuts down cleanly and
releases the mic.

Verify capture and VAD alone first — it starts instantly and writes each
detected utterance to `spill/` so you can listen and confirm segmentation:

```bash
venv/bin/python pipeline/run_realtime.py --capture-only
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
| MMS-LID 126 (fp32) | ~1.2 GB† |
| Gemma 3 4B (4-bit NF4) | 3.0 GB |
| Indic-Mio TTS (bf16) | ~1.5 GB* |
| **Total** | **~7.4 GB** |

\* Indic-Mio is a 0.6B causal LM held resident on the GPU like STT and the
LLM, not a per-voice file loaded lazily on CPU — the exact figure is reported
in the `tts` worker's `ready` event (`vram_gb`) at startup.

† The language gate that routes international turns to Set B. Optional:
set `stt.lid.enabled: false` (or skip the download) to reclaim it and route
every turn locally. `stt.lid.dtype: float16` halves it if VRAM is tight.

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

A fourth layer catches a different loop entirely:

4. **Input echo guard** — `echo.input.guard: true`. Everything above watches
   for the assistant's *replies* coming back. This catches the same *input*
   arriving twice — a user turn the pipeline already accepted being fed in
   again — which no reply window can see, because the text never was a reply.
   Compared by symmetric similarity over both texts rather than the containment
   fraction the reply guard uses: a re-fed input is the whole utterance again,
   differing only by STT jitter, where reply bleed is a short fragment of
   something long. `echo.input.ttl_s` is deliberately shorter than the reply
   window's, because a repeat is only echo while the original turn is still in
   flight — past that, someone saying "yes please" twice is a person saying it
   twice.

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

1. The language is identified **from the audio**, before transcription, by
   `facebook/mms-lid-126` in the STT worker — *currently suspended
   (`stt.lid.enabled: false`), so this step is skipped and every turn falls
   straight through to the transcript path below.* Where it abstains — a very
   short utterance, a low-confidence prediction — [languages.py](realtime/languages.py)
   falls back to the transcript: native script is unambiguous (a Tamil
   character proves Tamil), and romanized input ("enna panra", "kya kar rahe
   ho") is scored against per-language function words, the grammatical
   machinery a speaker cannot avoid.
2. That language travels with the turn — `Sentence.lang` → `Reply.lang` — and
   is never re-derived downstream. It cannot be: a Spanish transcript is Latin
   script with no markers, and the text-based detector reads it as English.
3. A directive naming that language is appended to the system prompt *and*
   restated just before the user turn. Position beats emphasis: a rule next to
   where generation starts is followed far more reliably than one in the middle
   of a long prompt.
4. Switching language clears history, since several turns in the old language
   outweigh any instruction.

A stray borrowed word does not count as a switch — "send it to my machan" stays
English. Set `llm.enforce_language: false` to fall back to prompt-only
behavior.

## International languages

**The same rule, one stack further out.** Speak Spanish and you are answered in
Spanish, in a Spanish voice.

The local models cannot do this and never will: SraVaani hears the scheduled
Indian languages plus English, Indic-Mio speaks the same set. So that set is
"local" and its complement is "international", and international turns are
served by Set B — Whisper large-v3 for the ear, MMS-TTS for the voice.
`route_for()` in [languages.py](realtime/languages.py) is the single place that
decision is made.

> **Currently suspended.** `stt.lid.enabled`, `stt.whisper.enabled` and
> `tts.mms_tts.enabled` are all `false` in
> [realtime.yaml](config/realtime.yaml), so none of this path is live: every
> turn is heard by SraVaani and spoken by Indic-Mio. A reply in a language the
> local voice cannot speak is printed as text rather than spoken. The section
> below describes the design, which is unchanged and returns by setting the
> three flags back to `true`.

The hard part is *knowing*. Handed Spanish, SraVaani does not fail — it returns
confident Devanagari gibberish — so the transcript cannot reveal that it should
never have been made. The decision therefore comes from the waveform, before
transcription:

```
        ┌──────────────┐   indic / english   ┌──────────────┐
audio ─▶│  MMS-LID 126 │────────────────────▶│  SraVaani    │─▶ transcript
        │  (one pass)  │                     └──────────────┘
        └───────┬──────┘   anything else     ┌──────────────┐
                └────────────────────────────▶│  Whisper l-v3│─▶ transcript
                                             └──────────────┘
```

Everything after that point is unchanged. The reply comes back as a WAV either
way, loudness-normalized by the same code, played by the same player, and
guarded by the same VAD gate, echo guard and barge-in — an international reply
is interruptible exactly like a local one.

Four details do the real work:

- **Failing toward local.** Below `stt.lid.min_confidence` the gate declines to
  commit and the turn stays local. A wrong local route costs one bad
  transcript; a wrong international route costs loading a 1.55B model and a
  slower turn, and on a mostly-Indic pipeline the ambiguous turns are mostly
  Indic.
- **Hysteresis that only confirms.** A conversation in Spanish survives one
  mumbled sentence (`stt.lid.sticky_ttl_s`), but a *confident* Tamil prediction
  wins immediately — switching back to an Indian language lands on the very
  next sentence, never after a delay.
- **Whisper outranks LID.** LID picks the route; Whisper heard the words, so
  its language is what the turn runs on. A misroute repairs itself: LID says
  Spanish, Whisper hears Tamil, and TTS goes straight back to the local voice.
- **The voice is chosen by the reply, not the route.** TTS calls `route_for()`
  on the reply's language, so the two halves cannot disagree.

Both halves fail soft. Without the LID weights every turn routes locally, as it
did before this existed; without the Whisper weights international turns are
reported rather than transcribed into gibberish. Either way Indic and English
are untouched, and both are stated at startup rather than discovered mid-
sentence.

Whisper hears more languages than there are MMS-TTS checkpoints configured for
in `MMS_TTS_VOICES`. Those get the existing text-only treatment: the reply is
printed, not mispronounced. Adding one is a line in that table plus its
checkpoint on disk. Full design in
[docs/internationalLanguages.md](../docs/internationalLanguages.md).

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

Fetch the model and codec weights into `tts/models/` with the Hugging Face
CLI:

```bash
hf download Aratako/Indic-Mio            --local-dir tts/models/Indic-Mio
hf download Aratako/MioCodec-25Hz-24kHz  --local-dir tts/models/MioCodec-25Hz-24kHz
```

The Set B voices are per-language and only the ones in use are needed:

```bash
hf download facebook/mms-tts-spa --local-dir tts/models/mms-tts/spa
```

## Tuning

Everything lives in [config/realtime.yaml](config/realtime.yaml).

**Assistant replies to itself** — confirm `echo.guard: true`; lower
`echo.threshold` toward `0.5`; lower `tts.normalize.target_lufs`; raise
`echo.mute_tail_ms`.

**Real speech gets swallowed as echo** — raise `echo.threshold` toward `0.75`.

**The same input is answered twice** — confirm `echo.input.guard: true`; lower
`echo.input.threshold` toward `0.8`.

**A deliberate repeat gets swallowed** — the user said the same thing twice on
purpose and the second was dropped. Raise `echo.input.threshold` toward `0.95`,
or lower `echo.input.ttl_s` so a repeat counts as echo for a shorter window.

**False triggers on room noise** — raise `capture.vad.noise_margin` (energy
gate multiplier over the calibrated floor) or `aggressiveness` (0–3). The
webrtc backend requires *both* a voiced classification and energy above the
gate, which is what keeps fan noise out.

**Clipped first syllable** — raise `pre_roll_ms`. Audio from before speech
onset is prepended so the leading phoneme survives.

**Sentences chopped mid-thought** — raise `silence_ms`.

**Replies too long/slow** — lower `llm.max_new_tokens`.

**Spanish (or Russian, or Japanese) comes out as Indic gibberish** — the LID
gate did not fire. Check the startup line: if it says language ID is off, the
weights are missing (`hf download facebook/mms-lid-126 --local-dir
stt/models/mms-lid-126`). Otherwise lower
`stt.lid.min_confidence` toward `0.4`, or raise `stt.lid.min_audio_s` if the
turns are short.

**Indic turns are being routed to Set B** — raise `stt.lid.min_confidence`
toward `0.7`. Every international route loads a 1.55B model and slows the turn,
and the `heard (… via Whisper)` tag on the transcript line is how you spot
them.

**International replies cut themselves off** — the same fix as any other
self-interrupt, since both voices share the normalizer: lower
`tts.normalize.target_lufs`, or raise `capture.vad.barge_in_energy_multiplier`.

## Files

```
pipeline/
├── run_realtime.py        entrypoint: config → queues → workers → shutdown
├── config/realtime.yaml   all tunables
├── config/prompts/        system prompt (edit and restart)
├── realtime/
│   ├── messages.py        Utterance, Sentence, Reply, WavJob
│   ├── capture.py         VAD endpointing + pluggable frame sources
│   ├── proc.py            subprocess model host (JSON-lines protocol)
│   ├── workers.py         stage threads: STT → LLM → TTS → playback
│   ├── echo_guard.py      text-level self-hearing and re-fed-input detection
│   ├── languages.py       language detection, stack routing, reply directive
│   ├── speakable.py       strips what a TTS voice cannot say
│   └── audio_out.py       blocking, strictly serial playback
├── realtime/
│   ├── session.py         the one wiring both front ends share
│   └── web_player.py      playback in a browser, behind Player's interface
├── workers/
│   ├── worker_stt.py      runs in venv/ — SraVaani + the LID gate + Whisper
│   ├── worker_llm.py      runs in reasoning/venv/
│   └── worker_tts.py      runs in tts/venv/ — Indic-Mio + the MMS-TTS voice
├── server/app.py          WebSocket front end for web/
├── tests/                 run each file directly with venv's python
│   └── stub_server.py     the server with the models faked — no GPU needed
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
