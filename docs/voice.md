# The voice

The assistant speaks through two different TTS models — Indic-Mio for the
languages it knows, MMS-TTS for everything else. This is how the Indic-Mio half
is pinned to one speaker, and why the other half cannot be.

## Why a reference clip is mandatory

Indic-Mio is a **zero-shot** cloning model. It does not have a built-in voice.
Given no reference speaker it invents one per utterance, sampled at
`temperature: 0.9`, so consecutive replies come back in different voices — the
model is working exactly as designed, and the result still sounds broken.

`tts.reference_wav` is what fixes this. The clip is encoded once at startup into
a 128-dim speaker embedding and conditions every synthesis, so every local reply
is the same person. The worker reports which state it is in at startup:

    {"event": "ready", ..., "reference_voice": true}

`false` means no usable clip was found and the voice will drift between turns.

### Requirements

* **16-bit mono WAV.** Anything else is refused with a log line and ignored,
  because a silently wrong voice is harder to notice than a missing one.
* **At least ~3s**, ideally 30s+ of varied speech. Short clips clone, but less
  stably.
* **Neutral delivery.** Cloning is faithful, tone included: a whispered or
  emotion-tagged reference produces a whispered or emotional assistant. The
  bundled `samples/sample3.wav` (`<disgust>`) and `sample4.wav` (`<surprise>`)
  are the wrong shape for this reason.

The current reference is `tts/voice/reference.wav`, a copy of the model's own
`sample1.wav` — neutral Hindi/English code-mixed speech, 7.2s. It is copied
rather than referenced in place so that re-downloading the model cannot silently
change the assistant's voice.

To change the voice, replace that file and restart. Nothing else needs editing.

## The Set B voice is a different speaker, and stays one

MMS-TTS has **no cloning and no voice selection**. Each language ships one
checkpoint with one speaker baked into the weights, so `tts.reference_wav` has
no effect on it — there is no parameter to pass the clip to.

This is a real, audible limit rather than a configuration gap: switching from
Tamil to Spanish mid-conversation switches speaker, and nothing in the config
can prevent that. Worth stating plainly because it looks like a bug the first
time it happens.

Two things make it less jarring than it sounds:

* **It only happens on a language switch.** Within a language the voice is
  fixed, because the checkpoint is.
* **Loudness normalization applies to both**, so the two voices at least match
  in level — which is also what keeps a Set B reply from tripping the barge-in
  gate and cutting itself off.

The alternative would be a multi-speaker or cloning model for Set B, which
costs either an API dependency or a much larger checkpoint per language. The
trade taken here is 145 MB per language, entirely local, at the price of one
fixed voice per language.

## Two fixes this depended on

Both were latent bugs that made the local voice impossible, and are worth
knowing about if the symptoms ever return:

1. **The codec must be the one Indic-Mio was trained with.** Its model card
   names `MioCodec-25Hz-24kHz`. Audio tokens index *that* codec's codebook, so
   decoding them with `MioCodec-25Hz-44.1kHz` does not fail — it returns
   fluent, correctly-timed, completely wrong speech. Transcribed back, "Hello
   there, I'm doing well" came out as *"whisked the danny well smostery die she
   om"*. To the ear that is the assistant answering in another language, which
   is exactly how it was reported.

   The two checkpoints are different architectures, not two sizes of one:

   | | decoder | vocoder | output |
   |---|---|---|---|
   | `25Hz-24kHz` | wave + iSTFT head | none needed | waveform, 24kHz |
   | `25Hz-44.1kHz` | mel | bundled `vocoder.*` | mel -> vocoder, 44.1kHz |

   `MioCodec.from_pretrained` only understands the second and rejects the first
   for having "no vocoder weights" — which is what made the wrong codec look
   like the only working one. The wave-decoder checkpoint loads through
   `MioCodecModel` instead, whose `decode()` returns a waveform directly when
   the config sets `use_wave_decoder: true`. The worker now detects which shape
   a checkpoint is by looking for `vocoder.`-prefixed tensors, and reads the
   output rate from the codec's own config rather than assuming one.

2. **`codec.decode(codes_tensor)` passed content tokens into the speaker slot.**
   The real signature is
   `decode(global_embedding=None, content_token_indices=None, ...)`, so the
   positional call put audio codes where the speaker embedding belongs and left
   content empty, raising `Either content_token_indices or content_embedding
   must be provided` on every utterance. With a reference clip the correct call
   is `synthesize_from_tokens(content_token_indices, reference_waveform)`, which
   encodes the speaker and vocodes in one step.

## Which languages are local

Three models, three different language sets — routing only works where they
overlap, so the sets are kept separately rather than assumed identical.

| | covers | notes |
|---|---|---|
| **Indic-Mio** (voice) | English + all 22 scheduled Indian languages | `LOCAL_TTS`, 23 total |
| **SraVaani** (ear) | English + 65 Indian languages and dialects | **minus Urdu and Kashmiri** — its model card is explicit |
| **MMS-LID-126** (router) | 126 languages | can only *name* 16 of the local set |

`route_for(lang, stage=...)` answers per stage, because the ear and the voice
do not agree:

* `stage="stt"` → `LOCAL_STT` (21): which model transcribes the audio.
* `stage="tts"` → `LOCAL_TTS` (23): which model speaks the reply.
* no stage → `LOCAL` (21), the intersection: fully local, both halves.

### Urdu and Kashmiri

These are the reason the split exists. SraVaani cannot hear them and Indic-Mio
speaks them, so they are **heard by Whisper and spoken locally** — the only
combination where both halves work. A single shared set would have broken one
half whichever way it was set:

* all-local → Urdu transcribed by a model that has never seen it
* all-international → Urdu spoken by a generic MMS checkpoint when a model
  trained on it was available

### The seven LID cannot name

`bodo`, `dogri`, `kashmiri`, `konkani`, `maithili`, `manipuri`, `santali` have
no MMS-LID label, so LID can never identify them. That is harmless here: the
gate abstains and `route_for` defaults to local, which is where they belong.
Worth knowing only because it means those languages are never *deliberately*
routed — they arrive local by falling through.

## Why English was being sent to the Set B stack

Two independent bugs, both visible in one log line:

    [u0002] heard (english via Whisper): Hi, how are you?

English is local, so that route should never have been taken.

### 1. The top-1 label is biased toward leaving

Only **15 of MMS-LID's 126 labels** route local; the other 111 route
international. So the argmax is structurally biased: an English utterance that
splits its probability across `eng`, `cym` (Welsh) and `nno` (Norwegian
Nynorsk) can top out on a foreign label and be sent abroad while local was the
right answer by a wide margin. Measured on silence, LID predicts `cym` at 0.478
and `nno` at 0.484 — confident-looking noise.

The gate now sums probability **per route** (`lid.min_local_mass`, default
0.30) and keeps the turn local when the local total clears the bar, whatever
the argmax said. Measured local mass: English 0.997–0.999, code-mixed
Hindi/English 0.822, Tamil 1.000, noise 0.015. All 111 foreign labels still
route international.

### 2. The Set B ear's language became the reply language, unchecked

`heard = language_from_code(result["language_code"])` was taken on faith. The
international ear misnames short or noisy clips — reporting Korean or Chinese
for English — and that name is not cosmetic: it becomes `Reply ONLY in Korean`
in the LLM's directive and a Korean voice at synthesis. That is the
wrong-language output.

The transcript is now cross-checked against its own script: when the reported
language and the text's script belong to **different writing systems**, the text
wins. Languages that share a script (Hindi/Marathi in Devanagari) are left
alone, since script cannot separate them and the disagreement is not evidence of
an error.

This survived the move from Scribe to Whisper unchanged — Whisper has the same
failure mode on short clips, and the same guard catches it.

| transcript | the ear said | result |
|---|---|---|
| `Hi, how are you?` | korean | **english** — corrected |
| `Hi, how are you?` | chinese | **english** — corrected |
| `Hola, buenos dias.` | spanish | spanish — untouched |
| Devanagari text | marathi | marathi — untouched (shared script) |
| Korean text | korean | korean — untouched |

## VRAM

Measured on this machine, loading each model in turn:

| model | role | VRAM |
|---|---|---|
| SraVaani | ASR (fp16 TorchScript) | 1.69 GB |
| MMS-LID-126 | language routing, float32 | 3.69 GB |
| MMS-LID-126 | language routing, **float16** | **1.86 GB** |

The router costing more than the ASR it gates for is worth noticing: MMS-LID is
966M parameters and its entire output is one argmax plus a probability sum, so
nothing on that path needs float32. `lid.dtype: float16` halves it with
identical predictions — verified to three decimals on clean English, Tamil,
Gujarati, a 1.8s fragment, a borderline Hindi/English code-mix (0.488 vs 0.492,
same route), and noise.

STT worker total: **5.39 GB -> 3.5 GB**.

LID adds ~90 ms per utterance in steady state. The first call is slower (~820
ms) — that is warmup, not per-turn cost.
