# Kural

A real-time speech-to-speech assistant for Indian languages.

You speak, it listens, it thinks, it answers out loud — and it answers in the
language you spoke. Tamil in, Tamil out. English in, English out. No buttons, no
language picker, no mode to switch.

The pipeline runs as four model stages connected by bounded queues, each stage in
its own process:

```
   microphone
       |
       v
   [ VAD ]  voice activity detection — where an utterance starts and stops
       |
       v
   [ facebook/mms-lid-126 ]  which speech stack should handle this turn?
       |
       +------------------+------------------+
       |                                     |
       v                                     v
   [ STT-A ]  SraVaani                   [ STT-B ]  Whisper large-v3
   23 Indian languages + English         26 world languages
       |                                     |
       +------------------+------------------+
                          |
                          v
                    [ LLM ]  Gemma 3 4B — reasoning, reply in the same language
                          |
       +------------------+------------------+
       |                                     |
       v                                     v
   [ TTS-A ]  Indic-Mio                  [ TTS-B ]  MMS-TTS (VITS)
   23 Indian languages + English         26 world languages
       |                                     |
       +------------------+------------------+
                          |
                          v
                      speaker
```

Five models total: **two for speech recognition**, **two for speech synthesis**,
one **router** that decides which pair a turn uses, and the language model in the
middle that does the actual thinking.

| | Set A | Set B |
|---|---|---|
| **recognition** | SraVaani | Whisper large-v3 |
| **synthesis** | Indic-Mio | MMS-TTS (VITS) |
| **router** | `sanchit-gandhi/mms-lid-ft-mix` — picks Set A or Set B, per utterance | |
| **reasoning** | Gemma 3 4B — replies in the language it was asked in | |

### Set A — 23 languages

English plus all 22 scheduled Indian languages:

> Assamese · Bengali · Bodo · Dogri · **English** · Gujarati · Hindi · Kannada ·
> Kashmiri · Konkani · Maithili · Malayalam · Manipuri · Marathi · Nepali ·
> Odia · Punjabi · Sanskrit · Santali · Sindhi · Tamil · Telugu · Urdu

### Set B — 26 languages

> Arabic · Bulgarian · Chinese · Croatian · Czech · Danish · Dutch · Filipino ·
> Finnish · French · German · Greek · Indonesian · Italian · Japanese · Korean ·
> Malay · Polish · Portuguese · Romanian · Slovak · Spanish · Swedish · Russian ·
> Turkish · Ukrainian

The two sets are specialised in opposite directions on purpose. Set A is *deep*:
trained almost exclusively on Indian speech, so accents, code-mixing and regional
phonology sit at the centre of its training distribution. Set B is *wide*: broad
coverage of everything else, at the cost of that depth. The router's only job is
to send each utterance to the one that will do it better.

---

## Running it

### 1. Setup

```bash
bash setup.sh          # Linux, macOS
```

```bat
setup.bat              REM Windows
```

That builds the three virtual environments, installs every package, downloads
the weights and verifies the result. It is safe to re-run: every step checks for
its own result first, so an interrupted run resumes where it stopped and a
finished one is a no-op.

Before the first run you need:

* **Python 3.14 and Python 3.12.** The stacks pin incompatible CUDA torch
  builds, so they cannot share an interpreter — see
  [Layout](#layout). Point at them with `PYTHON_314` / `PYTHON_312` if they are
  not on PATH.
* **A C toolchain.** `webrtcvad` has no prebuilt wheel and compiles on install
  — `build-essential` and the Python headers on Linux, Build Tools for Visual
  Studio with the C++ workload on Windows.
* **A Hugging Face token.** Two models are gated: accept the licence for
  `google/gemma-3-4b-it` and request access to `ARTPARK-IISc/SraVaani-1.0`, then
  `hf auth login` (or set `HF_TOKEN`).
* **About 25 GB of disk**, and a 12 GB NVIDIA card to actually run it. Every
  stage sets `require_cuda`.

The setup finishes with a smoke test — each environment imports what its worker
imports, the weights are where the config says, the reference clip is a shape
the voice will accept, and CUDA is visible. Nothing is loaded onto the GPU, so
it takes seconds.

### 2. Start the pipeline

```bash
venv/bin/python pipeline/run_realtime.py                  # Linux, macOS
venv\Scripts\python.exe pipeline\run_realtime.py           # Windows
```

Then talk. It listens continuously, answers out loud in the language you spoke,
and can be interrupted mid-reply.

To check the microphone and voice detection alone first — it starts instantly
and loads no models:

```bash
venv/bin/python pipeline/run_realtime.py --capture-only
```

To use a browser as the microphone and speakers instead of the local ones, with
the identical pipeline in between:

```bash
venv/bin/python -m pipeline.server
```

### SraVaani only — turning the router off

**This is how the repo ships.** The router and the whole international stack are
already suspended in [realtime.yaml](pipeline/config/realtime.yaml), so a fresh
clone runs SraVaani and Indic-Mio and nothing else. `setup.sh` fetches only
those weights to match.

Three flags control it, and **all three have to be `false`**:

```yaml
stt:
  lid:
    enabled: false       # the router: never classifies, never loads
  whisper:
    enabled: false       # the Set B ear: never loads
tts:
  mms_tts:
    enabled: false       # the Set B voice: never loads
```

Turning off only the router is not enough, and this is the part that catches
people. Routing has **two independent entry points**: STT picks the ear from the
audio, and TTS picks the voice from the *reply's* language. Silence the router
and a reply the model happens to produce in Spanish would still reach out for a
Spanish voice. `tts.mms_tts` closes that second door.

With all three off, every turn takes one path:

```
microphone → VAD → SraVaani → Gemma 3 4B → Indic-Mio → speaker
```

No language classification runs at all — `decide()` returns "local" before the
model is ever consulted, so the router costs nothing rather than being loaded
and ignored. STT reports no language, and the reply language is detected from
the transcript instead, which is what the pipeline did before the router
existed.

Startup says so plainly, and these three lines are the confirmation that the
router really is off:

```
! language ID off (disabled in config) — every turn routes to the local Indic models
! international stt off (disabled in config) — cannot transcribe languages outside the local set
! international tts off (disabled in config) — cannot speak languages outside the local set
```

What you give up is only what Set A never covered: speech in Spanish, Russian,
Japanese and the rest is transcribed by an Indic model that has never heard it,
which produces confident nonsense rather than an error. **Urdu and Kashmiri are
the two that quietly regress** — SraVaani's model card excludes both, and with
the router off they no longer reach an ear that can hear them. Everything else
in [Set A](#set-a--23-languages) is unaffected: 23 languages, entirely local.

Nothing is deleted by turning it off. Re-enabling is the three flags above.

### Turning the international stack on

Fetch the Set B weights, then set those same three flags to `true`:

```bash
bash setup.sh --with-set-b                             # ~6 GB more
MMS_TTS_LANGS=spa,fra,jpn bash setup.sh --with-set-b   # pick the voices
```

MMS-TTS ships one checkpoint per language, so only the ones you expect are
worth fetching. The codes are the ISO 639-3 keys of `MMS_TTS_VOICES` in
[languages.py](pipeline/realtime/languages.py).

`setup.sh` cross-checks the two: a flag set to `true` whose weights are missing
is reported as a failure rather than left to surface mid-sentence.

---

## The router

This is the part that makes the rest work.

The obvious design is to transcribe first and look at the text afterwards. That
fails, and it fails quietly. SraVaani is an *Indic* ASR — hand it Spanish and it
does not report an error, it returns confident-looking Devanagari gibberish. The
transcript cannot be used to decide whether the transcript should have been made,
because a wrong transcript looks exactly like a right one.

So the decision is made **before transcription, from the waveform itself**.

A language-identification model — **`sanchit-gandhi/mms-lid-ft-mix`**, a
966M-parameter wav2vec2 fine-tuned for spoken language ID — classifies the audio
directly in one small forward pass, about **90 ms**. Its answer picks the stack:

| audio is… | recognised by | spoken by |
|---|---|---|
| one of the 23 Set A languages | SraVaani | Indic-Mio |
| one of the 26 Set B languages | Whisper large-v3 | MMS-TTS |

Both Set B models are **loaded on demand** rather than held resident: Set A and
Set B are never both needed for the same turn, so the first turn that routes
abroad pays the load and every one after it is free.

### Routing by mass, not by label

The router has 126 possible labels, and only 15 of them are languages the local
stack handles. That imbalance matters: the top-scoring label is structurally
biased toward "foreign", because 111 of the 126 ways to be wrong point that way.

An English utterance that splits its probability across `eng`, `cym` (Welsh) and
`nno` (Norwegian Nynorsk) can top out on a foreign label while *local* was
overwhelmingly the right answer. Early on, this sent plain English to the foreign
stack.

The fix is to stop trusting the argmax. The router sums probability **per
destination** and keeps the turn local when the local total clears a threshold,
whatever the single highest label said:

| audio | local mass | routed |
|---|---|---|
| English, clean | 0.999 | local |
| English, 1.8 s fragment | 0.997 | local |
| Hindi/English code-mix | 0.822 | local |
| Tamil | 1.000 | local |
| noise / silence | 0.015 | — abstains |

Verified across the full label set: all 111 foreign labels still route foreign,
and none are wrongly trapped local. The fix is targeted, not a blanket "keep
everything at home".

### Failing toward local

Every default leans local, on purpose. A wrong local route costs one bad
transcript; a wrong foreign route costs a round trip and a delay. On a
mostly-Indic pipeline the ambiguous turns are mostly Indic, so ambiguity resolves
homeward.

The router also declines to answer at all when the evidence is too thin:

* **Utterances under 0.7 s** never reach it. Language ID on a fragment is a coin
  flip, and short turns ("yes", "mm", "wait") are overwhelmingly in the language
  already being spoken.
* **Low-confidence predictions** are discarded rather than acted on, and the turn
  stays local.

### Hysteresis, in one direction only

Once someone is speaking Spanish they keep speaking Spanish, and a single mumbled
sentence should not drop them onto a stack that cannot hear them. A short memory
window confirms a shaky prediction that agrees with the previous turn.

It only ever works that way — it can confirm a weak prediction, never overrule a
confident one. So switching *back* to Tamil takes effect on the very next
sentence rather than waiting for the window to lapse.

---

## Stage 1 — SraVaani (Indian languages + English)

A **~430M parameter FastConformer** with a hybrid TDT-CTC decoder, quantised to
FP16 (~900 MB, 1.69 GB resident).

Its training is what makes it the right ear for this pipeline:

1. **Pretraining** — 31,255 hours of speech from the Vaani dataset, spanning 105
   languages.
2. **Audio-image alignment** — 11 million audio-image pairs, using multimodal
   relationships to learn richer audio representations.
3. **Fine-tuning** — ~31,270 hours of transcribed speech across **65 Indian
   languages and dialects**.

That last stage is the point. General-purpose multilingual ASR treats Indian
languages as a long tail; SraVaani was fine-tuned on almost nothing else, so
accents, code-mixing and regional phonology are the centre of its training
distribution rather than its edge.

It covers all the major scheduled languages — Assamese, Bengali, Gujarati, Hindi,
Kannada, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi,
Sanskrit, Santali, Tamil, Telugu — plus English, and a long tail of
non-scheduled ones (Bhojpuri, Chhattisgarhi, Tulu, Mizo, Garo, Khortha and
dozens more).

**Two exceptions matter:** SraVaani does not support **Urdu** or **Kashmiri**.
See [Where the two halves disagree](#where-the-two-halves-disagree).

## Stage 1b — Whisper large-v3 (Set B, 26 languages)

For everything outside that range — Spanish, Russian, Chinese, Japanese, Arabic —
**Whisper large-v3** handles the turn. A 1.55B-parameter encoder-decoder
transformer covering 99 languages, it is the strongest open ASR available for
broad multilingual coverage, and it is deliberately the opposite specialisation
to SraVaani: wide rather than deep.

It detects the language itself as part of transcribing, and **its answer
outranks the router's**: it heard the words, not just the accent.

This is also how a misroute repairs itself. If the router guesses Spanish and the
recogniser hears Tamil, the turn is Tamil from that point on, and synthesis goes
back to the local voice.

---

## Stage 2 — the language model

**Gemma 3 4B**, 4-bit quantised (~3 GB instead of ~8 GB, so all the models fit on
one 12 GB card together).

The reply must be in the language the user spoke. The prompt asks for this, and a
4B model under load ignores it — it was observed correctly identifying "epdi
irukka" as Tamil and then answering in English anyway.

So the language is not left to the model to infer. It is decided in code and
stated as a per-turn directive:

* The **transcription stage** passes the language it established from the audio.
  That is strictly better than re-deriving it from the transcript, which for a
  Latin-script language would be plainly wrong.
* The directive is appended **last**, after the system prompt and again
  immediately before the user turn. Position beats emphasis — a rule at the end
  of the prompt is followed far more reliably than one marked "IMPORTANT" in the
  middle.
* **History is cleared on a language switch.** Several turns of English right
  before a Tamil question outweigh any instruction; dropping the history removes
  the pull.

Replies are capped short (96 tokens). Every token is also synthesis time, so this
is the main latency lever in the whole pipeline.

---

## Stage 3 — Indic-Mio (Indian languages + English)

A **0.6B causal LM** that speaks all 22 scheduled Indian languages plus English
from one set of weights. There is no per-language voice file: it infers the
language from the script of the text it is given.

Generation produces ordinary text tokens interleaved with audio tokens in a
reserved id range. Those audio tokens are extracted, shifted back to codec codes,
and decoded to a waveform.

### One voice, not a new one every turn

Indic-Mio is **zero-shot**: it has no built-in voice. Given no reference speaker
it invents one per utterance, sampled at temperature 0.9 — so consecutive replies
come back sounding like different people. The model is working as designed and
the result still sounds broken.

A reference clip fixes this. It is encoded once at startup into a 128-dimension
speaker embedding that conditions every synthesis. Measured speaker similarity
between independently generated turns: **0.99**, including across an
English → Tamil switch.

### The codec has to match

The audio tokens index a specific codec's codebook. Decoding them with a
different codec does not fail — it returns fluent, correctly-timed, **completely
wrong speech**, which to a listener sounds like the assistant answering in some
other language entirely.

Indic-Mio pairs with `MioCodec-25Hz-24kHz`, which decodes straight to a waveform.
Verified by transcribing the output back:

| asked to say | actually said |
|---|---|
| "Hello there. I'm doing well, thank you for asking." | `hello there im doing well thank you for asking` |
| "Did you want to talk about something specific today?" | `did you want to talk about something specific to day` |
| (Tamil greeting) | `வணக்கம் நான் நல்லா இருக்கேன்` |

## Stage 3b — MMS-TTS (Set B, 26 languages)

Replies in languages Indic-Mio does not speak are synthesised by **MMS-TTS**, the
VITS-based synthesis model from Meta's Massively Multilingual Speech project.
It ships a compact per-language checkpoint (~145 MB) for over
a thousand languages, so only the ones actually in use need to be resident.

Each checkpoint carries one fixed speaker, and there is no cloning: the Set B
voice is a different person from the local one, and a language switch is
audible as a speaker switch. That is the cost of keeping synthesis local and
small.

VITS is non-autoregressive, which matters here: it synthesises in roughly
constant time rather than token by token, so a long reply in Set B does not cost
proportionally more latency the way an autoregressive voice does.

Both paths are deliberately interchangeable below the waist: both produce a
float32 waveform, both go through the same loudness normalisation, both are
written to the same WAV.

That matters more than it looks — loudness normalisation is the first layer of
echo suppression, and an un-normalised reply would come back hot enough to trip
the barge-in gate and cut itself off mid-sentence.

---

## Where the two halves disagree

The ear and the voice do not cover the same languages, and assuming they do is
how a turn ends up at a model that cannot handle it.

| | covers |
|---|---|
| **Indic-Mio** (voice) | English + all 22 scheduled Indian languages |
| **SraVaani** (ear) | English + 65 Indian languages and dialects, **minus Urdu and Kashmiri** |

So routing is answered **per stage**, not once per turn:

* the **ear** asks which model should transcribe this audio
* the **voice** asks which model should speak this reply

**Urdu and Kashmiri** are the case that forces the split. SraVaani cannot hear
them, Indic-Mio speaks them fluently — so they are transcribed by Whisper and
spoken by Indic-Mio. Any single shared list would have broken one half or the
other.

The voice is also chosen from the *reply's* language, not from the route the
user's audio took. They usually agree; when they do not, the reply is what is
about to be spoken.

---

## Live conversation

The pipeline is interruptible, which means the microphone stays open while the
assistant is talking — and that creates a feedback loop to defend against.

**Barge-in** works in two tiers, because the two halves of the decision have
different deadlines:

* **Tier 1** — sustained speech over a strict voice-activity gate stops playback
  in ~240 ms. Fast, but it cannot tell the user from the assistant's own speakers.
* **Tier 2** — transcription and the echo guard rule on that utterance 1–2 s
  later. Real speech goes to the model and pending replies are flushed; the
  assistant's own bleed is discarded.

Stopping is **final** — audio never resumes once interrupted. A reply therefore
cannot be re-triggered by its own bleed.

**Echo suppression** runs in layers, from cheapest to most precise: output
loudness normalisation, muting capture during playback, a reverb-tail delay after
the speakers go quiet, and finally a text-level guard that compares each
transcript against recent replies and drops matches. The last layer is the only
one that *identifies* echo rather than reducing its odds.

---

## Resource use

All six models are sized to share one 12 GB card — see
[Models at a glance](#models-at-a-glance) for the per-model breakdown.

Each stage runs as a **separate process in its own virtual environment**, talking
JSON-lines over stdin/stdout. That is not architectural purity — the three model
stacks have mutually incompatible dependency pins and cannot share an interpreter.
Bounded queues between stages provide backpressure, so a slow stage throttles its
upstream rather than accumulating a backlog.

---

## Layout

```
setup.sh           one-shot setup: environments, packages, weights, smoke test
setup.bat          the same, for Windows
tools/             what both setup scripts share, so they cannot drift
pipeline/
  run_realtime.py  the entrypoint: microphone in, speakers out
  realtime/        capture, VAD, routing tables, echo guard, barge-in, session
  workers/         one subprocess per model stage
  config/          realtime.yaml and the system prompt
  server/          WebSocket front end
stt/models/        SraVaani, the router, and Whisper large-v3
tts/models/        Indic-Mio, its codec, and the MMS-TTS voices
tts/voice/         the reference clip that fixes the assistant's voice
reasoning/models/  Gemma 3 4B
web/               browser front end
docs/voice.md      voice setup, language coverage, and past failure modes
```

The three environments are built beside the code they host — `venv/` at the
root, `reasoning/venv/` and `tts/venv/` next to their models — because the
stacks pin incompatible CUDA torch builds and cannot share an interpreter.

Two entrypoints share the identical graph and differ only at the ends:

```
terminal   microphone -> ... -> speakers
web        browser    -> ... -> browser
```

Everything in between — voice activity detection, routing, barge-in, the echo
guard — is the same code.

---

## Models at a glance

| stage | model | role | params | VRAM (approx) |
|---|---|---|---:|---:|
| **Router** | `sanchit-gandhi/mms-lid-ft-mix` | picks Set A or Set B from the waveform | 966 M | 1.9 GB |
| **STT — Set A** | SraVaani (FastConformer, TDT-CTC) | 23 Indian languages + English | 430 M | 1.7 GB |
| **STT — Set B** | Whisper large-v3 | 26 world languages | 1.55 B | 3.1 GB |
| **Reasoning** | Gemma 3 4B Instruct (4-bit) | generates the reply | 4 B | 3.0 GB |
| **TTS — Set A** | Indic-Mio (+ MioCodec 25 Hz) | 23 Indian languages + English | 0.6 B | 1.1 GB |
| **TTS — Set B** | MMS-TTS (VITS, per language) | 26 world languages | 36 M | 0.15 GB |

**Steady state ≈ 7.7 GB** — router, SraVaani, Gemma and Indic-Mio resident
together. Set A and Set B are never both needed for the same turn, so the Set B
models are loaded on demand rather than held alongside their Set A counterparts;
holding all six at once would come to roughly 11 GB.

As the repo ships the router is suspended
([SraVaani only](#sravaani-only--turning-the-router-off)), which drops the
resident set to SraVaani, Gemma and Indic-Mio — **≈ 5.8 GB**. The router is the
second-largest model here, so turning it off is the single biggest VRAM saving
available short of dropping a stage.

Precision matters most at the router. It is the second-largest model in the
pipeline and its entire output is one argmax plus a probability sum, so nothing
on that path needs float32 — running it in half precision costs **1.9 GB instead
of 3.7 GB**, with predictions identical to three decimal places across clean
English, Tamil, Gujarati, a 1.8 s fragment, a borderline code-mix and noise.

Quantisation elsewhere: Gemma runs 4-bit (~3 GB rather than ~8 GB), SraVaani
ships FP16, Indic-Mio runs bf16.
