# International Languages — Two Ears, Two Voices, One Pipeline

How a turn spoken in Spanish, Russian or Japanese is heard and answered, when
every local model in the stack is Indic.

- Code: [languages.py](../pipeline/realtime/languages.py),
  [elevenlabs.py](../pipeline/realtime/elevenlabs.py),
  [worker_stt.py](../pipeline/workers/worker_stt.py),
  [worker_tts.py](../pipeline/workers/worker_tts.py),
  [workers.py](../pipeline/realtime/workers.py)
- Config: [realtime.yaml](../pipeline/config/realtime.yaml) (`elevenlabs.*`,
  `stt.lid.*`)
- Prerequisite: `ELEVENLABS_API_KEY` in the environment
- Tests: [test_languages.py](../pipeline/tests/test_languages.py) (routing
  tables, the gate policy), [test_elevenlabs.py](../pipeline/tests/test_elevenlabs.py)
  (wire format, retry policy),
  [test_stt_routing.py](../pipeline/tests/test_stt_routing.py) (the real worker
  loop against stubbed models),
  [test_international_flow.py](../pipeline/tests/test_international_flow.py)
  (the language surviving every stage)

---

## 1. Why this is not just "add a second model"

The local stack has a hard boundary that no amount of tuning moves:

| Stage | Model | Hears / speaks |
|---|---|---|
| STT | SraVaani 1.0 | 22 scheduled Indian languages + English |
| TTS | Indic-Mio | the same set, from one set of weights |

So the set of languages the pipeline can handle locally is fixed, and its
complement — Spanish, Russian, Japanese, Arabic, Korean, everything else — has
**no local ear and no local voice**. That complement is what this path is for.

The obvious implementation is a branch on the detected language. The problem is
where the branch can possibly go.

**SraVaani does not fail on Spanish.** It is not a multilingual model with an
out-of-vocabulary signal; it is an Indic ASR that maps any audio onto Indic
output. Hand it Spanish and it returns fluent-looking Devanagari or romanized
text, with no flag and no error. Downstream, that transcript is indistinguishable
from a real one — the LLM answers it, the TTS speaks the answer, and the user
gets a confident reply to a sentence they never said.

Which means the transcript **cannot** be the input to the routing decision. By
the time it exists, the mistake has already been made and erased.

```
Spanish audio ──▶ SraVaani ──▶ "मेरे को समझ नहीं आया" ──▶ ??? 
                                 ↑
                                 nothing here says "this was Spanish"
```

The decision has to happen earlier, on the only artefact that still carries the
answer: the waveform.

---

## 2. The language gate

`facebook/mms-lid-126` classifies the language of raw audio in one forward pass
— a wav2vec2 encoder with a 126-way classification head, ~1.2 GB, tens of
milliseconds on the same card the other models are already on. It runs in the
STT worker, before transcription, and its answer chooses the ear:

```
        ┌──────────────┐   indic / english   ┌──────────────┐
audio ─▶│  MMS-LID 126 │────────────────────▶│  SraVaani    │─▶ transcript
        │              │                     └──────────────┘
        └───────┬──────┘   anything else     ┌──────────────┐
                └────────────────────────────▶│  Scribe      │─▶ transcript
                                             └──────────────┘
```

The gate never decides *what was said* — only *who should listen*. That split
matters, and §4 is about why.

### 2.1 The policy is not the model

The forward pass lives in the worker; what to do with its answer lives in
`RouteGate` in [languages.py](../pipeline/realtime/languages.py). The split is
deliberate: the thresholds are the part that gets tuned and the part that can
be wrong, and none of it needs a GPU to reason about or to test.

| Knob | Default | What it protects against |
|---|---|---|
| `min_confidence` | `0.55` | acting on a guess |
| `min_audio_s` | `0.7` | LID on a fragment, which is a coin flip |
| `max_audio_s` | `5.0` | paying for a 15 s forward pass to learn a 3 s answer |
| `sticky_ttl_s` | `60` | one mumbled sentence dropping a conversation |
| `hint_confidence` | `0.85` | handing Scribe a confidently wrong hint |

### 2.2 Every default fails toward local

When the gate is unsure, the turn stays local. This is asymmetric on purpose:

- A wrong **local** route costs one bad transcript, visible immediately, on a
  turn the user will simply repeat.
- A wrong **international** route costs a paid API call, and on a pipeline whose
  traffic is mostly Indic, ambiguous turns are mostly Indic too.

So `min_confidence` is a floor for *spending*, not for accuracy.

### 2.3 Hysteresis that can only confirm

People do not alternate languages sentence by sentence. Once someone is speaking
Spanish they keep speaking Spanish, so a single dip below `min_confidence`
should not drop them onto an ear that cannot hear them.

But the naive fix — a sticky session language — breaks the requirement that
matters most: *"suddenly the user talks in an Indian language, use the local
models."* A sticky language that overrides predictions would strand them abroad
until it expired.

So the window only ever **confirms a weak prediction that already agrees with
the previous turn**. It can never overrule a confident one:

| Previous turn | This prediction | Confidence | Route |
|---|---|---|---|
| Spanish | Spanish | 0.30 | **international** — confirmed by the window |
| Spanish | Tamil | 0.90 | **local** — confident, immediately, no delay |
| Spanish | Russian | 0.30 | local — weak, and not what the window holds |
| Tamil | Spanish | 0.30 | local — a local turn clears the window |

Switching back to Tamil therefore lands on the very next sentence.

One further distinction does a lot of work: a turn where the gate **abstained**
leaves the window open, while a turn that resolved to a local language closes
it. An abstention is exactly the case the window exists to cover, so treating
it as a language change would close the window on the turns it was built for.
Only a confident local prediction closes it — which is the same thing that
makes switching back immediate.

Set `sticky_ttl_s: 0` to remove the mechanism entirely.

---

## 3. What is "local" is not configurable

There is no list of international languages to maintain, and no `enabled`
switch per language. There is one set —

```python
LOCAL = {english, hindi, bengali, marathi, telugu, kannada, tamil, malayalam,
         gujarati, punjabi, odia, urdu, nepali, assamese, sanskrit, konkani,
         maithili, dogri, bodo, santali, sindhi, manipuri, kashmiri}
```

— and `route_for()` is `local if lang in LOCAL else international`. The
complement is implicit, so the two halves cannot drift apart, and a language
nobody anticipated routes abroad by default rather than being silently called
English.

That default is load-bearing. `language_from_code()` returns an unrecognized ISO
code *as-is* rather than dropping it, precisely so it lands outside `LOCAL` and
reaches the stack that might actually handle it.

### 3.1 A side effect worth knowing about

Before audio-level LID, the script-range detector bucketed every Devanagari
language under `hindi` and every Bengali-script one under `bengali` — names like
`marathi` and `konkani` could not occur. LID names them directly, so they now
reach the TTS worker, which had to learn to recognize them or refuse a reply it
can perfectly well speak. `test_international_flow.py` pins this: every language
in `LOCAL` must be one Indic-Mio speaks.

---

## 4. Scribe outranks the gate

The gate picks the route. **Scribe picks the language.**

Scribe transcribed the words; LID classified the accent from a few seconds of
audio. Where they disagree, the one that heard the sentence is right — so the
language reported by Scribe is what the rest of the turn runs on.

This makes a misroute self-repairing:

```
LID: "spanish" (0.72)  ──▶  Scribe  ──▶  language_code: "tam"
                                             │
                                             ▼
                            the turn is Tamil from here on:
                            Tamil directive to the LLM,
                            Tamil reply, Indic-Mio voice
```

The cost of the mistake is one API call. The user never hears it.

The LID answer is passed to Scribe as a `language_code` hint only above
`hint_confidence` (0.85), because a confidently wrong hint is worse than none —
it pins Scribe to a language it would otherwise have detected correctly.

---

## 5. The language travels; it is never re-derived

`Sentence.lang` → `Reply.lang`, established once at STT and carried down.

This is not an optimization. The text-based detector in `languages.py` works on
script ranges and romanized function words, and **Latin script has neither for
Spanish**. Re-running detection on a Spanish transcript returns `english` — the
fallback — every single time:

```python
detect_language("¿Cómo estás hoy?")   # -> "english"
```

So a pipeline that re-detects at each stage would transcribe Spanish correctly
and then answer it in English, in an English voice. The LLM worker takes
`req["lang"]` when it is present and only falls back to `detect_language()` when
it is not — which is the pre-existing behaviour for every turn the gate declined
to commit to.

The non-Indic script ranges added to the detector (Cyrillic, Greek, Hangul,
kana, Han, Hebrew, Thai, Arabic) are for that fallback path only. Two of them
needed disambiguation:

- **Han is shared.** A Japanese sentence lands in both the `japanese` and
  `chinese` buckets and can lose the majority vote to its own kanji. Kana are
  Japanese-only, so their presence settles it before counting.
- **Perso-Arabic is shared, and the two halves route differently.** Urdu is one
  of Indic-Mio's languages; Arabic is not. Arabic is the default and Urdu is
  promoted out of it by its own letters (ٹ ڈ ڑ ں ھ ہ ے), which real Urdu text
  is dense with.

---

## 6. The voice is chosen by the reply

TTS does not inherit the STT route. It calls `route_for()` on the *reply's*
language, because the reply is what is about to be spoken. When the two agree —
almost always — nothing is different. When they do not, §4's self-repair works
end to end.

Below that choice the two backends are deliberately interchangeable:

```
Indic-Mio ──┐
            ├──▶ float32 waveform ──▶ normalize() ──▶ WAV ──▶ Player
ElevenLabs ─┘
```

**Both paths normalize.** This is not tidiness — `normalize()` is echo-reduction
layer 1 ([echoReduction.md](echoReduction.md)), and the barge-in gate
([bargeIn.md](bargeIn.md)) is tuned against its output. An un-normalized
ElevenLabs reply would arrive at full scale in a room whose gate expects −23
LUFS, bleed into the mic, clear the strict VAD threshold, and interrupt itself
mid-sentence. Same file, same player, same guard: an international reply is
interruptible exactly like a local one.

Audio is requested as raw PCM (`output_format: pcm_24000`), not MP3, so there is
nothing to decode — the bytes are reinterpreted as float32 and handed straight to
the normalizer. A non-PCM `output_format` is refused at construction rather than
producing a WAV that fails to open one reply later.

---

## 7. Failure is soft, and stated out loud

Both halves are optional, and each degrades to exactly the behaviour that
existed before this path did.

| Missing | Effect | Where you find out |
|---|---|---|
| LID weights | every turn routes locally | startup: `! language ID off (…)` |
| `ELEVENLABS_API_KEY` | international turns dropped | startup, and per turn |
| Network / 5xx | that one turn fails | `tts_failed` / `stt_failed` |
| No ElevenLabs voice | reply printed, not spoken | `no <lang> voice, text only:` |

Indic and English are unaffected in every row.

The one thing the pipeline refuses to do is transcribe international audio
locally anyway. That would produce confident gibberish that reads as a working
pipeline giving a strange answer — the single worst outcome available — so the
turn is dropped and the reason named:

```
[u7] ! spanish speech, but ElevenLabs is not configured — turn dropped.
     Set $ELEVENLABS_API_KEY and restart.
```

### 7.1 Heard is not the same as speakable

Scribe transcribes 99 languages; `eleven_multilingual_v2` speaks 29. The gap is
real and is handled by the existing missing-voice path rather than by guessing:
**Sinhala, Vietnamese, Thai, Hebrew, Hungarian, Norwegian, Persian, Swahili,
Afrikaans**. Those replies are printed instead of mispronounced.

`test_international_flow.py` pins that list exactly, so a new gap can appear
only deliberately.

### 7.2 Retries are silence

A retry in a realtime loop is time the user spends listening to nothing. So only
transient failures are retried (429, 5xx, transport), once by default, with a
brief fixed backoff — exponential would be right for a batch job and wrong here,
where past a second or so the turn is stale and failing fast is the better
answer. A 4xx is a bad key or a bad request and returns the same answer however
many times it is asked, so it is raised immediately. A 401 says which
environment variable to set.

---

## 8. Cost, and seeing it

Every international turn is two paid calls: one Scribe transcription and one
synthesis. Nothing else in the pipeline costs money, so the transcript line
names the backend only when it is not the local one — a tag on every line would
be noise, a tag on these is the only visible sign that a call was made:

```
[u3] heard: எப்படி இருக்கீங்க                      ← free
[u4] heard (spanish via elevenlabs): ¿Cómo estás?  ← paid
[u4] synth 1.4s in 0.9s via elevenlabs             ← paid
```

If those tags appear on turns that were actually Indic, raise
`stt.lid.min_confidence`.

---

## 9. Setup

```bash
bash download.sh --skip-venvs        # fetches facebook/mms-lid-126 (~1.2 GB)
export ELEVENLABS_API_KEY=sk_xxx
```

To turn the path off entirely, without removing anything:

```yaml
stt:
  lid:
    enabled: false      # every turn routes locally
elevenlabs:
  enabled: false        # no calls made from either worker
```

---

## Related docs

- [pipline.md](pipline.md) — the pipeline this sits inside
- [echoReduction.md](echoReduction.md) — why both voices must normalize
- [bargeIn.md](bargeIn.md) — the gate that normalization protects
- [systemPrompt.md](systemPrompt.md) — the reply-language directive
