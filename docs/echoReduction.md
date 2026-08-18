# Echo Reduction — Stopping the Assistant From Hearing Itself

How the real-time speech loop avoids replying to its own voice when it plays
through **open speakers** instead of headphones.

- Code: [echo_guard.py](../pipeline/realtime/echo_guard.py),
  [workers.py](../pipeline/realtime/workers.py),
  [capture.py](../pipeline/realtime/capture.py),
  [run_realtime.py](../pipeline/run_realtime.py)
- Config: [realtime.yaml](../pipeline/config/realtime.yaml) (`capture.vad`, `reason.*`, `tts.normalize`)

---

## 1. The problem

The loop is mic → STT → reasoning → TTS → speakers. With speakers open, the
mic hears the reply. Two distinct failures follow, and they need different
fixes:

| Failure | What happens |
|---|---|
| **Self-interrupt** | Barge-in fires on the assistant's own first words, cutting its own reply short and (with `reset_memory_on_barge_in`) wiping conversation memory. |
| **Self-reply** | The bleed is transcribed, reaches the model as a user turn, and the assistant answers itself — a runaway loop. |

The root cause is that the acoustic layer has no echo cancellation. VAD
decides on raw frame energy, roughly one debounce window into *any* sound,
long before a single word exists. There is nothing in the waveform that
distinguishes "your voice" from "our voice coming back". Tuning VAD
aggressiveness only trades false interrupts against slow interrupts; it never
separates the two sources.

What *is* separable arrives one step later: **the transcript**. If the mic
heard "I can help with that" while the assistant was in the middle of saying
"I can help with that", that is bleed, not a human.

---

## 2. Defence in depth

Five independent layers. The first three reduce how much bleed exists; the
last two catch what survives.

```text
                 ┌─ (1) quieter output      tts.normalize.target_lufs
                 ├─ (2) mic muted while replying   reason.mute_capture_while_replying
   acoustic ─────┤
                 ├─ (3) grace window        reason.barge_in_grace_ms
                 └─ (4) longer debounce     capture.vad.speech_start_debounce_ms
                                 │
   textual ──────────────────────┴─ (5) echo guard   reason.echo_guard
```

### (1) Put less energy in the room — `tts.normalize.target_lufs: -23.0`

The feedback loop is ultimately acoustic, so the most effective software
lever is amplitude. Default streaming loudness (`-16 LUFS`) is hot enough to
bleed on most desk setups; `-23` is well below it.

Turn the *system* volume up if replies are too quiet. A quiet signal amplified
downstream keeps a better speaker-to-mic ratio than a hot signal does.

### (2) Mute capture while replying — `reason.mute_capture_while_replying: true`

The strongest guarantee: capture drops frames entirely while the `speaking`
event is set, so bleed is never recorded. Note this is **mutually exclusive
with barge-in** — a muted mic cannot hear an interruption. `run_realtime.py`
enforces the choice: `mute_while_replying` is only enabled when `barge_in` is
off.

### (3) Grace window — `reason.barge_in_grace_ms: 2000`

When barge-in *is* on, ignore speech starts for this long after audio output
begins. The opening of a reply is its loudest, most bleed-prone moment, and
the original `900 ms` let the assistant's first words trigger a self-interrupt.
Raised to `2000 ms`.

Cost: you cannot barge in during that window.

### (4) Longer speech debounce — `capture.vad.speech_start_debounce_ms: 700`

Barge-in requires this many milliseconds of *consecutive* speech frames.
Speaker bleed tends to arrive in short bursts, so a longer continuous run is
much harder for it to clear, while sustained human speech still does.
Raised `300 → 700`.

Cost: ~400 ms slower barge-in response.

### (5) Echo guard — the text layer

Layers 1–4 all trade responsiveness for safety and none of them can be made
airtight. The echo guard is the only layer that actually *identifies* echo
rather than reducing its odds, and it is the layer that stops the self-reply
loop. Independent of `barge_in`; on by default via `reason.echo_guard`.

---

## 3. How the echo guard works

Two pieces in [echo_guard.py](../pipeline/realtime/echo_guard.py):

**`RecentSpeech`** — a thread-safe rolling window (default 6 chunks) of what
the assistant has recently said aloud. A window rather than just the newest
chunk because a reply is streamed as several chunks and playback lags
generation, so at the moment bleed is captured the audible sentence may be any
of the last few.

**`is_echo_of(transcript, spoken, threshold=0.6)`** — scores the fraction of
transcript words that appear in what was spoken.

Three deliberate design choices:

1. **Containment, not similarity.** The mic catches a *fragment* of a much
   longer reply. Symmetric measures (SequenceMatcher ratio, Jaccard) are
   dominated by the length mismatch and score real echoes near zero.
2. **Normalize before comparing.** STT punctuates and capitalizes differently
   from the raw reply text, so a literal string compare would miss nearly
   every real echo. Both sides are lowercased, stripped of punctuation, and
   split into words.
3. **Never flag anything under 3 words.** "yes", "ok", "sure" are common to
   both speakers and would false-positive against any long reply containing
   them. A missed suppression costs one memory reset; wrongly ignoring a real
   interruption costs the user their turn.

It is **not** a human-vs-bot or language classifier. The same sentence is
valid from either speaker — the only signal is overlap with what the assistant
is *currently* saying.

---

## 4. Where it is wired in

**Write side — TTS worker** ([workers.py](../pipeline/realtime/workers.py),
`tts_worker`): just before synthesis, `recent_speech.add(tr.text)`.

Recorded at synthesis rather than at generation because text that never
reaches synthesis never reaches the room. Aborted output would otherwise
poison the window with words nobody heard.

**Read side — STT worker** (`stt_worker`): after the noise/filler filter, if
the transcript is an echo of the window snapshot, log it as
`<echo of assistant output, dropped>` and `continue` — it never reaches
`transcript_queue`, so the model never sees it.

**Ownership** — `run_realtime.py` constructs one `RecentSpeech` (only in
`reason` mode, only when `reason.echo_guard` is set) and hands the same
instance to both workers, plus an `on_echo` callback.

---

## 5. What the echo guard cannot undo

Dropping the transcript stops the *self-reply*. It does **not** un-do a
self-interrupt that already happened, because barge-in fires on VAD energy
seconds before the transcript exists — and waiting for the transcript would
mean seconds of the assistant talking over you. By the time echo is known:

- the aborted audio cannot be resumed (tokens are discarded on abort), and
- the memory wipe has already fired.

So `on_echo` logs a loud warning instead:

> Echo detected — assistant heard its own output. If this repeats, lower the
> output volume or set `reason.barge_in: false`.

That message means the acoustic layers (1–4) are mis-tuned for the room, and
the text layer is covering for them.

---

## 6. Config reference

| Key | Value | Purpose |
|---|---|---|
| `reason.echo_guard` | `true` | Enable text-level echo detection |
| `reason.echo_threshold` | `0.6` | Word-overlap fraction to call it echo. Lower = more aggressive (may swallow real speech that quotes the assistant); higher = more echo slips through |
| `reason.barge_in_grace_ms` | `2000` | Ignore speech starts this long after output begins |
| `reason.mute_capture_while_replying` | `true` | Drop mic frames while speaking (excludes barge-in) |
| `capture.vad.speech_start_debounce_ms` | `700` | Consecutive speech required for barge-in |
| `tts.normalize.target_lufs` | `-23.0` | Output loudness; lower = less bleed |

---

## 7. Tuning guide

| Symptom | Fix |
|---|---|
| Assistant replies to itself | Confirm `reason.echo_guard: true`; lower `echo_threshold` toward `0.5` |
| Assistant cuts itself off | Raise `barge_in_grace_ms`; raise `speech_start_debounce_ms`; lower `target_lufs` |
| Real interruptions ignored | Raise `echo_threshold` toward `0.75`; lower `barge_in_grace_ms` |
| Barge-in feels sluggish | Lower `speech_start_debounce_ms` (accepting more false interrupts) |
| Nothing works on this desk | Use headphones, or `reason.barge_in: false` + `mute_capture_while_replying: true` — the only airtight combination |

The genuinely airtight fix is physical: **headphones**, or a directional mic
pointed away from the speakers. Every software layer above is mitigation.

---

## 8. Tests

- [test_echo_guard.py](../pipeline/tests/test_echo_guard.py) — `_norm`,
  `RecentSpeech` windowing/threading, `is_echo_of` thresholds and the
  short-transcript guard.
- [test_echo_guard_integration.py](../pipeline/tests/test_echo_guard_integration.py)
  — the worker path: TTS records into the window, STT drops an echoing
  transcript, real speech passes through while the assistant talks, and the
  guard is inert when disabled.