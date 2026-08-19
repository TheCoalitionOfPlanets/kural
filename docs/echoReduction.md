# Echo Reduction — Stopping the Assistant From Hearing Itself

How the real-time speech loop avoids replying to its own voice when it plays
through **open speakers** instead of headphones.

- Code: [echo_guard.py](../pipeline/realtime/echo_guard.py),
  [interrupt.py](../pipeline/realtime/interrupt.py),
  [workers.py](../pipeline/realtime/workers.py),
  [capture.py](../pipeline/realtime/capture.py),
  [audio_out.py](../pipeline/realtime/audio_out.py),
  [run_realtime.py](../pipeline/run_realtime.py)
- Config: [realtime.yaml](../pipeline/config/realtime.yaml) (`capture.vad`, `barge_in.*`,
  `echo.*`, `tts.normalize`)

> Barge-in — interrupting a reply in progress — is the other half of this
> problem and is documented in [bargeIn.md](bargeIn.md). The echo guard is what
> makes it possible: it is the only layer that can tell the user's voice from the
> assistant's own, so it is what decides whether an interrupt was real.

---

## 1. The problem

The loop is mic → STT → reasoning → TTS → speakers. With speakers open, the
mic hears the reply. Two distinct failures follow, and they need different
fixes:

| Failure | What happens |
|---|---|
| **Self-interrupt** | Barge-in fires on the assistant's own first words, cutting its own reply short. |
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

Five independent layers. The first four reduce how much bleed exists; the
last one catches what survives.

```text
                 ┌─ (1) quieter output      tts.normalize.target_lufs
                 ├─ (2) mic muted while replying   echo.mute_capture_while_replying
   acoustic ─────┤       (or, with barge-in on, the strict gate below)
                 ├─ (3) grace window        capture.vad.barge_in_grace_ms
                 └─ (4) strict gate + debounce
                 │         capture.vad.barge_in_energy_multiplier
                 │         capture.vad.barge_in_aggressiveness
                 │         capture.vad.barge_in_debounce_ms
   textual ──────┴─ (5) echo guard   echo.guard
```

### (1) Put less energy in the room — `tts.normalize.target_lufs: -23.0`

The feedback loop is ultimately acoustic, so the most effective software
lever is amplitude. Default streaming loudness (`-16 LUFS`) is hot enough to
bleed on most desk setups; `-23` is well below it.

Turn the *system* volume up if replies are too quiet. A quiet signal amplified
downstream keeps a better speaker-to-mic ratio than a hot signal does.

### (2) Mute capture while replying — `echo.mute_capture_while_replying: true`

The strongest guarantee: capture drops frames entirely while the `speaking`
event is set, so bleed is never recorded. This is **mutually exclusive with
barge-in** — a muted mic cannot hear an interruption — so it is ignored when
`barge_in.enabled` is set, and layers 3–4 take over as the acoustic defence.

### (3) Grace window — `capture.vad.barge_in_grace_ms: 350`

Ignore speech starts for this long after audio output begins. The opening of a
reply is its loudest, most bleed-prone moment.

Cost: you cannot barge in during that window. It is short here only because
layer 5 can now *reverse* a false interrupt — see
[bargeIn.md](bargeIn.md) — so a mistake in this window costs a replayed
sentence rather than a swallowed reply.

### (4) Strict gate + debounce — `capture.vad.barge_in_*`

While the speakers are audible, VAD demands more before it will call something
an interruption:

- `barge_in_energy_multiplier: 2.5` — energy must exceed `2.5 ×` the calibrated
  threshold. The user's voice at the mic is far louder than a reply that has
  crossed the room, so most bleed is rejected outright.
- `barge_in_aggressiveness: 3` — the strictest WebRTC setting, used only during
  playback.
- `barge_in_debounce_ms: 240` — the speech must be *consecutive*. Bleed arrives
  in bursts shaped by the reply's own syllables; a sustained run is much harder
  for it to clear than a single loud frame.

Bleed is voiced audio, so the VAD is right to call it speech. Demanding it be
louder and sustained is the only acoustic lever available.

### (5) Echo guard — the text layer

Layers 1–4 all trade responsiveness for safety and none of them can be made
airtight. The echo guard is the only layer that actually *identifies* echo
rather than reducing its odds. It stops the self-reply loop, and with barge-in
on it is also the verdict that decides whether an interrupt was the user or the
assistant hearing itself. On by default via `echo.guard`.

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

**Write side — TTS stage** ([workers.py](../pipeline/realtime/workers.py),
`TTSStage.run`): just before synthesis, `recent_speech.add(reply.text)`.

Recorded at synthesis rather than at generation because text that never
reaches synthesis never reaches the room. A reply that fails to synthesize
would otherwise poison the window with words nobody heard.

**Read side — STT stage** (`STTStage.run`): after the empty-transcript filter,
if the transcript is an echo of the window snapshot, emit `echo_dropped` and
`continue` — it never reaches `transcript_queue`, so the model never sees it.

**Ownership** — `run_realtime.py` constructs one `RecentSpeech` (when
`echo.guard` is set) and hands the same instance to both stages, plus an
`on_echo` callback.

---

## 5. What the echo guard can and cannot undo

Dropping the transcript stops the *self-reply*, always.

Whether it can undo a *self-interrupt* depends on whether barge-in is on:

- **`barge_in.enabled: false`** — nothing to undo. The mic is muted during
  playback, so no interrupt can fire in the first place.
- **`barge_in.enabled: true`** — yes, and this is the point of the two-tier
  design in [bargeIn.md](bargeIn.md). Stopping playback is made *provisional*
  precisely so the echo guard's verdict, ~1–2 s later, can reverse it and play
  the reply again. The cost of a false interrupt is a restarted sentence.

Either way `on_echo` logs a warning, because echo reaching the text layer at all
means the acoustic layers (1–4) are mis-tuned for the room:

> Echo detected — the assistant heard its own output. If this repeats, lower
> `tts.normalize.target_lufs` or raise `echo.mute_tail_ms`.

Repeated `replay_exhausted` is the loud version of the same message: the room is
echoing badly enough that the reply cannot get through its own bleed.

---

## 6. Config reference

| Key | Value | Purpose |
|---|---|---|
| `echo.guard` | `true` | Enable text-level echo detection. Required by `barge_in.enabled` |
| `echo.threshold` | `0.6` | Word-overlap fraction to call it echo. Lower = more aggressive (may swallow real speech that quotes the assistant); higher = more echo slips through |
| `echo.window` | `6` | Recent replies compared against |
| `echo.mute_capture_while_replying` | `true` | Drop mic frames while speaking. Ignored when `barge_in.enabled` |
| `echo.mute_tail_ms` | `400` | Hold the mic closed past the last sample, for speaker/room ring |
| `tts.normalize.target_lufs` | `-23.0` | Output loudness; lower = less bleed |

Barge-in's own keys (`barge_in.*`, `capture.vad.barge_in_*`) are documented in
[bargeIn.md](bargeIn.md).

---

## 7. Tuning guide

| Symptom | Fix |
|---|---|
| Assistant replies to itself | Confirm `echo.guard: true`; lower `echo.threshold` toward `0.5` |
| Assistant cuts itself off | Raise `capture.vad.barge_in_energy_multiplier` and `barge_in_debounce_ms`; lower `tts.normalize.target_lufs` |
| Real interruptions ignored | Raise `echo.threshold` toward `0.75`; lower `barge_in_debounce_ms` |
| Barge-in feels sluggish | Lower `barge_in_debounce_ms` / `barge_in_grace_ms` (accepting more false interrupts, which are now recoverable) |
| Replies keep restarting | The room is echoing: lower `target_lufs`, raise `barge_in_energy_multiplier` |
| Nothing works on this desk | Use headphones, or `barge_in.enabled: false` + `mute_capture_while_replying: true` — the only airtight combination |

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
- [test_barge_in.py](../pipeline/tests/test_barge_in.py) — the interrupt state
  machine and its verdict paths, including echo reversing an interrupt.

```
venv\Scripts\python.exe pipeline\tests\test_echo_guard.py
venv\Scripts\python.exe pipeline\tests\test_echo_guard_integration.py
venv\Scripts\python.exe pipeline\tests\test_barge_in.py
```