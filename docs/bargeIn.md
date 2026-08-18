# Barge-In — Interrupting a Reply Without Interrupting Ourselves

How the pipeline lets the user cut a reply short while it is playing, without the
assistant's own voice doing the same thing to it.

- Code: [interrupt.py](../pipeline/realtime/interrupt.py),
  [capture.py](../pipeline/realtime/capture.py),
  [audio_out.py](../pipeline/realtime/audio_out.py),
  [workers.py](../pipeline/realtime/workers.py),
  [run_realtime.py](../pipeline/run_realtime.py)
- Config: [realtime.yaml](../pipeline/config/realtime.yaml) (`barge_in.*`,
  `capture.vad.barge_in_*`)
- Prerequisite: `echo.guard` — see [echoReduction.md](echoReduction.md)

---

## 1. The ordering problem

Two requirements pull in opposite directions:

| Requirement | Deadline |
|---|---|
| Stopping the reply must feel like an interruption | **~250 ms** |
| Knowing whether the sound *was the user* needs the transcript | **~1–2 s** |

The second deadline is not an implementation shortcoming, it is structural.
The VAD decides on frame energy and voicedness, and **nothing in the raw
waveform separates "your voice" from "our voice coming back"** — with speakers
open, both are voiced audio arriving at the same mic. The first point in the
pipeline where the two are separable at all is the *transcript*: if the mic heard
"I can help with that" while the assistant was saying "I can help with that",
that is bleed, not a human.

But the transcript does not exist until the VAD closes the utterance
(`capture.vad.silence_ms`, 600 ms of trailing silence) and STT has run on it.
Waiting for it means 1.5–3 s of the assistant talking over you — by which point
short replies have finished on their own and there was nothing left to interrupt.

So neither "decide fast" nor "decide correctly" is achievable alone.

---

## 2. Two tiers — audio stops immediately and stays stopped

The decision is split, but **only the utterance's fate is deferred**. The audio
is gone the moment the interrupt fires:

```text
  user starts talking over the reply
            │
            │  ~240ms   strict VAD gate + debounce clears
            ▼
     ┌──────────────┐
     │   TIER 1     │  interrupt.claim()
     │   stop       │  playback stops mid-file — FINAL, never resumed
     └──────┬───────┘  the reply is dropped either way
            │
            │  speakers go silent → deafen window discards the ring
            │  utterance closes (silence_ms) → STT runs
            │  ~1-2s
            ▼
     ┌──────────────┐
     │   TIER 2     │  echo guard rules on the transcript.
     │   verdict    │  Decides what happens to the UTTERANCE,
     └──┬────────┬──┘  not to the audio.
        │        │
   not echo    echo
        │        │
        ▼        ▼
    TO MODEL   DISCARD
  reply_q +    bleed never
  wav_q        reaches the
  flushed      model
```

Two consequences worth being explicit about:

**The reply never comes back.** No replay, no resume. This is what guarantees the
assistant cannot re-trigger its own interrupt: a replayed reply would bleed again,
stop again, and replay again. Stopping once and staying stopped removes that loop
entirely.

**The acoustic gate carries the full weight.** Because a false interrupt
truncates a reply for good, layer 1 — the strict gate in §3 — is the only thing
preventing the assistant from cutting itself off. Tuning it too loose costs
truncated replies, and `barge_in_rejected` in the log is the symptom.

The STT "pre-check" is therefore real, but it is a **verdict**, not a *gate*. It
cannot run before the interrupt, because it cannot run before the transcript, and
the transcript cannot exist until the user stops talking.

---

## 3. Tier 1 — the acoustic claim

In [capture.py](../pipeline/realtime/capture.py). While the assistant is
audible the mic **stays live** (this is the difference from
`echo.mute_capture_while_replying`, which drops the frames), but VAD is held to a
stricter standard:

| Key | Default | Effect |
|---|---|---|
| `capture.vad.barge_in_energy_multiplier` | `2.5` | Energy must exceed `2.5 ×` the calibrated threshold |
| `capture.vad.barge_in_aggressiveness` | `3` | Strictest WebRTC setting, playback only |
| `capture.vad.barge_in_debounce_ms` | `240` | Speech must be *consecutive* for this long |
| `capture.vad.barge_in_grace_ms` | `350` | Ignore speech starts this soon after audio begins |

The rationale for each: the user's voice at the mic is far louder than a reply
that crossed the room (multiplier); bleed arrives in bursts shaped by the reply's
own syllables, so a sustained run is much harder for it to clear than a single
loud frame (debounce); and a reply's opening is its loudest, most bleed-prone
moment (grace).

Both VAD backends implement `is_speech(frame, strict=False)`, so the gate works
whether the `energy` or `webrtc` backend is selected.

The utterance that claimed the interrupt is **recorded from its first phoneme** —
the mic was never muted — so nothing the user said is lost, and it carries
`barge_in=True` through the queues so tier 2 knows it owes a verdict on it.

---

## 4. Tier 2 — the transcript verdict

In `STTStage` ([workers.py](../pipeline/realtime/workers.py)). Once the
utterance is transcribed, one of three things happens:

| Transcript | Verdict | Result |
|---|---|---|
| Echoes `RecentSpeech` | `reject` | Reply played again from the start |
| Real speech | `confirm` | Reply dropped; `reply_q` + `wav_q` flushed |
| Empty / STT failed / too short / queue-dropped | `abandon` | Reply played again from the start |

**Inconclusive defaults to resuming.** No words means no evidence either way, and
a duck with nothing behind it is more likely bleed than a real turn — so the
reply is restored rather than discarded.

**Every exit path resolves the claim.** An unresolved one would leave the
interrupted reply silent forever, so `_resolve` is called on the STT-crash,
STT-failed and empty-transcript paths too, and capture calls `abandon` when the
utterance is below `min_utterance_ms` or falls off a full `audio_queue`.

### Why the flush is limited to `reply_q` and `wav_q`

On a confirmed barge-in, anything in those two queues answers a question the user
has moved on from; playing it after they have started a new turn is worse than
dropping it. `transcript_queue` is deliberately left alone — an item there is a
user turn that has *not been answered yet*, not a stale reply.

---

## 5. Interruptible playback

`sd.play` + `sd.wait` cannot be interrupted: `sd.wait` blocks until the whole
buffer has drained, so barge-in would take effect only after the reply had
finished saying itself.

[audio_out.py](../pipeline/realtime/audio_out.py) instead writes the file to an
open `OutputStream` in 1024-frame blocks (~23 ms at 44.1 kHz), polling a stop
callback between them, and calls `stream.abort()` on a stop so the driver's
buffered audio is discarded rather than played out. `play()` returns
`(seconds, completed)`; `completed=False` is how `PlaybackStage` distinguishes a
barge-in from a reply that simply ended.

Block size sets the interrupt resolution, and at ~23 ms it never dominates
barge-in latency — the 240 ms debounce does.

---

## 6. The failure modes this has to survive

Each of these was a real bug or a real risk, and each has a named guard:

**A reply replaying forever.** A room where every reply reliably triggers a false
duck would replay the same sentence endlessly, bleeding again each time — the
runaway loop in a new shape. `PlaybackStage.MAX_REPLAYS = 2` caps it and emits
`replay_exhausted`, which is the signal that the acoustic layers are mis-tuned.

**A verdict that never arrives.** If tier 2 dies or its utterance vanishes,
playback would wait forever. `barge_in.verdict_timeout_ms` (6 s) expires and
resumes the reply. It must exceed `silence_ms` + worst-case STT latency, or real
verdicts get pre-empted by the timeout.

**Losing the claim in the handoff window.** Playback breaks out of its write loop
and *then* releases the reply — capture can claim in between. Releasing `playing`
there would make `claim()` refuse, silently dropping the interrupt and abandoning
the reply instead of judging it. So `end_playback()` deliberately does **not**
release a reply while a verdict is outstanding; `clear()` does that once playback
is finished with it however it ended.

**Two interrupts over one reply.** A stuttered start could open a second
arbitration while the first is unjudged. `claim()` refuses while `pending` is
set, and `note_capture()` binds exactly one utterance id, so verdicts from any
other utterance are rejected.

**Barge-in without a verdict source.** Tier 1 ducks on acoustics alone; the echo
guard is the only thing that can tell it the duck was our own reply. With
`echo.guard` off, every bleed burst would abort the reply and flush the pipeline.
`run_realtime.py` refuses to start on that combination.

---

## 7. Config reference

```yaml
barge_in:
  enabled: true
  verdict_timeout_ms: 6000   # > capture.vad.silence_ms + worst-case STT

capture:
  vad:
    barge_in_aggressiveness: 3
    barge_in_energy_multiplier: 2.5
    barge_in_debounce_ms: 240
    barge_in_grace_ms: 350

echo:
  guard: true                # required by barge_in.enabled
  threshold: 0.6             # the verdict's sensitivity
```

Setting `barge_in.enabled: false` restores the previous behaviour exactly: the
mic is muted during playback (`echo.mute_capture_while_replying`), nothing can
interrupt, and no interrupt controller is constructed.

---

## 8. Tuning guide

| Symptom | Fix |
|---|---|
| Reply cuts itself off, then restarts | Bleed is clearing tier 1. Raise `barge_in_energy_multiplier` / `barge_in_debounce_ms`, lower `tts.normalize.target_lufs` |
| `replay_exhausted` in the log | Same cause, worse. The room needs headphones or a directional mic |
| Interrupting takes too long | Lower `barge_in_debounce_ms` (false ducks are recoverable, so this is cheaper than it looks) |
| Cannot interrupt the first second | Lower `barge_in_grace_ms` |
| Interrupt works, then the old reply still plays | The flush is not running — check that `on_flush` is wired in `run_realtime.py` |
| Reply goes silent and never resumes | A claim is unresolved. Check for a tier-2 path returning without `_resolve`; `verdict_timeout_ms` is the backstop |

---

## 9. Tests

[test_barge_in.py](../pipeline/tests/test_barge_in.py) — no audio device, GPU or
model needed:

```
venv\Scripts\python.exe pipeline\tests\test_barge_in.py
```

Covers the `InterruptController` state machine (claim gating, verdict binding,
the confirm/reject/abandon paths, and the handoff window), `STTStage` as tier 2
(echo reverses, real speech confirms and flushes, inconclusive resumes,
unrelated utterances leave the claim alone), and `PlaybackStage` against a fake
player (plays once uninterrupted, aborts on confirm, replays on reject, honours
the replay cap, and resumes on verdict timeout).
