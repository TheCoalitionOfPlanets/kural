# Web Interface — Moving the Microphone into a Browser

How the same pipeline that owns a local microphone and a local speaker ends up
driven from a browser tab, without any of the tuned parts knowing.

- Code: [session.py](../pipeline/realtime/session.py),
  [capture.py](../pipeline/realtime/capture.py),
  [web_player.py](../pipeline/realtime/web_player.py),
  [server/app.py](../pipeline/server/app.py), [web/](../web)
- Config: [realtime.yaml](../pipeline/config/realtime.yaml) (`server.*`)
- Tests: [test_server.py](../pipeline/tests/test_server.py),
  [web/e2e/voice.mjs](../web/e2e/voice.mjs)

---

## 1. What actually had to change

Almost nothing, which was the point.

The pipeline was written around `sounddevice` at both ends. But nothing in the
tuned middle depends on that: the VAD endpointer depends on *frames arriving*,
and barge-in depends on *knowing when the assistant is audible*. Neither cares
where the frames came from or which speaker the audio reached.

So both ends became pluggable and the middle was left alone:

| | terminal | web |
|---|---|---|
| frames in | `MicSource` (sounddevice) | `StreamSource` (WebSocket) |
| audio out | `Player` (sounddevice) | `WebPlayer` (a browser) |
| everything else | identical | identical |

`Player`'s interface is two methods — `play(wav_path, should_stop) -> (seconds,
completed)` and `stop()` — which is narrow enough that the speakers can move to
the far end of a socket without `PlaybackStage`, the barge-in tiers or the echo
guard noticing. `completed=False` still means "the user cut this off", and it
still arrives mid-sentence rather than after.

### 1.1 Session, so there is one graph and not two

The wiring — four stage threads, a capture thread, one echo guard shared
between four places, the barge-in validation, the flush-on-confirm, the
shutdown order — is subtle, and every way it can drift is silent. A stage that
does not get the echo guard. A queue flushed in the wrong order. A PortAudio
stream never closed.

So it lives in `Session` once, and the two entrypoints differ only in what they
plug into the ends. `run_realtime.py` is now config → `Session(Player)` → run.

---

## 2. Frames are re-chunked, not trusted

An `AudioWorkletProcessor` is handed 128 samples at a time, at whatever rate
the `AudioContext` is really running — which is neither 16 kHz nor a divisor of
a 20 ms frame at any rate anyone uses.

That matters more than it sounds. `webrtcvad` accepts **only** exact 10/20/30 ms
frames, and the energy gate's threshold is calibrated against a fixed frame
length. Feed it ragged buffers and it does not fail — it silently gets worse.

So the conversion happens twice, on purpose:

```
128 @ 48 kHz ──▶ [worklet] linear resample to 16 kHz ──▶ 320-sample frames ──▶ socket
                                                                                │
                          [StreamSource] re-chunk to exactly frame_samples ◀────┘
```

The browser side does it so the audio thread owns it — layout jank on the main
thread can then never drop microphone data. The server side does it again
because the socket is not the only thing that could ever push frames, and a
source that hands the VAD a 193-sample buffer should be impossible by
construction rather than by convention.

The resampler carries its read position and its last sample across `process()`
calls. Restarting the phase every block would put a click at every 128 samples —
audible, and worse, loud enough to trip the VAD.

---

## 3. Who owns the clock during playback

Locally, the write loop *is* the playback: stopping is immediate and the number
of seconds played is exact. In a browser, the audio is handed over and the
browser reports back, so `WebPlayer` is a state machine over those reports:

```
── audio (JSON meta, then bytes) ──▶  decode, start an AudioBufferSourceNode
◀─ playback_started ──                 speakers are live
── stop_audio ──▶                      (only if should_stop() fires)
◀─ playback_finished / playback_stopped ──
```

`stop()` fires `onended` exactly like a natural finish does, so the browser
tells the two apart by who asked. Reporting a barge-in as a natural finish would
let the server believe a reply was heard in full when it was cut off two words
in.

**Every wait has a deadline.** A backgrounded tab, a blocked autoplay policy or
a socket that dies mid-reply must not park the playback stage forever. The reply
is lost either way; a wedged pipeline is much worse than a dropped sentence. So
a browser that never reports having started gets `playback_never_started` and
the turn is released.

---

## 4. Models load once; sessions are cheap

The three subprocesses take minutes and hold ~7 GB of VRAM, so they are started
with the server and shared. A browser connecting builds only queues, four stage
threads and a capture thread — milliseconds — and they are torn down when the
tab closes.

**One browser at a time.** There is one GPU and one conversation history; a
second connection is refused with a reason rather than silently interleaved into
the first one's turns.

The noise floor is calibrated per connection, from the connecting browser's own
first second of frames. That is the right behaviour and not a compromise: it
measures *that* room and *that* microphone rather than whatever the server
machine can hear.

### 4.1 The bug this shape hides

Teardown originally sat in the WebSocket handler's `finally` and awaited. A
client disconnecting *cancels* that task, and a cancelled task resumes its
`finally` only as far as the first `await` — everything after it is skipped.

`conn.stop()` ran. `hub.release(conn)` never did. The pipeline stayed locked to
a browser that had already closed its tab, and every later connection was told
the server was busy. It presented as "works once, then permanently busy".

Nothing in that `finally` may await now. Teardown runs on its own thread and
releases the hub when it is genuinely finished; a connection arriving in the
meantime is told the pipeline is busy, which is the truth — the previous
session's threads are still winding down.

---

## 5. Echo, revisited

The browser's own acoustic echo cancellation is far better than anything the
server can do, because it has the reference signal: it knows exactly what was
sent to the speakers and can subtract it. `echoCancellation: true` does more for
the self-hearing problem than every server-side layer combined.

Those layers all stay. [echoReduction.md](echoReduction.md) is unchanged and
still applies — the loudness normalization, the mute-while-replying gate, the
text guard. They are now a second line behind a much better first one, which is
the right order rather than a redundancy.

---

## 6. Events

The server forwards its whole event stream as JSON, so the UI shows what the
pipeline is actually doing rather than a guess:

| event | in the UI |
|---|---|
| `calibrated`, `listening` | orb breathing, "Listening" |
| `speech_start` | orb tracks the microphone |
| `utterance` | orb sweeps, "Thinking" |
| `stt` | the user's turn, with its language and — only when it was not Set A — a `whisper` tag |
| `llm` | the reply |
| `audio` + bytes | decoded and played; orb tracks the output |
| `barge_in` | one ripple outward |
| `tts_no_voice` | the reply, tagged "text only — no voice" |
| `stt_no_international` | a warning naming the missing API key |
| `latency` | per-stage timings above the controls |

`level` fires per frame — 50/s — and is thinned to ~20/s before it reaches the
socket. The orb cannot use more than a handful, and each one is a frame on the
wire.

The one tag that costs money is the only one that gets the accent colour. That
is deliberate: an international turn is two paid API calls, and the transcript
line is where you notice them.

---

## Related docs

- [pipline.md](pipline.md) — the pipeline this wraps
- [internationalLanguages.md](internationalLanguages.md) — what the language tags mean
- [echoReduction.md](echoReduction.md) — why the browser's AEC changes the ordering
- [bargeIn.md](bargeIn.md) — the two tiers, unchanged by any of this
- [web/README.md](../web/README.md) — running and designing the front end
