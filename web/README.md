# Kural — voice interface

A ChatGPT-voice-style front end for the pipeline in [`../pipeline`](../pipeline).
The browser is the microphone and the speakers; everything between them — VAD
endpointing, barge-in, the echo guard, the language gate — is the same Python
that the terminal pipeline runs.

```
browser mic ─ WS binary ─▶ StreamSource ─▶ [ the whole pipeline ] ─▶ WebPlayer ─ WS ─▶ browser
```

## Running it

Two processes. The server holds the models; this holds the UI.

```bash
# terminal 1 — the pipeline (root venv, ~7 GB VRAM, minutes to load)
venv/Scripts/python.exe -m pipeline.server

# terminal 2 — the UI
cd web
npm install
npm run dev            # http://localhost:3000
```

The start button reports what the server is actually doing — `Loading models…`
until the three subprocesses are up, then `Start talking`. It polls `/health`,
so a server that is still loading looks like a server that is still loading
rather than a button that fails on click.

Point it somewhere other than the default with `.env.local`:

```bash
NEXT_PUBLIC_PIPELINE_URL=http://127.0.0.1:8000
```

### Without a GPU

`pipeline/tests/stub_server.py` is the same server with the three models faked
out. It needs no weights and no GPU, and is the right way to work on the UI:

```bash
python3 ../pipeline/tests/stub_server.py    # port 8123
NEXT_PUBLIC_PIPELINE_URL=http://127.0.0.1:8123 npm run dev
```

## Microphone permission

`getUserMedia` needs a secure context: `localhost` counts, a bare LAN IP does
not. To use this from a phone on the same network, put the Next app behind
HTTPS or tunnel it — Chrome and Safari will otherwise refuse the microphone
with no visible error beyond a denied permission.

Echo cancellation, noise suppression and auto gain are all requested from the
browser. The browser's AEC has the reference signal and is far better at this
than anything the server can do after the fact; the server-side echo guard
stays as a second line rather than the first.

## What is where

```
web/
├── app/
│   ├── layout.tsx          theme applied before first paint
│   ├── page.tsx            the whole screen
│   └── globals.css         the two-colour palette and its blue elevation
├── components/
│   ├── Orb.tsx             every pipeline state, driven by rAF not by React
│   ├── Transcript.tsx      turns, with language and backend tags
│   ├── Controls.tsx        mic, end, transcript, theme
│   └── StatusRail.tsx      state, hint, notices, per-turn latency
├── lib/
│   ├── voice.ts            the socket and both AudioContexts
│   ├── useVoice.ts         the React surface over it
│   └── types.ts            the wire protocol
├── public/worklets/
│   └── capture-processor.js  resample to 16 kHz, emit exact 20 ms frames
└── e2e/                    a real browser against a real server
```

### Two AudioContexts, deliberately

`capture` runs at 16 kHz — the rate the server's VAD and both ASR models
expect. `output` runs at the device default, because replies come back at
24 kHz or 44.1 kHz and decoding them into a 16 kHz context would resample them
down to telephone quality on the way to the speakers.

### The worklet does the awkward part

A worklet is handed 128 samples at a time at whatever rate the context is
running, which is neither 16 kHz nor a divisor of a 20 ms frame. Both
conversions happen on the audio thread, before anything crosses to the main
thread — so layout jank can never drop microphone data, and the server always
receives exactly the frames its VAD was calibrated against.

## Design

Pure white and pure black, with blue doing all the work in between. Every sense
of depth is a blue-tinted shadow rather than a grey fill: a grey card would make
the white read as off-white and the black as charcoal, and the whole palette
would collapse into the usual soft neutrals.

On black, shadow cannot darken — so elevation becomes emitted light instead of
cast shadow, using the same three steps and the same blue.

The orb carries every state the pipeline has: idle is monochrome with no blue at
all, so "live" is unmistakable; thinking sweeps an arc; speaking gets the
strongest glow; a barge-in sends one ripple outward, which is the only feedback
that the interrupt registered, since the audio simply stops.

Amplitude is written straight to the DOM from a `requestAnimationFrame` loop.
Re-rendering a React tree sixty times a second for a number that only moves a
circle would be the entire cost of the page.

## Keyboard

| Key | Action |
|---|---|
| `Space` | mute / unmute |
| `Esc` | end the session |

## Tests

```bash
npx playwright install chromium   # once
node e2e/voice.mjs                # see e2e/README.md for the two servers it needs
```

It drives the real UI in a real browser with a WAV file as the microphone, and
is the only test that covers the audio path end to end. The Python side is
covered by `pipeline/tests/test_server.py`, which exercises the same socket
without a browser.
