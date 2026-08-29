# Browser end-to-end check

Drives the real UI in a real browser against the real server, with only the
three models faked. It is the only test that covers the audio path end to end —
the AudioWorklet's resampling, the binary frames on the socket, the VAD
endpointing on the far side, and reply audio being decoded and played back.

Chrome is given a WAV file as its microphone (`--use-file-for-fake-audio-capture`)
rather than its built-in beep generator, because the noise floor has to be
calibrated from real silence before a burst arrives — Chrome's beep pattern is
too short and too regular to close an utterance.

```bash
# once
npm install
npx playwright install chromium

# terminal 1 — the pipeline server, with the models stubbed out
python3 ../pipeline/tests/stub_server.py

# terminal 2 — the UI. A production build, not `next dev`: HMR's own socket
# adds noise and the point here is to test what ships.
NEXT_PUBLIC_PIPELINE_URL=http://127.0.0.1:8123 npm run build
npx next start -p 3100

# terminal 3
node e2e/voice.mjs
```

Nothing here needs a GPU or model weights.
