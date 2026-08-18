# 05 — VAD Real-Time Speech-to-Speech Integration (PC)

Implementation spec for turning the current **one-shot** `pipeline/run_pc.py` into an **always-listening** English speech → Hindi speech translator.

This document is a build plan. Nothing here is implemented yet.

- Design reference: [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md)
- Target now: **PC (RTX 3050 4 GB)**. Jetson Nano remains draft.

---

## 1. Current state vs target

| Capability | Today (`pipeline/run_pc.py`) | Target (`pipeline/run_realtime.py`) |
|---|---|---|
| Input | one WAV file or `--text` | continuous mic stream |
| Segmentation | none (whole file) | VAD utterance endpointing |
| Concurrency | strictly serial | 5 workers + 4 queues |
| Model loading | per invocation | once at startup, resident |
| Playback | single file, then exit | serial drain, never overlapping |
| Runs until | one utterance | Ctrl+C / stop signal |

The existing stage code is reusable as-is. `run_pc.py` already proves STT → MT → TTS works; the real-time layer only changes **how audio arrives** and **how stages are scheduled**.

---

## 2. Target architecture

```text
Mic (16 kHz mono)
   │  20 ms frames, never blocks
   ▼
VAD endpointer ──► Q0 audio_queue ──► Whisper worker (CUDA, required)
                                            │
                                            ▼
                                     Q1 transcript_queue
                                            │
                                            ▼
                                    Marian worker (CUDA, GPU lock)
                                            │
                                            ▼
                                     Q2 translation_queue
                                            │
                                            ▼
                                     Piper worker (CPU)
                                            │
                                            ▼
                                        Q3 wav_queue
                                            │
                                            ▼
                              Playback worker (one WAV at a time)
```

Workers are threads in one process. They share nothing except queues and one `gpu_lock`.

---

## 3. Files to add

```text
pipeline/
├── run_realtime.py          # entrypoint: wires config → queues → workers → shutdown
├── config/
│   └── realtime.yaml        # all tunables (see §8)
└── realtime/
    ├── __init__.py
    ├── messages.py          # Utterance, Sentence, Translation, WavJob dataclasses
    ├── capture.py           # mic reader + VAD endpointer  → Q0
    ├── workers.py           # stt_worker, mt_worker, tts_worker, playback_worker
    └── audio_out.py         # blocking WAV playback + device selection
```

No existing file needs to change. `stt/src/runtime/` (`STTEngine`, `AudioChunk`, `Transcript`) is imported unmodified.

---

## 4. Message types

Every queue item carries `utt_id` so a single utterance can be traced end-to-end in logs and latency reports.

```python
@dataclass
class Utterance:
    utt_id: str
    pcm: bytes          # int16 mono
    sample_rate: int    # 16000
    duration_s: float
    t_captured: float   # perf_counter at VAD close

@dataclass
class Sentence:
    utt_id: str
    text: str
    src_lang: str       # "en"
    t_captured: float

@dataclass
class Translation:
    utt_id: str
    text: str
    tgt_lang: str       # "hi"
    t_captured: float

@dataclass
class WavJob:
    utt_id: str
    wav_path: Path
    sample_rate: int
    t_captured: float
```

`Utterance.pcm` maps directly onto the existing `AudioChunk(pcm=..., sample_rate=...)`, so the STT worker needs no conversion logic beyond the constructor call.

---

## 5. Capture and VAD

### 5.1 Audio source

Use `sounddevice.RawInputStream` with a callback that only appends bytes to a `collections.deque`. Never run VAD, inference, or file I/O inside the callback — a slow callback causes ALSA overruns and dropped audio.

- Format: `int16`, mono, 16 000 Hz
- Blocksize: 320 frames (20 ms) — matches WebRTC VAD's accepted frame sizes
- The callback pushes to a bounded ring buffer; a separate thread pops and runs VAD

### 5.2 Endpointing state machine

```text
IDLE ──speech in N_start consecutive frames──► SPEAKING
SPEAKING ──silence for silence_ms──► emit Utterance ──► IDLE
SPEAKING ──duration >= max_utterance_ms──► force emit ──► IDLE
```

Parameters that matter:

| Parameter | Suggested | Effect |
|---|---|---|
| `silence_ms` | 500 | Lower = snappier but chops mid-sentence pauses |
| `min_utterance_ms` | 300 | Below this, discard as noise/click |
| `max_utterance_ms` | 10000 | Hard cut so one long monologue can't stall the pipeline |
| `pre_roll_ms` | 200 | Prepend buffered audio before speech onset so the first phoneme is not clipped |
| `aggressiveness` | 2 | WebRTC VAD 0–3; raise in noisy rooms |

**Pre-roll is not optional.** Without it, WebRTC VAD reliably eats the leading consonant and Whisper transcribes "ello" instead of "hello". Keep a rolling buffer of the last `pre_roll_ms` and prepend it when transitioning to SPEAKING.

### 5.3 VAD backend choice

| Backend | Package | Notes |
|---|---|---|
| WebRTC (recommended start) | `webrtcvad` | ~zero CPU, C implementation, 10/20/30 ms frames only, 8/16/32/48 kHz only |
| Silero | `torch.hub` / `silero-vad` | Far better in noise, costs ~1–2 % CPU, run on **CPU** so it never contends with Whisper on the GPU |
| Energy threshold | none | Only as a debugging fallback; unreliable with variable mic gain |

Start with WebRTC. Swap to Silero only if false triggers from keyboard/fan noise become a problem — keep the endpointer's interface identical so the swap is a one-line change.

---

## 6. Workers

### 6.1 STT worker (CUDA only)

Reuses `WhisperPytorchSTT` with the existing hard-CUDA policy — `require_cuda=True`, `allow_cpu_fallback=False`. Load once before the loop starts, call `warmup()`, then block on `audio_queue.get()`.

An empty transcript means non-speech (the `Transcript.is_speech` flag already models this). Drop it silently and continue; do **not** push an empty sentence into Q1, or Piper will be asked to synthesize silence.

If CUDA is unavailable at startup, abort the whole service rather than degrade — this matches the locked device policy.

### 6.2 MT worker

Reuses the `load_mt` / `mt_translate` logic from `run_pc.py`, hoisted into the worker so the model loads once. Wrap the `generate` call in `gpu_lock`.

On `torch.cuda.OutOfMemoryError`, call `torch.cuda.empty_cache()` and retry once on the CT2 int8 CPU export (`translate/models/export/en-hi-v1-ct2-int8`). This fallback is allowed for MT and only for MT.

### 6.3 GPU lock

Whisper and Marian share one 4 GB card. Both models stay resident; only the inference calls are serialized:

```python
gpu_lock = threading.Lock()   # held ONLY around .transcribe() / .generate()
```

Hold the lock for the shortest possible span. Never hold it across a queue `put()` — that couples the stages back together and reintroduces the serial behaviour this design exists to remove.

Peak memory to expect: Whisper base fp16 ≈ 0.3 GB, Marian fp16 ≈ 0.3 GB, plus activations. Comfortable on 3050 4 GB provided nothing else holds VRAM.

### 6.4 TTS worker

Reuses `piper_synth` from `run_pc.py`. Writes to `pipeline/spill/{utt_id}.wav`. CPU-bound, so it genuinely overlaps with GPU work — this is where the pipeline's latency hiding comes from.

### 6.5 Playback worker

Pops Q3 and plays **blocking**, one job at a time. Delete the WAV after playback unless `keep_wavs` is set for debugging.

**Hard rule:** exactly one active playback. Overlapping speaker output makes the product unusable regardless of how good the translation is.

---

## 7. Backpressure and drop policy

Every queue is bounded. What happens when one fills is a product decision, not an implementation detail:

| Queue | maxsize | On full |
|---|---|---|
| Q0 `audio_queue` | 8 | **Drop oldest** + log. Never block the capture thread. |
| Q1 `transcript_queue` | 32 | Block STT worker (natural backpressure) |
| Q2 `translation_queue` | 32 | Block MT worker |
| Q3 `wav_queue` | 8 | Block TTS worker |

Q0 is the exception that matters. Blocking there would stall the mic callback and corrupt the audio stream, so old audio is dropped instead. Log every drop with `utt_id` — a steady stream of drops means the GPU cannot keep up and the model or VAD settings need to change.

---

## 8. Configuration

```yaml
capture:
  device: default
  sample_rate: 16000
  frame_ms: 20
  vad:
    backend: webrtc        # webrtc | silero | energy
    aggressiveness: 2
    silence_ms: 500
    min_utterance_ms: 300
    max_utterance_ms: 10000
    pre_roll_ms: 200

queues:
  audio_queue: { maxsize: 8, on_full: drop_oldest }
  transcript_queue: { maxsize: 32, on_full: block }
  translation_queue: { maxsize: 32, on_full: block }
  wav_queue: { maxsize: 8, on_full: block }

stt:
  model_dir: stt/models/export/en-hi-base-v1-fp16
  device: cuda
  require_cuda: true
  allow_cpu_fallback: false
  language: en
  beam_size: 1

translate:
  pair: en-hi
  model_dir: translate/models/export/en-hi-v1-fp16
  device: cuda
  allow_cpu_fallback: true
  fallback_model_dir: translate/models/export/en-hi-v1-ct2-int8
  max_new_tokens: 96
  num_beams: 2

tts:
  voice: tts/models/export/hi_official_v1/voice.onnx
  device: cpu

playback:
  blocking: true
  device: default
  keep_wavs: false

runtime:
  gpu_lock: true
  spill_dir: pipeline/spill
  log_latency: true
```

Language pair stays config-only. Switching to another pair must not require touching worker code — that is the loose-coupling guarantee from [01 — Architecture](./01-architecture.md).

---

## 9. New dependencies

```bash
uv pip install --python venv/bin/python sounddevice webrtcvad
```

`sounddevice` needs PortAudio present on the system. On Arch: `sudo pacman -S portaudio`.

Silero VAD, if adopted later, needs no extra package beyond `torch`, which is already installed.

---

## 10. Latency budget

Measure per stage using the `t_captured` timestamp carried on every message. Target for a ~3 s English utterance on the 3050:

| Segment | Target |
|---|---|
| VAD close → STT done | 300–700 ms |
| MT | 50–150 ms |
| Piper synth | 150–400 ms |
| Queue waits | < 100 ms when not saturated |
| **VAD close → first audio out** | **< 1.5 s** |

Log a single line per utterance with the stage breakdown. If total time regularly exceeds the utterance duration, the pipeline is falling behind and queues will saturate — the fix is a smaller Whisper model or shorter `max_utterance_ms`, not deeper queues.

---

## 11. Build order

Each step is independently testable. Do not skip ahead — a broken capture layer is very hard to diagnose once four workers are running on top of it.

1. **Capture + VAD only.** Write each detected utterance to `spill/{utt_id}.wav` and print duration. Verify segments align with your speech by listening to them.
2. **Add STT worker.** Print transcripts live. Confirm no CPU fallback and no audio drops.
3. **Add MT worker + GPU lock.** Print `en → hi` pairs. Watch VRAM with `nvidia-smi` under sustained speech.
4. **Add TTS + playback.** Full loop. Verify playback never overlaps.
5. **Add backpressure, latency logging, clean shutdown.**

Clean shutdown means: stop capture, push a sentinel through each queue in order, join workers with a timeout, then close the audio device. A `Ctrl+C` that leaves the PortAudio stream open will lock the mic until the process is killed.

---

## 12. Acceptance checklist

- [ ] Mic capture runs continuously; speaking during playback still gets captured
- [ ] VAD segments match spoken sentences; no clipped leading phonemes
- [ ] STT runs CUDA-only and aborts at startup if CUDA is missing
- [ ] Whisper and Marian share the GPU via lock without OOM under sustained speech
- [ ] Piper runs on CPU and overlaps with GPU work
- [ ] Playback is strictly serial, one utterance at a time
- [ ] Q0 drops are logged, never silent
- [ ] Per-utterance latency breakdown is logged
- [ ] Ctrl+C shuts down cleanly and releases the audio device
- [ ] Language pair changes require config edits only

---

## Related docs

- [01 — Architecture](./01-architecture.md)
- [02 — Models & fine-tuning](./02-models-and-finetuning.md)
- [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md)
- [pipeline/README.md](../pipeline/README.md)