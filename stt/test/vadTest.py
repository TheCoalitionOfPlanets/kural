"""
Continuous mic -> VAD -> SraVaani STT loop.

Captures microphone audio in small frames, uses an energy-based VAD (with an
auto-calibrated noise floor) to detect speech vs silence, buffers each spoken
utterance, and once the user goes quiet for HANGOVER_MS it sends the buffered
audio straight to the GPU-resident model for transcription. Runs forever,
printing live status + each finalized transcript, until Ctrl+C.
"""
import os
import sys
import time
import queue

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320

CALIBRATION_SECONDS = 1.0
NOISE_MARGIN = 3.0        # speech threshold = noise_floor * NOISE_MARGIN
MIN_THRESHOLD = 0.003     # floor so a dead-silent room doesn't trigger on any tiny blip

HANGOVER_MS = 600         # silence needed to end an utterance
PREBUFFER_MS = 300        # audio kept before speech onset so words aren't clipped
MAX_UTTERANCE_S = 15.0    # force a flush even if the user keeps talking
MIN_UTTERANCE_S = 0.25    # ignore blips shorter than this (coughs/clicks)

HANGOVER_FRAMES = HANGOVER_MS // FRAME_MS
PREBUFFER_FRAMES = PREBUFFER_MS // FRAME_MS


def rms(frame):
    return float(np.sqrt(np.mean(np.square(frame))) + 1e-12)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Loading SraVaani...")
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    model._ensure_loaded()
    print("Model loaded. Running on:", device)
    return model


def calibrate_noise_floor(audio_q):
    print(f"Calibrating noise floor ({CALIBRATION_SECONDS:.1f}s) - stay quiet...")
    levels = []
    deadline = time.time() + CALIBRATION_SECONDS
    while time.time() < deadline:
        levels.append(rms(audio_q.get()))
    floor = float(np.median(levels)) if levels else MIN_THRESHOLD
    threshold = max(floor * NOISE_MARGIN, MIN_THRESHOLD)
    print(f"Noise floor={floor:.5f}  threshold={threshold:.5f}\n")
    return threshold


def level_meter(level, threshold, width=20):
    ratio = min(level / (threshold * 2.0), 1.0) if threshold > 0 else 0.0
    filled = int(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main():
    model = load_model()
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("audio status:", status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        threshold = calibrate_noise_floor(audio_q)
        print("Listening... (Ctrl+C to stop)\n")

        prebuffer = []
        utterance = []
        speaking = False
        silence_run = 0
        last_meter_print = 0.0

        while True:
            frame = audio_q.get()
            level = rms(frame)
            is_speech = level > threshold

            if not speaking:
                prebuffer.append(frame)
                if len(prebuffer) > PREBUFFER_FRAMES:
                    prebuffer.pop(0)

                if is_speech:
                    speaking = True
                    silence_run = 0
                    utterance = prebuffer + [frame]
                    print("\r" + " " * 60, end="")
                    print("\rSpeech detected, recording...", end="", flush=True)
                else:
                    now = time.time()
                    if now - last_meter_print > 0.1:
                        print(f"\rListening... {level_meter(level, threshold)}", end="", flush=True)
                        last_meter_print = now
                continue

            utterance.append(frame)
            duration_s = len(utterance) * FRAME_MS / 1000.0

            if is_speech:
                silence_run = 0
            else:
                silence_run += 1

            if silence_run >= HANGOVER_FRAMES or duration_s >= MAX_UTTERANCE_S:
                speaking = False
                silence_run = 0
                wav = np.concatenate(utterance)
                utterance = []
                prebuffer = []

                if len(wav) / SAMPLE_RATE < MIN_UTTERANCE_S:
                    print("\r" + " " * 60, end="")
                    continue

                print("\r" + " " * 60, end="")
                print("\rTranscribing...", end="", flush=True)
                with torch.no_grad():
                    hyps = model.transcribe([wav], return_hypotheses=True)
                text = hyps[0].text.strip()
                print("\r" + " " * 60)
                print(f">> {text or '(no speech recognized)'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
