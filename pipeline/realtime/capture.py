"""Mic capture and VAD endpointing.

The sounddevice callback does nothing but append to a deque — running VAD or
any I/O inside it causes overruns and dropped audio. A separate thread pops
frames and drives the endpointing state machine.
"""
import collections
import itertools
import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from .messages import Utterance


class _EnergyVAD:
    """RMS threshold with an auto-calibrated noise floor.

    Cheap and dependency-free. Struggles with variable background noise —
    swap in _WebRtcVAD if false triggers become a problem.
    """

    def __init__(self, cfg):
        self.margin = float(cfg.get("noise_margin", 3.0))
        self.min_threshold = float(cfg.get("min_threshold", 0.003))
        self.threshold = self.min_threshold

    def calibrate(self, frames):
        levels = [self._rms(f) for f in frames]
        floor = float(np.median(levels)) if levels else self.min_threshold
        self.threshold = max(floor * self.margin, self.min_threshold)
        return floor, self.threshold

    @staticmethod
    def _rms(frame):
        return float(np.sqrt(np.mean(np.square(frame))) + 1e-12)

    def is_speech(self, frame):
        return self._rms(frame) > self.threshold

    def level(self, frame):
        return self._rms(frame)


class _WebRtcVAD:
    """WebRTC VAD, gated by an energy floor.

    WebRTC classifies *voicedness*, not loudness, so on its own it happily fires
    on faint room tone and fan noise. Requiring both a positive VAD decision and
    energy above the calibrated floor rejects that without dulling real speech.

    Requires 10/20/30ms frames at 8/16/32/48kHz.
    """

    def __init__(self, cfg, sample_rate, frame_ms):
        import webrtcvad  # optional dependency

        if frame_ms not in (10, 20, 30):
            raise ValueError(f"webrtc VAD needs frame_ms in 10/20/30, got {frame_ms}")
        self.vad = webrtcvad.Vad(int(cfg.get("aggressiveness", 2)))
        self.sample_rate = sample_rate
        self.gate = _EnergyVAD(cfg)

    def calibrate(self, frames):
        return self.gate.calibrate(frames)

    def is_speech(self, frame):
        if not self.gate.is_speech(frame):
            return False
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return self.vad.is_speech(pcm16, self.sample_rate)

    def level(self, frame):
        return self.gate.level(frame)


def make_vad(cfg, sample_rate, frame_ms):
    backend = cfg.get("backend", "energy")
    if backend == "webrtc":
        return _WebRtcVAD(cfg, sample_rate, frame_ms)
    if backend == "energy":
        return _EnergyVAD(cfg)
    raise ValueError(f"unknown VAD backend: {backend}")


class CaptureThread(threading.Thread):
    """Mic -> VAD -> audio_queue.

    Implements the endpointing state machine from the spec:
        IDLE     --speech--------------------> SPEAKING
        SPEAKING --silence for silence_ms----> emit, IDLE
        SPEAKING --duration >= max_utt_ms----> force emit, IDLE
    """

    def __init__(self, cfg, audio_queue, stop_event, on_status=None,
                 speaking_event=None):
        super().__init__(name="capture", daemon=True)
        self.cfg = cfg
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.on_status = on_status or (lambda *a, **k: None)
        # When set, the assistant is talking through the speakers. Frames are
        # discarded rather than fed to the VAD so bleed is never recorded.
        self.speaking_event = speaking_event

        self.sample_rate = int(cfg["sample_rate"])
        self.frame_ms = int(cfg["frame_ms"])
        self.frame_samples = self.sample_rate * self.frame_ms // 1000

        v = cfg["vad"]
        self.vad = make_vad(v, self.sample_rate, self.frame_ms)
        self.calibration_s = float(v.get("calibration_s", 1.0))
        self.silence_frames = int(v["silence_ms"]) // self.frame_ms
        self.min_frames = int(v["min_utterance_ms"]) // self.frame_ms
        self.max_frames = int(v["max_utterance_ms"]) // self.frame_ms
        self.preroll_frames = int(v["pre_roll_ms"]) // self.frame_ms

        self._frames = queue.Queue(maxsize=256)
        self._counter = itertools.count(1)
        self._dropped_frames = 0

    # -- audio device ------------------------------------------------------

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.on_status("audio", str(status))
        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # Ring buffer overflow: the VAD thread is not keeping up. Dropping
            # here is still better than blocking the device callback.
            self._dropped_frames += 1

    def _open_stream(self):
        device = self.cfg.get("device")
        if device in (None, "default"):
            device = None
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=device,
            callback=self._callback,
        )

    # -- endpointing -------------------------------------------------------

    def _next_frame(self, timeout=0.5):
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def _calibrate(self):
        deadline = time.time() + self.calibration_s
        frames = []
        while time.time() < deadline and not self.stop_event.is_set():
            f = self._next_frame()
            if f is not None:
                frames.append(f)
        floor, threshold = self.vad.calibrate(frames)
        self.on_status("calibrated", floor=floor, threshold=threshold)

    def _emit(self, frames, t_captured):
        pcm = np.concatenate(frames)
        duration = len(pcm) / self.sample_rate
        utt = Utterance(
            utt_id=f"u{next(self._counter):04d}",
            pcm=pcm,
            sample_rate=self.sample_rate,
            duration_s=duration,
            t_captured=t_captured,
        )
        # Q0 policy: drop oldest rather than block the capture path.
        try:
            self.audio_queue.put_nowait(utt)
        except queue.Full:
            try:
                stale = self.audio_queue.get_nowait()
                self.on_status("dropped", utt_id=stale.utt_id)
                self.audio_queue.put_nowait(utt)
            except queue.Empty:
                pass
        return utt

    def run(self):
        with self._open_stream():
            self._calibrate()
            self.on_status("listening")

            preroll = collections.deque(maxlen=self.preroll_frames)
            utterance = []
            speaking = False
            silence_run = 0

            muted = False

            while not self.stop_event.is_set():
                frame = self._next_frame()
                if frame is None:
                    continue

                # Layer 2: drop frames entirely while the assistant is audible.
                # Airtight against self-hearing, at the cost of not being able
                # to interrupt mid-reply.
                if self.speaking_event is not None and self.speaking_event.is_set():
                    if not muted:
                        muted = True
                        self.on_status("muted")
                        # Discard any partial utterance rather than stitching
                        # pre-playback audio onto post-playback audio.
                        utterance = []
                        speaking = False
                        silence_run = 0
                    preroll.clear()
                    continue

                if muted:
                    muted = False
                    self.on_status("unmuted")

                is_speech = self.vad.is_speech(frame)

                if not speaking:
                    preroll.append(frame)
                    if is_speech:
                        speaking = True
                        silence_run = 0
                        # Pre-roll matters: without it the leading phoneme is
                        # clipped and STT hears "ello" instead of "hello".
                        utterance = list(preroll)
                        preroll.clear()
                        self.on_status("speech_start")
                    else:
                        self.on_status("level", level=self.vad.level(frame))
                    continue

                utterance.append(frame)
                silence_run = 0 if is_speech else silence_run + 1

                closed = silence_run >= self.silence_frames
                forced = len(utterance) >= self.max_frames

                if closed or forced:
                    speaking = False
                    frames, utterance = utterance, []
                    preroll.clear()
                    if len(frames) >= self.min_frames:
                        utt = self._emit(frames, time.perf_counter())
                        self.on_status(
                            "utterance", utt_id=utt.utt_id,
                            duration=utt.duration_s, forced=forced,
                        )
                    else:
                        self.on_status("too_short")
                    silence_run = 0

        if self._dropped_frames:
            self.on_status("frame_drops", count=self._dropped_frames)
