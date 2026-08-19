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
        # See _WebRtcVAD: a stricter bar while the speakers are active, so most
        # bleed is rejected before it can claim an interrupt.
        self.barge_in_gate = float(cfg.get("barge_in_energy_multiplier", 2.5))

    def calibrate(self, frames):
        levels = [self._rms(f) for f in frames]
        floor = float(np.median(levels)) if levels else self.min_threshold
        self.threshold = max(floor * self.margin, self.min_threshold)
        return floor, self.threshold

    @staticmethod
    def _rms(frame):
        return float(np.sqrt(np.mean(np.square(frame))) + 1e-12)

    def is_speech(self, frame, strict=False):
        floor = self.threshold * (self.barge_in_gate if strict else 1.0)
        return self._rms(frame) > floor

    def level(self, frame):
        return self._rms(frame)


class _WebRtcVAD:
    """WebRTC VAD, gated by an energy floor.

    WebRTC classifies *voicedness*, not loudness, so on its own it happily fires
    on faint room tone and fan noise. Requiring both a positive VAD decision and
    energy above the calibrated floor rejects that without dulling real speech.

    Requires 10/20/30ms frames at 8/16/32/48kHz.

    Two aggressiveness settings are kept: the normal one, and a stricter one used
    only while the speakers are active. Bleed is voiced audio — the VAD is right
    to call it speech — so the only acoustic lever during playback is demanding
    more of it. `barge_in_aggressiveness` and the louder gate below are the
    acoustic half of the interrupt decision; the text guard is the other half.
    """

    def __init__(self, cfg, sample_rate, frame_ms):
        import webrtcvad  # optional dependency

        if frame_ms not in (10, 20, 30):
            raise ValueError(f"webrtc VAD needs frame_ms in 10/20/30, got {frame_ms}")
        self.vad = webrtcvad.Vad(int(cfg.get("aggressiveness", 2)))
        strict = int(cfg.get("barge_in_aggressiveness",
                             min(3, int(cfg.get("aggressiveness", 2)) + 1)))
        self._strict_vad = webrtcvad.Vad(strict) if strict != int(
            cfg.get("aggressiveness", 2)) else self.vad
        self.sample_rate = sample_rate
        self.gate = _EnergyVAD(cfg)
        # Multiplier on the energy threshold while the assistant is audible.
        # The user's own voice at the mic is far louder than a reply that has
        # crossed the room, so raising the bar rejects most bleed outright.
        self.barge_in_gate = float(cfg.get("barge_in_energy_multiplier", 2.5))

    def calibrate(self, frames):
        return self.gate.calibrate(frames)

    def is_speech(self, frame, strict=False):
        level = self.gate.level(frame)
        floor = self.gate.threshold * (self.barge_in_gate if strict else 1.0)
        if level <= floor:
            return False
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        vad = self._strict_vad if strict else self.vad
        return vad.is_speech(pcm16, self.sample_rate)

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
                 speaking_event=None, interrupt=None):
        super().__init__(name="capture", daemon=True)
        self.cfg = cfg
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.on_status = on_status or (lambda *a, **k: None)
        # When set, the assistant is talking through the speakers. With
        # `interrupt` present the mic stays live and this only selects the
        # stricter VAD gate; without it, frames are discarded outright (the old
        # airtight-but-uninterruptible behaviour).
        self.speaking_event = speaking_event
        # Barge-in arbitration. None disables interruption entirely.
        self.interrupt = interrupt

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
        # Consecutive speech frames required to claim an interrupt. Bleed
        # arrives in bursts shaped by the reply's own syllables, so a sustained
        # run is much harder for it to clear than a single loud frame; real
        # speech clears it easily. This is the main barge-in tuning lever.
        self.barge_in_frames = max(
            1, int(v.get("barge_in_debounce_ms", 240)) // self.frame_ms)
        # Ignore speech starts for this long after playback begins. The opening
        # of a reply is its loudest, most bleed-prone moment.
        self.barge_in_grace_s = float(v.get("barge_in_grace_ms", 350)) / 1000.0
        # Frames are discarded for this long after the speakers go silent.
        # Stopping the write loop does not stop the room: speakers and reverb
        # ring on past the last sample, and that ringing is exactly what must
        # not be recorded as the start of the user's turn.
        self.deafen_s = float(v.get("post_playback_deafen_ms", 250)) / 1000.0

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

    def _emit(self, frames, t_captured, barged=False):
        pcm = np.concatenate(frames)
        duration = len(pcm) / self.sample_rate
        utt = Utterance(
            utt_id=f"u{next(self._counter):04d}",
            pcm=pcm,
            sample_rate=self.sample_rate,
            duration_s=duration,
            t_captured=t_captured,
            barge_in=barged,
        )
        # An utterance that claimed an interrupt must be the one tier 2 judges,
        # so the claim is bound to its id before it is queued — otherwise the
        # STT stage cannot tell it apart from an unrelated utterance.
        if barged and self.interrupt is not None:
            self.interrupt.note_capture(utt.utt_id)
        # Q0 policy: drop oldest rather than block the capture path.
        try:
            self.audio_queue.put_nowait(utt)
        except queue.Full:
            try:
                stale = self.audio_queue.get_nowait()
                self.on_status("dropped", utt_id=stale.utt_id)
                if stale.barge_in and self.interrupt is not None:
                    # The utterance holding the interrupt just fell off the
                    # queue, so no verdict is ever coming for it.
                    self.interrupt.abandon("utterance_dropped")
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
            # Consecutive strict-gate speech frames seen during playback, and
            # whether the utterance being recorded is the one that interrupted.
            barge_run = 0
            barged = False
            playback_since = None
            # Deadline until which post-playback ring is discarded.
            deafen_until = 0.0
            was_audible = False

            while not self.stop_event.is_set():
                frame = self._next_frame()
                if frame is None:
                    continue

                assistant_audible = (self.speaking_event is not None
                                     and self.speaking_event.is_set())

                # Without an interrupt controller, fall back to the old layer-2
                # behaviour: drop frames while the assistant is audible.
                # Airtight against self-hearing, but nothing can interrupt.
                # `mute_tail_ms` already covers the ring on this path, so the
                # deafen window below is not applied here.
                if assistant_audible and self.interrupt is None:
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

                # The moment the speakers go silent, open a short deafen window:
                # playback stopping does not stop the room, and that ring must
                # not be recorded as the start of a turn.
                if self.interrupt is not None:
                    if was_audible and not assistant_audible:
                        deafen_until = time.monotonic() + self.deafen_s
                    was_audible = assistant_audible

                    # Never while an utterance is already being recorded:
                    # mid-utterance that audio is the user talking (they are why
                    # playback stopped), and cutting it here would clip the very
                    # turn this mechanism exists to capture.
                    if (not assistant_audible and not speaking
                            and time.monotonic() < deafen_until):
                        preroll.clear()
                        continue

                # The mic stays live during playback, so the utterance that
                # interrupts is recorded from its first phoneme. The stricter
                # gate applies only while the speakers are actually audible.
                if assistant_audible:
                    if playback_since is None:
                        playback_since = time.monotonic()
                else:
                    playback_since = None
                    barge_run = 0

                is_speech = self.vad.is_speech(frame, strict=assistant_audible)

                # Tier 1: sustained speech over the strict gate stops playback.
                # Provisional — the STT stage reverses it if the transcript
                # turns out to be our own reply coming back.
                if not (assistant_audible and is_speech):
                    # The run must be *consecutive*; bleed rarely sustains. Reset
                    # on silence AND whenever the speakers go quiet, so a run
                    # built up against one reply cannot carry over and interrupt
                    # the next one on its first frame.
                    barge_run = 0
                elif not barged:
                    in_grace = (time.monotonic() - playback_since
                                < self.barge_in_grace_s)
                    barge_run += 1
                    if not in_grace and barge_run >= self.barge_in_frames:
                        if self.interrupt.claim():
                            barged = True
                            self.on_status("barge_in")

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
                        utt = self._emit(frames, time.perf_counter(), barged=barged)
                        self.on_status(
                            "utterance", utt_id=utt.utt_id,
                            duration=utt.duration_s, forced=forced,
                            barge_in=barged,
                        )
                    else:
                        self.on_status("too_short")
                        if barged and self.interrupt is not None:
                            # Too short to transcribe, so tier 2 will never
                            # rule on it. Treat the duck as a false positive
                            # and let the reply resume.
                            self.interrupt.abandon("too_short")
                    barged = False
                    barge_run = 0
                    silence_run = 0

        if self._dropped_frames:
            self.on_status("frame_drops", count=self._dropped_frames)
