"""Blocking WAV playback.

Hard rule from the spec: exactly one active playback. Overlapping speaker
output makes the product unusable no matter how good the reply is.
"""
import threading

import sounddevice as sd
import soundfile as sf


class Player:
    def __init__(self, device=None):
        if device in (None, "default"):
            device = None
        self.device = device
        self._lock = threading.Lock()

    def play(self, wav_path):
        """Play one file to completion. Serialized across callers."""
        data, sr = sf.read(str(wav_path), dtype="float32")
        with self._lock:
            sd.play(data, sr, device=self.device)
            sd.wait()
        return len(data) / sr

    def stop(self):
        sd.stop()
