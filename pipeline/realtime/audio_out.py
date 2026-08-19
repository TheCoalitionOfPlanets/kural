"""Interruptible WAV playback.

Hard rule from the spec: exactly one active playback. Overlapping speaker
output makes the product unusable no matter how good the reply is.

`sd.play` + `sd.wait` cannot be interrupted — `sd.wait` blocks until the whole
buffer has drained, so barge-in would take effect only after the reply had
already finished saying itself. Instead the file is written to an open
`OutputStream` in small blocks, checking a stop flag between them. Latency to
silence is therefore one block plus whatever the device has already buffered,
not the length of the reply.
"""
import threading

import sounddevice as sd
import soundfile as sf

# Write granularity. Also the interrupt resolution: playback can only stop on a
# block boundary. 1024 frames is ~23ms at 44.1kHz — well under the debounce that
# gates an interrupt, so it never dominates barge-in latency.
BLOCK_FRAMES = 1024


class Player:
    """One-at-a-time playback that can be stopped part-way through a file."""

    def __init__(self, device=None):
        if device in (None, "default"):
            device = None
        self.device = device
        self._lock = threading.Lock()

    def play(self, wav_path, should_stop=None):
        """Play one file, block-by-block. Serialized across callers.

        `should_stop` is polled between blocks; a truthy return abandons the
        rest of the file. Returns `(seconds_played, completed)` — `completed`
        is False when the file was cut short, which is how the caller
        distinguishes a barge-in from a reply that simply ended.
        """
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        played = 0
        completed = True
        with self._lock:
            # `latency="low"` keeps the device's own buffer short. A large
            # buffer would keep playing audio already handed to the driver
            # after the write loop has stopped feeding it.
            with sd.OutputStream(samplerate=sr, channels=data.shape[1],
                                 dtype="float32", device=self.device,
                                 blocksize=BLOCK_FRAMES, latency="low") as stream:
                for start in range(0, len(data), BLOCK_FRAMES):
                    if should_stop is not None and should_stop():
                        completed = False
                        break
                    block = data[start:start + BLOCK_FRAMES]
                    stream.write(block)
                    played += len(block)
                if not completed:
                    # Discard whatever the driver still holds, so the reply
                    # goes quiet now rather than a buffer-length later. On a
                    # normal finish the stream is left to drain instead, or the
                    # closing context manager clips the final syllable.
                    stream.abort()
        return played / sr, completed

    def stop(self):
        sd.stop()
