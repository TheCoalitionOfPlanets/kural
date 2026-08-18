"""Barge-in arbitration between capture, STT and playback.

The problem this solves is an ordering one. Interrupting has to happen within a
couple of hundred milliseconds or it is not an interruption; deciding *whether
the sound was the user or our own speakers* is only possible from the
transcript, which does not exist until the VAD closes the utterance
(`silence_ms` later) and STT has run — 1.5-3s in. Nothing in the raw waveform
separates "your voice" from "our voice coming back".

So the decision is split in two and the fast half is made reversible:

    tier 1  capture sees sustained speech during playback -> `provisional()`
            playback stops immediately. Fast, and wrong sometimes.

    tier 2  STT + echo guard return a verdict on that utterance:
              confirm()  real user  -> stay stopped, flush the pipeline
              reject()   own echo   -> replay the reply from the start

Tier 1 being reversible is what makes its false positives survivable, and it is
also what lets the acoustic gate be tuned aggressively: the cost of a wrong duck
is a restarted sentence, not a swallowed reply.

One utterance at a time holds the interrupt. `claim()` returns False for a
second speech burst arriving before the first has been judged, so a stuttered
start cannot open two arbitrations over one reply.
"""
import threading


class InterruptController:
    """Shared, thread-safe barge-in state.

    Owned by `run_realtime`; read and written by the capture thread (tier 1),
    the STT stage (tier 2) and the playback stage (which acts on both).
    """

    def __init__(self, on_event=None):
        self._lock = threading.Lock()
        self.on_event = on_event or (lambda *a, **k: None)

        # Set while a wav is being written to the device.
        self.playing = threading.Event()
        # Set by tier 1 to make the playback loop break out of its write loop.
        self.stopped = threading.Event()
        # Set by tier 2 (reject) to tell playback the file it just abandoned
        # should be played again from the beginning.
        self.replay = threading.Event()
        # Set by tier 2 (confirm) to tell playback to give the file up and to
        # tell the flusher to clear downstream queues.
        self.aborted = threading.Event()
        # Raised while a verdict is outstanding, so playback waits for tier 2
        # instead of racing ahead to the next queued wav.
        self.pending = threading.Event()

        # utt_id of the utterance that triggered the outstanding interrupt, and
        # of the reply it interrupted. Both are logging/tracing aids.
        self._claim_utt = None
        self._interrupted_utt = None

    # -- tier 1: capture ---------------------------------------------------

    def claim(self):
        """Try to open an interrupt. True if this caller now owns it.

        Called from the capture thread the moment sustained speech is seen
        during playback. Refuses when nothing is playing (there is nothing to
        interrupt) or when a verdict on an earlier burst is still outstanding.
        """
        with self._lock:
            if not self.playing.is_set() or self.pending.is_set():
                return False
            self.pending.set()
            self.stopped.set()
            self.replay.clear()
            self.aborted.clear()
            self._claim_utt = None
            self._interrupted_utt = None
        self.on_event("barge_in_provisional")
        return True

    def note_capture(self, utt_id):
        """Attach the closed utterance's id to the outstanding interrupt.

        The claim happens at speech *start*; the utterance only gets an id when
        the VAD closes it. Without this, tier 2 cannot tell whether the
        transcript it is judging is the one that caused the interrupt or an
        unrelated utterance that arrived afterwards.
        """
        with self._lock:
            if self.pending.is_set() and self._claim_utt is None:
                self._claim_utt = utt_id
                return True
            return False

    # -- tier 2: STT verdict -----------------------------------------------

    def owns(self, utt_id):
        """True if `utt_id` is the utterance the outstanding interrupt is waiting on."""
        with self._lock:
            return self.pending.is_set() and self._claim_utt == utt_id

    def confirm(self, utt_id):
        """Real user speech. Playback stays dead and the pipeline is flushed."""
        with self._lock:
            if not (self.pending.is_set() and self._claim_utt == utt_id):
                return False
            self.aborted.set()
            self.replay.clear()
            self.pending.clear()
        self.on_event("barge_in_confirmed", utt_id=utt_id)
        return True

    def reject(self, utt_id, reason="echo"):
        """Own output, not the user. The interrupted reply is played again."""
        with self._lock:
            if not (self.pending.is_set() and self._claim_utt == utt_id):
                return False
            self.replay.set()
            self.aborted.clear()
            self.pending.clear()
        self.on_event("barge_in_rejected", utt_id=utt_id, reason=reason)
        return True

    def abandon(self, reason):
        """Give up on an interrupt that will never get a verdict.

        An utterance can vanish between tier 1 and tier 2 — below
        `min_utterance_ms`, transcribed empty, dropped by a full queue, or STT
        failed. Without this the reply would sit unresumed forever, so the
        default is to resume it: a duck with no evidence behind it is treated
        as the false positive it probably was.
        """
        with self._lock:
            if not self.pending.is_set():
                return False
            claimed = self._claim_utt
            self.replay.set()
            self.aborted.clear()
            self.pending.clear()
        self.on_event("barge_in_abandoned", utt_id=claimed, reason=reason)
        return True

    # -- playback ----------------------------------------------------------

    def begin_playback(self, utt_id):
        with self._lock:
            self.stopped.clear()
            self.replay.clear()
            self.aborted.clear()
            self._interrupted_utt = utt_id
        self.playing.set()

    def end_playback(self):
        self.playing.clear()
        with self._lock:
            self._interrupted_utt = None

    def clear(self):
        """Drop all interrupt state. Used on shutdown and after a flush."""
        with self._lock:
            self.pending.clear()
            self.stopped.clear()
            self.replay.clear()
            self.aborted.clear()
            self._claim_utt = None
            self._interrupted_utt = None
