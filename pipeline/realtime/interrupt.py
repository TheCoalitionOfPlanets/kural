"""Barge-in arbitration between capture, STT and playback.

The problem this solves is an ordering one. Interrupting has to happen within a
couple of hundred milliseconds or it is not an interruption; deciding *whether
the sound was the user or our own speakers* is only possible from the
transcript, which does not exist until the VAD closes the utterance
(`silence_ms` later) and STT has run — 1.5-3s in. Nothing in the raw waveform
separates "your voice" from "our voice coming back".

So the decision is split in two and the fast half is made reversible:

    tier 1  capture sees sustained speech during playback -> `claim()`
            playback stops immediately, and stays stopped.

    tier 2  STT + echo guard return a verdict on that utterance:
              confirm()  real user  -> it goes to the model, flush the pipeline
              reject()   own echo   -> it is discarded

Stopping is final. The reply is never resumed, so the assistant's own voice can
never re-trigger it and no reply can loop on its own bleed. What tier 2 decides
is the fate of the *utterance*, not of the audio.

The cost of that is a false interrupt truncating a reply for good, so the acoustic
gate in capture.py carries the weight: it must reject bleed before `claim()` is
ever reached.

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
        # Never cleared to bring audio back — an interrupt is final.
        self.stopped = threading.Event()
        # Set by tier 2 (reject): the utterance was our own echo, so it is
        # discarded rather than sent to the model. A verdict label only; the
        # audio does not come back either way.
        self.rejected = threading.Event()
        # Set by tier 2 (confirm) to tell the flusher to clear downstream queues.
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
            self.rejected.clear()
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
            self.rejected.clear()
            self.pending.clear()
        self.on_event("barge_in_confirmed", utt_id=utt_id)
        return True

    def reject(self, utt_id, reason="echo"):
        """Own output, not the user. The utterance is discarded.

        The audio does not come back — stopping is final either way. This only
        keeps the bleed from reaching the model as a user turn.
        """
        with self._lock:
            if not (self.pending.is_set() and self._claim_utt == utt_id):
                return False
            self.rejected.set()
            self.aborted.clear()
            self.pending.clear()
        self.on_event("barge_in_rejected", utt_id=utt_id, reason=reason)
        return True

    def abandon(self, reason):
        """Give up on an interrupt that will never get a verdict.

        An utterance can vanish between tier 1 and tier 2 — below
        `min_utterance_ms`, transcribed empty, dropped by a full queue, or STT
        failed. Without this, `pending` would stay raised forever and playback
        would wait on a verdict that is never coming.

        Treated as a rejection: with no transcript there is no evidence of a
        user turn, so nothing is sent to the model. The audio stays stopped
        regardless, as it does for every other verdict.
        """
        with self._lock:
            if not self.pending.is_set():
                return False
            claimed = self._claim_utt
            self.rejected.set()
            self.aborted.clear()
            self.pending.clear()
        self.on_event("barge_in_abandoned", utt_id=claimed, reason=reason)
        return True

    # -- playback ----------------------------------------------------------

    def begin_playback(self, utt_id):
        """Mark a reply as live. Called before playing a wav."""
        with self._lock:
            # `stopped` is *not* cleared while a verdict is outstanding: a claim
            # can land in the window between the previous write loop exiting and
            # this call, and clearing it there would let the next reply start
            # playing over the user who just interrupted.
            if not self.pending.is_set():
                self.stopped.clear()
            self.rejected.clear()
            self.aborted.clear()
            self._interrupted_utt = utt_id
        self.playing.set()

    def end_playback(self):
        """Release the reply.

        Does nothing while a verdict is outstanding. `playing` gates `claim()`,
        so clearing it here would drop an interrupt whose claim landed in the
        window between the write loop breaking and this call — the reply would
        be abandoned instead of judged.
        """
        with self._lock:
            if self.pending.is_set():
                return
            self._interrupted_utt = None
        self.playing.clear()

    def clear(self):
        """Drop all interrupt state and release the reply.

        Called when playback is finished with a reply however it ended, and on
        shutdown. This is what finally releases `playing`, which
        `end_playback()` holds while a verdict is outstanding.
        """
        with self._lock:
            self.pending.clear()
            self.stopped.clear()
            self.rejected.clear()
            self.aborted.clear()
            self._claim_utt = None
            self._interrupted_utt = None
        self.playing.clear()
