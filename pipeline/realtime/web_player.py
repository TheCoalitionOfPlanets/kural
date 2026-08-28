"""Playback through a browser, behind the local Player's interface.

`PlaybackStage` needs exactly two things from a player: `play(wav_path,
should_stop) -> (seconds, completed)` and `stop()`. That is a narrow enough
contract that the speakers can be moved to the other end of a WebSocket without
the stage, the barge-in tiers or the echo guard noticing — `completed=False`
still means "the user cut this off", and it still arrives while the reply is
mid-sentence rather than after it.

The difference is who owns the clock. Locally the write loop *is* the playback,
so stopping is immediate and `played` is exact. Here the audio is handed to the
browser and the browser reports back, so this class is a state machine over
those reports:

    ── audio ──▶  browser decodes, starts an AudioBufferSourceNode
    ◀─ playback_started ──   speakers are live
    ── stop_audio ──▶        (only if should_stop() fires)
    ◀─ playback_finished / playback_stopped ──

Every wait has a deadline. A backgrounded tab, a blocked autoplay policy or a
socket that dies mid-reply must not leave the playback stage parked forever —
the reply is lost either way, and a wedged pipeline is much worse than a
dropped sentence.
"""
import threading
import time
import wave
from pathlib import Path

# How long to wait for the browser to say it began. Decode plus scheduling is
# milliseconds; this is generous because the cost of being wrong is a lost
# reply, not a stall.
START_TIMEOUT_S = 5.0

# Extra time allowed past the audio's own duration before assuming the report
# was lost. Covers decode latency and a browser that throttles timers.
FINISH_SLACK_S = 3.0

# How long to wait for the browser to confirm it actually stopped. Barge-in is
# already final by this point, so this only keeps the reports tidy.
STOP_CONFIRM_S = 0.5


class WebPlayer:
    """One-at-a-time playback in a connected browser."""

    def __init__(self, send_json, send_bytes, on_notice=None,
                 start_timeout_s=START_TIMEOUT_S, finish_slack_s=FINISH_SLACK_S):
        self.send_json = send_json
        self.send_bytes = send_bytes
        self.on_notice = on_notice or (lambda *a, **k: None)
        self.start_timeout_s = float(start_timeout_s)
        self.finish_slack_s = float(finish_slack_s)
        self._lock = threading.Lock()
        self._current = None
        # Set when the client goes away, so a play() already in flight gives up
        # instead of waiting out both of its deadlines.
        self._gone = threading.Event()

    # -- reports from the browser -----------------------------------------

    def note(self, msg):
        """Route one client message. Called on the socket's thread."""
        kind = msg.get("type")
        if kind not in ("playback_started", "playback_finished",
                        "playback_stopped", "playback_failed"):
            return False
        with self._lock:
            state = self._current
            # A report for a reply that has already been resolved — a late
            # "finished" after a barge-in, say — must not touch the next one.
            if state is None or msg.get("utt_id") not in (None, state["utt_id"]):
                return True
        if kind == "playback_started":
            state["t0"] = time.monotonic()
            state["started"].set()
            return True
        if kind == "playback_failed":
            self.on_notice("playback_failed", error=msg.get("error"))
        state["completed"] = kind == "playback_finished"
        state["started"].set()      # release a play() still waiting to begin
        state["done"].set()
        return True

    def abandon(self):
        """The client is gone; release anything waiting on it."""
        self._gone.set()
        with self._lock:
            state = self._current
        if state is not None:
            state["started"].set()
            state["done"].set()

    def reattach(self):
        """A client connected again."""
        self._gone.clear()

    # -- the Player interface ---------------------------------------------

    def play(self, wav_path, should_stop=None):
        wav_path = Path(wav_path)
        try:
            data = wav_path.read_bytes()
            with wave.open(str(wav_path), "rb") as fh:
                rate = fh.getframerate()
                duration = fh.getnframes() / float(rate or 1)
        except Exception as exc:
            self.on_notice("playback_unreadable", error=repr(exc))
            return 0.0, True

        state = {
            "utt_id": wav_path.stem,
            "started": threading.Event(),
            "done": threading.Event(),
            "completed": False,
            "t0": None,
        }
        with self._lock:
            self._current = state

        try:
            self.send_json({
                "type": "audio",
                "utt_id": state["utt_id"],
                "sample_rate": rate,
                "duration_s": round(duration, 3),
                "bytes": len(data),
            })
            self.send_bytes(data)
        except Exception as exc:
            self.on_notice("playback_send_failed", error=repr(exc))
            self._clear(state)
            return 0.0, True

        if not state["started"].wait(self.start_timeout_s) or self._gone.is_set():
            # Autoplay blocked, tab discarded, or the socket died. Reporting
            # "completed" consumes the reply and lets the pipeline move on;
            # parking here would wedge every turn after it.
            if not self._gone.is_set():
                self.on_notice("playback_never_started", utt_id=state["utt_id"])
            self._clear(state)
            return 0.0, True

        t0 = state["t0"] or time.monotonic()
        deadline = t0 + duration + self.finish_slack_s
        while True:
            if state["done"].wait(0.02):
                break
            if self._gone.is_set():
                self._clear(state)
                return time.monotonic() - t0, True
            if should_stop is not None and should_stop():
                # Barge-in. The audio is already being cut in the browser by
                # the time this returns; the confirmation is only bookkeeping.
                self._send_stop(state["utt_id"])
                state["done"].wait(STOP_CONFIRM_S)
                self._clear(state)
                return time.monotonic() - t0, False
            if time.monotonic() > deadline:
                # The report was lost. The audio has had its full duration plus
                # slack, so treating it as finished is very likely the truth.
                self.on_notice("playback_report_lost", utt_id=state["utt_id"])
                break

        completed = state["completed"] or not state["done"].is_set()
        self._clear(state)
        return min(time.monotonic() - t0, duration), completed

    def stop(self):
        with self._lock:
            state = self._current
        if state is not None:
            self._send_stop(state["utt_id"])

    # -- internals ---------------------------------------------------------

    def _send_stop(self, utt_id):
        try:
            self.send_json({"type": "stop_audio", "utt_id": utt_id})
        except Exception:
            # The socket is gone, which already means nothing is playing.
            pass

    def _clear(self, state):
        with self._lock:
            if self._current is state:
                self._current = None
