"""Pipeline worker threads.

Each thread owns one stage: pop its input queue, drive the corresponding model
subprocess, push to the next queue. Bounded queues provide backpressure — a
blocked put is the intended way for a slow stage to throttle its upstream.
"""
import threading
import time
from pathlib import Path

import numpy as np

from .messages import Reply, Sentence, WavJob


class _Stage(threading.Thread):
    def __init__(self, name, in_q, out_q, stop_event, on_event):
        super().__init__(name=name, daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.on_event = on_event

    def _get(self):
        """Pop with a timeout so the thread can observe stop_event."""
        import queue as _queue
        while not self.stop_event.is_set():
            try:
                return self.in_q.get(timeout=0.25)
            except _queue.Empty:
                continue
        return None

    def _put(self, item):
        """Push, but stay responsive to shutdown while the queue is full."""
        import queue as _queue
        while not self.stop_event.is_set():
            try:
                self.out_q.put(item, timeout=0.25)
                return True
            except _queue.Full:
                continue
        return False


class STTStage(_Stage):
    """audio_queue -> transcript_queue

    Also tier 2 of barge-in. When an utterance interrupted playback, this stage
    owns the verdict: the transcript is the first point in the pipeline where the
    user and the assistant's own voice are separable at all.

    The verdict does not restore the audio — an interrupt is final. It decides
    whether the utterance reaches the model (real turn) or is discarded (our own
    bleed). Every exit path below must still resolve the claim, or playback sits
    waiting on a verdict that never comes.
    """

    def __init__(self, worker, spill_dir, *args, recent_speech=None,
                 echo_threshold=0.6, on_echo=None, interrupt=None,
                 on_flush=None):
        super().__init__("stt", *args)
        self.worker = worker
        self.spill_dir = Path(spill_dir)
        self.recent_speech = recent_speech
        self.echo_threshold = echo_threshold
        self.on_echo = on_echo or (lambda *a, **k: None)
        self.interrupt = interrupt
        # Called after a confirmed barge-in to clear replies for turns the user
        # has already moved past.
        self.on_flush = on_flush or (lambda *a, **k: None)

    def _resolve(self, utt, verdict, reason=""):
        """Issue the tier-2 verdict for an utterance that claimed an interrupt."""
        if not (getattr(utt, "barge_in", False) and self.interrupt is not None):
            return
        if not self.interrupt.owns(utt.utt_id):
            return
        if verdict == "confirm":
            self.interrupt.confirm(utt.utt_id)
            self.on_flush(utt.utt_id)
        elif verdict == "reject":
            self.interrupt.reject(utt.utt_id, reason)
        else:
            self.interrupt.abandon(reason)

    def run(self):
        while not self.stop_event.is_set():
            utt = self._get()
            if utt is None:
                break

            # The child process reads audio from disk rather than through the
            # pipe — a few hundred KB of float32 per utterance would otherwise
            # need base64 framing through stdin.
            pcm_path = self.spill_dir / f"{utt.utt_id}.npy"
            np.save(pcm_path, utt.pcm)

            try:
                # The rate travels with the audio: the language-ID model and
                # SraVaani are both 16kHz-only, and a mismatch there is silent
                # rather than loud, so the worker checks rather than assumes.
                res = self.worker.run({
                    "utt_id": utt.utt_id,
                    "pcm_path": str(pcm_path),
                    "sample_rate": utt.sample_rate,
                })
            except Exception as exc:
                self.on_event("stage_error", stage="stt", error=str(exc))
                self._resolve(utt, "abandon", "stt_crashed")
                break
            finally:
                pcm_path.unlink(missing_ok=True)

            if not res.get("ok"):
                if res.get("error") == "no_international_stt":
                    # Heard, identified, and then nowhere to send it. Reported
                    # on its own because the fix is a config one — an API key —
                    # not something wrong with the audio or the model.
                    self.on_event("stt_no_international", utt_id=utt.utt_id,
                                  lang=res.get("lang"))
                else:
                    self.on_event("stt_failed", utt_id=utt.utt_id,
                                  error=res.get("error"))
                self._resolve(utt, "abandon", "stt_failed")
                continue

            text = (res.get("text") or "").strip()
            if not text:
                # Non-speech. Dropping here keeps the LLM from being asked to
                # reason about silence, and as an interrupt verdict it releases
                # playback from waiting.
                self.on_event("stt_empty", utt_id=utt.utt_id)
                self._resolve(utt, "abandon", "stt_empty")
                continue

            # Layer 5: the assistant hearing itself. Dropped here so the model
            # never sees it as a user turn — otherwise it answers itself and
            # the loop runs away.
            if self.recent_speech is not None and self.recent_speech.is_echo(
                text, self.echo_threshold
            ):
                self.on_event("echo_dropped", utt_id=utt.utt_id, text=text,
                              barge_in=getattr(utt, "barge_in", False))
                self.on_echo(utt.utt_id, text)
                # The interrupt was our own reply bleeding back. The audio is
                # already gone for good; this just stops the bleed from being
                # treated as a user turn.
                self._resolve(utt, "reject", "echo")
                continue

            # Real user speech over the assistant. The interrupt stands.
            self._resolve(utt, "confirm")

            self.on_event("stt", utt_id=utt.utt_id, text=text,
                          lang=res.get("lang"), backend=res.get("backend"),
                          confidence=res.get("confidence"),
                          elapsed_s=res.get("elapsed_s"))
            self._put(Sentence(
                utt_id=utt.utt_id, text=text,
                lang=res.get("lang"), backend=res.get("backend"),
                t_captured=utt.t_captured, t_stt_done=time.perf_counter(),
                barge_in=getattr(utt, "barge_in", False),
            ))


class LLMStage(_Stage):
    """transcript_queue -> reply_queue"""

    def __init__(self, worker, *args, recent_speech=None):
        super().__init__("llm", *args)
        self.worker = worker
        # Cached here, the moment the text exists, rather than only at synthesis.
        # TTS can lag far enough behind that bleed from a fast reply reaches the
        # STT check before TTSStage has recorded it, and an unrecorded reply is
        # invisible to the guard — exactly the hole the loop escapes through.
        self.recent_speech = recent_speech

    def run(self):
        while not self.stop_event.is_set():
            sent = self._get()
            if sent is None:
                break

            try:
                # Forwarded, not re-derived: STT identified this from the audio
                # (and on the international path from Scribe), which the
                # transcript alone cannot reproduce.
                res = self.worker.run({
                    "utt_id": sent.utt_id,
                    "text": sent.text,
                    "lang": sent.lang,
                })
            except Exception as exc:
                self.on_event("stage_error", stage="llm", error=str(exc))
                break

            if not res.get("ok"):
                self.on_event("llm_failed", utt_id=sent.utt_id, error=res.get("error"))
                continue

            text = (res.get("text") or "").strip()
            if not text:
                self.on_event("llm_empty", utt_id=sent.utt_id)
                continue

            if self.recent_speech is not None:
                self.recent_speech.add(text)

            lang = res.get("lang") or sent.lang
            self.on_event("llm", utt_id=sent.utt_id, text=text,
                          lang=lang, elapsed_s=res.get("elapsed_s"))
            self._put(Reply(
                utt_id=sent.utt_id, text=text, prompt=sent.text,
                lang=lang,
                t_captured=sent.t_captured, t_stt_done=sent.t_stt_done,
                t_llm_done=time.perf_counter(),
            ))


class TTSStage(_Stage):
    """reply_queue -> wav_queue

    One stage, two voices. Which one speaks is decided inside the worker from
    the reply's language, so nothing here has to know that ElevenLabs exists —
    a WAV comes back either way, and playback, barge-in and the echo guard
    treat an international reply exactly like a local one.
    """

    def __init__(self, worker, spill_dir, *args, recent_speech=None):
        super().__init__("tts", *args)
        self.worker = worker
        self.spill_dir = Path(spill_dir)
        self.recent_speech = recent_speech

    def run(self):
        while not self.stop_event.is_set():
            reply = self._get()
            if reply is None:
                break

            # Recorded at synthesis rather than generation: text that never
            # reaches synthesis never reaches the room, and would otherwise
            # poison the window with words nobody heard.
            if self.recent_speech is not None:
                self.recent_speech.add(reply.text)

            wav_path = self.spill_dir / f"{reply.utt_id}.wav"
            try:
                res = self.worker.run({
                    "utt_id": reply.utt_id,
                    "text": reply.text,
                    "lang": reply.lang,
                    "wav_path": str(wav_path),
                })
            except Exception as exc:
                self.on_event("stage_error", stage="tts", error=str(exc))
                break

            if not res.get("ok"):
                # A language neither voice speaks is a known gap, not a
                # failure — surface it as its own event so the reply can still
                # be shown as text instead of looking like a crash.
                if res.get("error") == "no_voice":
                    self.on_event("tts_no_voice", utt_id=reply.utt_id,
                                  lang=res.get("lang"), text=reply.text,
                                  reason=res.get("reason"))
                else:
                    self.on_event("tts_failed", utt_id=reply.utt_id,
                                  error=res.get("error"))
                continue

            self.on_event("tts", utt_id=reply.utt_id, backend=res.get("backend"),
                          elapsed_s=res.get("elapsed_s"), audio_s=res.get("audio_s"))
            self._put(WavJob(
                utt_id=reply.utt_id, wav_path=Path(res["wav_path"]),
                sample_rate=int(res["sample_rate"]), text=reply.text,
                t_captured=reply.t_captured, t_stt_done=reply.t_stt_done,
                t_llm_done=reply.t_llm_done, t_tts_done=time.perf_counter(),
            ))


class PlaybackStage(_Stage):
    """wav_queue -> speaker (terminal stage)

    An interrupt is final: tier 1 breaks the write loop out of the file and the
    reply is abandoned there. It is never replayed, so the assistant's own voice
    cannot re-trigger it and no reply can loop on its own bleed.

    Tier 2's verdict still runs, but it rules on the *user's utterance* — real
    speech goes to the model, echo is discarded — not on the audio.
    """

    def __init__(self, player, keep_wavs, *args, speaking_event=None,
                 mute_tail_s=0.0, interrupt=None, verdict_timeout_s=6.0,
                 recent_speech=None):
        super().__init__("playback", *args)
        self.player = player
        self.keep_wavs = keep_wavs
        self.speaking_event = speaking_event
        # Restamped here so the echo cache's TTL runs from when the room could
        # actually hear the reply, not from when synthesis began.
        self.recent_speech = recent_speech
        # Speakers and room reverb keep ringing briefly after the file ends;
        # unmuting exactly at the last sample lets the tail back in.
        self.mute_tail_s = float(mute_tail_s)
        self.interrupt = interrupt
        # Longest wait for a tier-2 verdict: the utterance still has to close
        # (silence_ms) and be transcribed. On expiry the reply resumes, since a
        # duck with no verdict behind it is more likely bleed than a real turn.
        self.verdict_timeout_s = float(verdict_timeout_s)

    def _should_stop(self):
        return (self.interrupt is not None and self.interrupt.stopped.is_set()) \
            or self.stop_event.is_set()

    def _await_verdict(self, utt_id):
        """Block until tier 2 rules on the interrupt.

        The verdict no longer decides whether the *audio* comes back — once
        stopped it stays stopped. It decides only what happens to the user's
        utterance: a real turn goes on to the model, an echo is discarded. So
        the return value is informational and playback ends either way.
        """
        deadline = time.monotonic() + self.verdict_timeout_s
        while not self.stop_event.is_set():
            if not self.interrupt.pending.is_set():
                return "echo" if self.interrupt.rejected.is_set() else "user"
            if time.monotonic() >= deadline:
                self.interrupt.abandon("verdict_timeout")
                return "unknown"
            time.sleep(0.02)
        return "unknown"

    def _play_once(self, job):
        """One pass over the file. Returns True when it played to the end."""
        if self.recent_speech is not None:
            self.recent_speech.touch(job.text)
        if self.speaking_event is not None:
            self.speaking_event.set()
        if self.interrupt is not None:
            self.interrupt.begin_playback(job.utt_id)
        try:
            _played, completed = self.player.play(job.wav_path, self._should_stop)
            return completed
        except Exception as exc:
            self.on_event("playback_failed", utt_id=job.utt_id, error=str(exc))
            return True  # nothing to resume; treat as finished
        finally:
            if self.interrupt is not None:
                self.interrupt.end_playback()
            if self.speaking_event is not None:
                # Cleared as soon as the audio stops, interrupt or not: it means
                # "the speakers are live", and during the verdict wait they are
                # not. Leaving it set would hold the strict barge-in gate over a
                # silent room and clip the user mid-sentence.
                #
                # The reverb tail is only paid on a natural finish. After an
                # interrupt the user is already talking, so sleeping here would
                # eat the start of their turn.
                if self.mute_tail_s and not self.interrupt_pending():
                    time.sleep(self.mute_tail_s)
                self.speaking_event.clear()

    def interrupt_pending(self):
        return self.interrupt is not None and self.interrupt.pending.is_set()

    def run(self):
        while not self.stop_event.is_set():
            job = self._get()
            if job is None:
                break

            t_first_audio = time.perf_counter()
            self.on_event(
                "latency",
                utt_id=job.utt_id,
                stt_ms=int((job.t_stt_done - job.t_captured) * 1000),
                llm_ms=int((job.t_llm_done - job.t_stt_done) * 1000),
                tts_ms=int((job.t_tts_done - job.t_llm_done) * 1000),
                total_ms=int((t_first_audio - job.t_captured) * 1000),
            )

            completed = self._play_once(job)

            if not completed and self.interrupt is not None:
                # Interrupted: the audio is already silent and stays that way.
                # The verdict still matters, but only for the user's utterance —
                # a real turn proceeds to the model, an echo is discarded — so
                # it is awaited rather than acted on here.
                verdict = self._await_verdict(job.utt_id)
                self.on_event("playback_aborted", utt_id=job.utt_id,
                              verdict=verdict)

            # The reply is finished with, however it ended. `clear()` also
            # releases `playing`, which end_playback() deliberately holds while
            # a verdict is outstanding.
            if self.interrupt is not None:
                self.interrupt.clear()
            if not self.keep_wavs:
                job.wav_path.unlink(missing_ok=True)
