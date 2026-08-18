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
    """audio_queue -> transcript_queue"""

    def __init__(self, worker, spill_dir, *args, recent_speech=None,
                 echo_threshold=0.6, on_echo=None):
        super().__init__("stt", *args)
        self.worker = worker
        self.spill_dir = Path(spill_dir)
        self.recent_speech = recent_speech
        self.echo_threshold = echo_threshold
        self.on_echo = on_echo or (lambda *a, **k: None)

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
                res = self.worker.run({"utt_id": utt.utt_id, "pcm_path": str(pcm_path)})
            except Exception as exc:
                self.on_event("stage_error", stage="stt", error=str(exc))
                break
            finally:
                pcm_path.unlink(missing_ok=True)

            if not res.get("ok"):
                self.on_event("stt_failed", utt_id=utt.utt_id, error=res.get("error"))
                continue

            text = (res.get("text") or "").strip()
            if not text:
                # Non-speech. Dropping here keeps the LLM from being asked to
                # reason about silence.
                self.on_event("stt_empty", utt_id=utt.utt_id)
                continue

            # Layer 5: the assistant hearing itself. Dropped here so the model
            # never sees it as a user turn — otherwise it answers itself and
            # the loop runs away.
            if self.recent_speech is not None and self.recent_speech.is_echo(
                text, self.echo_threshold
            ):
                self.on_event("echo_dropped", utt_id=utt.utt_id, text=text)
                self.on_echo(utt.utt_id, text)
                continue

            self.on_event("stt", utt_id=utt.utt_id, text=text,
                          elapsed_s=res.get("elapsed_s"))
            self._put(Sentence(
                utt_id=utt.utt_id, text=text,
                t_captured=utt.t_captured, t_stt_done=time.perf_counter(),
            ))


class LLMStage(_Stage):
    """transcript_queue -> reply_queue"""

    def __init__(self, worker, *args):
        super().__init__("llm", *args)
        self.worker = worker

    def run(self):
        while not self.stop_event.is_set():
            sent = self._get()
            if sent is None:
                break

            try:
                res = self.worker.run({"utt_id": sent.utt_id, "text": sent.text})
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

            self.on_event("llm", utt_id=sent.utt_id, text=text,
                          lang=res.get("lang"), elapsed_s=res.get("elapsed_s"))
            self._put(Reply(
                utt_id=sent.utt_id, text=text, prompt=sent.text,
                lang=res.get("lang"),
                t_captured=sent.t_captured, t_stt_done=sent.t_stt_done,
                t_llm_done=time.perf_counter(),
            ))


class TTSStage(_Stage):
    """reply_queue -> wav_queue"""

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
                # No Piper voice for this language is a known gap, not a
                # failure — surface it as its own event so the reply can still
                # be shown as text instead of looking like a crash.
                if res.get("error") == "no_voice":
                    self.on_event("tts_no_voice", utt_id=reply.utt_id,
                                  lang=res.get("lang"), text=reply.text)
                else:
                    self.on_event("tts_failed", utt_id=reply.utt_id,
                                  error=res.get("error"))
                continue

            self.on_event("tts", utt_id=reply.utt_id,
                          elapsed_s=res.get("elapsed_s"), audio_s=res.get("audio_s"))
            self._put(WavJob(
                utt_id=reply.utt_id, wav_path=Path(res["wav_path"]),
                sample_rate=int(res["sample_rate"]), text=reply.text,
                t_captured=reply.t_captured, t_stt_done=reply.t_stt_done,
                t_llm_done=reply.t_llm_done, t_tts_done=time.perf_counter(),
            ))


class PlaybackStage(_Stage):
    """wav_queue -> speaker (terminal stage)"""

    def __init__(self, player, keep_wavs, *args, speaking_event=None,
                 mute_tail_s=0.0):
        super().__init__("playback", *args)
        self.player = player
        self.keep_wavs = keep_wavs
        self.speaking_event = speaking_event
        # Speakers and room reverb keep ringing briefly after the file ends;
        # unmuting exactly at the last sample lets the tail back in.
        self.mute_tail_s = float(mute_tail_s)

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

            if self.speaking_event is not None:
                self.speaking_event.set()
            try:
                self.player.play(job.wav_path)
            except Exception as exc:
                self.on_event("playback_failed", utt_id=job.utt_id, error=str(exc))
            finally:
                if self.speaking_event is not None:
                    if self.mute_tail_s:
                        time.sleep(self.mute_tail_s)
                    self.speaking_event.clear()
                if not self.keep_wavs:
                    job.wav_path.unlink(missing_ok=True)
