"""Barge-in tests: the two-tier interrupt and its verdict paths.

Covers the InterruptController state machine, the STT stage as tier 2, and the
playback stage's stop/replay/abort behaviour with a fake player. No audio device,
GPU or model needed.

    venv\\Scripts\\python.exe pipeline\\tests\\test_barge_in.py
"""
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from pipeline.realtime.echo_guard import RecentSpeech  # noqa: E402
from pipeline.realtime.interrupt import InterruptController  # noqa: E402
from pipeline.realtime.messages import Utterance, WavJob  # noqa: E402
from pipeline.realtime.workers import PlaybackStage, STTStage  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


def wait_for(pred, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# -- InterruptController ---------------------------------------------------

print("InterruptController — claim gating")
ic = InterruptController()
check("cannot claim when nothing is playing", ic.claim() is False)

ic.begin_playback("r1")
check("claim succeeds during playback", ic.claim() is True)
check("claim stops playback", ic.stopped.is_set())
check("claim marks a verdict outstanding", ic.pending.is_set())
check("second claim refused while verdict outstanding", ic.claim() is False)

print("\nInterruptController — verdict binding")
check("no verdict before an utterance is bound", ic.owns("u1") is False)
check("note_capture binds the utterance", ic.note_capture("u1") is True)
check("bound utterance is owned", ic.owns("u1") is True)
check("other utterances are not owned", ic.owns("u2") is False)
check("a second bind is refused", ic.note_capture("u2") is False)
check("verdict from the wrong utterance is refused", ic.confirm("u2") is False)

print("\nInterruptController — confirm")
check("confirm accepted from the owner", ic.confirm("u1") is True)
check("confirm sets aborted", ic.aborted.is_set())
check("confirm does not set replay", not ic.replay.is_set())
check("confirm resolves the verdict", not ic.pending.is_set())

print("\nInterruptController — reject")
ic2 = InterruptController()
ic2.begin_playback("r2")
ic2.claim()
ic2.note_capture("u5")
check("reject accepted from the owner", ic2.reject("u5") is True)
check("reject sets replay", ic2.replay.is_set())
check("reject does not set aborted", not ic2.aborted.is_set())
check("reject resolves the verdict", not ic2.pending.is_set())

print("\nInterruptController — abandon resumes rather than aborts")
ic3 = InterruptController()
ic3.begin_playback("r3")
ic3.claim()
check("abandon accepted", ic3.abandon("too_short") is True)
check("abandon resumes the reply", ic3.replay.is_set())
check("abandon does not abort", not ic3.aborted.is_set())
check("abandon is a no-op with nothing pending", ic3.abandon("again") is False)

print("\nInterruptController — playback bookkeeping")
ic4 = InterruptController()
ic4.begin_playback("r4")
ic4.end_playback()
check("end_playback releases an unclaimed reply", not ic4.playing.is_set())
check("cannot claim after playback ended", ic4.claim() is False)

# The window between the write loop breaking and end_playback() running is real:
# capture can claim in it. Releasing `playing` there would drop the interrupt and
# abandon the reply instead of judging it.
ic4b = InterruptController()
ic4b.begin_playback("r4b")
ic4b.claim()
ic4b.end_playback()
check("end_playback holds the reply while a verdict is outstanding",
      ic4b.playing.is_set())
check("verdict still resolvable after end_playback", ic4b.note_capture("u1")
      and ic4b.reject("u1") is True)
ic4b.clear()
check("clear releases the reply", not ic4b.playing.is_set())


# -- STT stage as tier 2 ---------------------------------------------------

class FakeWorker:
    def __init__(self, reply_fn):
        self.reply_fn = reply_fn

    def run(self, payload):
        return self.reply_fn(payload)


def run_tier2(transcript, recent_speech, ok=True, utt_id="u1"):
    """Push one barge-in utterance through STTStage, return the interrupt state."""
    tmp = Path(tempfile.mkdtemp())
    in_q, out_q = queue.Queue(8), queue.Queue(8)
    stop = threading.Event()
    flushes = []

    ic = InterruptController()
    ic.begin_playback("r0")
    ic.claim()
    ic.note_capture(utt_id)

    worker = FakeWorker(lambda p: {"ok": ok, "utt_id": p["utt_id"],
                                   "text": transcript, "elapsed_s": 0.1,
                                   "error": None if ok else "boom"})
    stage = STTStage(worker, tmp, in_q, out_q, stop, lambda k, **kw: None,
                     recent_speech=recent_speech, echo_threshold=0.6,
                     interrupt=ic, on_flush=lambda u: flushes.append(u))
    stage.start()
    in_q.put(Utterance(utt_id, np.zeros(1600, dtype=np.float32), 16000, 0.1,
                       time.perf_counter(), barge_in=True))
    wait_for(lambda: not ic.pending.is_set())
    passed = []
    try:
        passed.append(out_q.get(timeout=0.3))
    except queue.Empty:
        pass
    stop.set()
    stage.join(timeout=2)
    return ic, passed, flushes


print("\nSTT tier 2 — echo reverses the interrupt")
window = RecentSpeech(6)
window.add("Paris is the capital of France. It is a beautiful and historic city.")
ic, passed, flushes = run_tier2("paris is the capital of france", window)
check("echo sets replay", ic.replay.is_set())
check("echo does not abort", not ic.aborted.is_set())
check("echo never reaches the model", passed == [])
check("echo does not flush the pipeline", flushes == [])

print("\nSTT tier 2 — real speech confirms the interrupt")
ic, passed, flushes = run_tier2("what about tomorrow morning instead", window)
check("real speech aborts playback", ic.aborted.is_set())
check("real speech does not replay", not ic.replay.is_set())
check("real speech reaches the model", len(passed) == 1)
check("real speech flushes stale replies", flushes == ["u1"])
check("barge_in flag carried to the transcript",
      passed and passed[0].barge_in is True)

print("\nSTT tier 2 — inconclusive verdicts resume the reply")
ic, passed, flushes = run_tier2("   ", window)
check("empty transcript resumes rather than aborts",
      ic.replay.is_set() and not ic.aborted.is_set())
check("empty transcript resolves the verdict", not ic.pending.is_set())

ic, passed, flushes = run_tier2("anything", window, ok=False)
check("failed STT resumes rather than aborts",
      ic.replay.is_set() and not ic.aborted.is_set())
check("failed STT resolves the verdict", not ic.pending.is_set())

print("\nSTT tier 2 — non-barge-in utterances leave the interrupt alone")
tmp = Path(tempfile.mkdtemp())
in_q, out_q = queue.Queue(8), queue.Queue(8)
stop = threading.Event()
ic5 = InterruptController()
ic5.begin_playback("r9")
ic5.claim()
ic5.note_capture("u_owner")
stage = STTStage(FakeWorker(lambda p: {"ok": True, "utt_id": p["utt_id"],
                                       "text": "hello there friend",
                                       "elapsed_s": 0.1}),
                 tmp, in_q, out_q, stop, lambda k, **kw: None,
                 recent_speech=window, interrupt=ic5)
stage.start()
in_q.put(Utterance("u_other", np.zeros(1600, dtype=np.float32), 16000, 0.1,
                   time.perf_counter(), barge_in=False))
wait_for(lambda: not out_q.empty())
stop.set()
stage.join(timeout=2)
check("unrelated utterance does not resolve the verdict", ic5.pending.is_set())
check("unrelated utterance still passes through", not out_q.empty())


# -- Playback stage --------------------------------------------------------

class FakePlayer:
    """Stands in for Player: honours should_stop, records attempts.

    `length_ticks` is how long a full reply takes; the judge thread has to claim
    within that window to interrupt it, exactly as a real speaker would.
    """

    def __init__(self, length_ticks=60):
        self.length_ticks = length_ticks
        self.attempts = 0

    def play(self, wav_path, should_stop=None):
        self.attempts += 1
        for _ in range(self.length_ticks):
            if should_stop is not None and should_stop():
                return 0.1, False
            time.sleep(0.01)
        return 1.0, True


def make_job(tmp, utt_id="r1"):
    p = Path(tmp) / f"{utt_id}.wav"
    p.write_bytes(b"")
    now = time.perf_counter()
    return WavJob(utt_id=utt_id, wav_path=p, sample_rate=16000, text="hi",
                  t_captured=now, t_stt_done=now, t_llm_done=now, t_tts_done=now)


def run_playback(verdicts, length_ticks=60, timeout_s=2.0, settle=1.2):
    """Play one job while a judge thread stands in for capture + STT.

    `verdicts` is one entry per interrupt to stage: "confirm", "reject", or None
    to claim and never rule (the verdict-timeout path). The judge claims as soon
    as a pass is audible, mirroring what the capture thread does on sustained
    speech.
    """
    tmp = Path(tempfile.mkdtemp())
    in_q = queue.Queue(8)
    stop = threading.Event()
    events = []
    # run_realtime routes controller events to the same sink as stage events, so
    # barge_in_* and playback_* land together here too.
    ic = InterruptController(on_event=lambda k, **kw: events.append((k, kw)))
    player = FakePlayer(length_ticks)

    stage = PlaybackStage(player, True, in_q, None, stop,
                          lambda k, **kw: events.append((k, kw)),
                          speaking_event=threading.Event(), interrupt=ic,
                          verdict_timeout_s=timeout_s)
    stage.start()
    in_q.put(make_job(tmp))

    def judge():
        for i, verdict in enumerate(verdicts):
            if not wait_for(lambda: ic.playing.is_set() and not ic.pending.is_set(),
                            3.0):
                return
            time.sleep(0.08)          # let the pass get going
            if not ic.claim():
                return
            utt = f"u{i}"
            ic.note_capture(utt)
            if verdict is None:
                return                # tier 2 never rules
            time.sleep(0.04)
            (ic.confirm if verdict == "confirm" else ic.reject)(utt)
            wait_for(lambda: not ic.pending.is_set(), 1.0)

    threading.Thread(target=judge, daemon=True).start()
    # Settle rather than racing a predicate: several of these cases are asserting
    # that nothing *further* happens.
    time.sleep(settle + timeout_s if None in verdicts else settle)
    stop.set()
    stage.join(timeout=3)
    return events, player


print("\nPlayback — uninterrupted reply plays once")
events, player = run_playback([], length_ticks=10)
check("a reply nobody interrupts is played exactly once", player.attempts == 1)
check("no abort reported", not any(k == "playback_aborted" for k, _ in events))

print("\nPlayback — confirmed barge-in abandons the reply")
events, player = run_playback(["confirm"])
check("confirmed barge-in does not replay", player.attempts == 1)
check("abort is reported", any(k == "playback_aborted" for k, _ in events))
check("no replay on a confirmed barge-in",
      not any(k == "playback_replay" for k, _ in events))

print("\nPlayback — rejected barge-in replays the reply")
events, player = run_playback(["reject"], length_ticks=30)
check("rejected barge-in replays", player.attempts == 2)
check("replay is reported", any(k == "playback_replay" for k, _ in events))
check("no abort on a rejected barge-in",
      not any(k == "playback_aborted" for k, _ in events))

print("\nPlayback — replay cap stops a runaway echo loop")
# Every pass is interrupted and every verdict says echo: without the cap this
# replays the same sentence forever, bleeding again on each attempt.
events, player = run_playback(["reject"] * 6, length_ticks=60, settle=2.5)
check("replay is capped", any(k == "replay_exhausted" for k, _ in events))
check("attempts bounded by MAX_REPLAYS",
      player.attempts <= PlaybackStage.MAX_REPLAYS + 1)

print("\nPlayback — a verdict that never arrives resumes the reply")
events, player = run_playback([None], length_ticks=30, timeout_s=0.3)
check("verdict timeout resumes rather than dropping", player.attempts >= 2)
check("timeout is reported as abandoned",
      any(k == "barge_in_abandoned" for k, _ in events))
check("timeout does not abort the reply",
      not any(k == "playback_aborted" for k, _ in events))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all barge-in tests passed")
