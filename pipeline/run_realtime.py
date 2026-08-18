"""Always-listening speech -> reasoning -> speech pipeline.

    mic -> VAD -> [SraVaani STT] -> [Gemma 3 4B] -> [Piper TTS] -> speaker

Each model runs as a subprocess in its own venv (their transformers pins are
mutually incompatible); this process owns capture, the queues, scheduling and
playback.

    python pipeline/run_realtime.py
    python pipeline/run_realtime.py --config pipeline/config/realtime.yaml
    python pipeline/run_realtime.py --capture-only     # stage 1 of the build order
"""
import argparse
import queue
import signal
import sys
import threading
import time
from pathlib import Path

# Import the package whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.realtime.audio_out import Player
from pipeline.realtime.capture import CaptureThread
from pipeline.realtime.echo_guard import RecentSpeech
from pipeline.realtime.interrupt import InterruptController
from pipeline.realtime.proc import WorkerProcess
from pipeline.realtime.workers import LLMStage, PlaybackStage, STTStage, TTSStage

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "pipeline" / "config" / "realtime.yaml"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_config(path):
    try:
        import yaml
    except ImportError:
        print("PyYAML is required: pip install pyyaml", file=sys.stderr)
        raise SystemExit(1)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Console:
    """Single writer for all status output.

    The capture thread emits a live level meter on one rewritten line; every
    other event scrolls. Without a lock the two interleave into garbage.
    """

    def __init__(self, log_latency=True):
        self._lock = threading.Lock()
        self._meter_active = False
        self._last_meter = 0.0
        self.log_latency = log_latency

    def _clear_meter(self):
        if self._meter_active:
            sys.stdout.write("\r" + " " * 72 + "\r")
            self._meter_active = False

    def line(self, text):
        with self._lock:
            self._clear_meter()
            print(text, flush=True)

    def meter(self, level, threshold):
        now = time.time()
        if now - self._last_meter < 0.1:
            return
        self._last_meter = now
        width = 20
        ratio = min(level / (threshold * 2.0), 1.0) if threshold else 0.0
        bar = "#" * int(ratio * width) + "-" * (width - int(ratio * width))
        with self._lock:
            sys.stdout.write(f"\rlistening [{bar}]")
            sys.stdout.flush()
            self._meter_active = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--capture-only", action="store_true",
                    help="run VAD only; write utterances to spill/ instead of "
                         "starting the models")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    spill_dir = ROOT / runtime.get("spill_dir", "pipeline/spill")
    spill_dir.mkdir(parents=True, exist_ok=True)

    console = Console(log_latency=runtime.get("log_latency", True))
    stop_event = threading.Event()

    # -- status routing ----------------------------------------------------

    threshold_holder = {"value": 0.0}

    def on_capture(kind, *args, **kw):
        if kind == "level":
            console.meter(kw.get("level", 0.0), threshold_holder["value"])
        elif kind == "calibrated":
            threshold_holder["value"] = kw.get("threshold", 0.0)
            console.line(f"  noise floor={kw.get('floor', 0):.5f} "
                         f"threshold={kw.get('threshold', 0):.5f}")
        elif kind == "listening":
            console.line("\nListening... (Ctrl+C to stop)\n")
        elif kind == "speech_start":
            console.line("  speech detected...")
        elif kind == "utterance":
            note = " (max length)" if kw.get("forced") else ""
            console.line(f"[{kw['utt_id']}] captured {kw['duration']:.1f}s{note}")
        elif kind == "muted":
            console.line("  (mic muted while replying)")
        elif kind == "unmuted":
            console.line("  (mic live)")
        elif kind == "barge_in":
            console.line("  interrupted — stopping playback")
        elif kind == "dropped":
            console.line(f"  ! dropped {kw['utt_id']} — pipeline saturated")
        elif kind == "frame_drops":
            console.line(f"  ! {kw['count']} mic frames dropped")
        elif kind == "audio":
            console.line(f"  ! audio status: {args[0] if args else ''}")

    def on_stage(kind, **kw):
        utt = kw.get("utt_id", "")
        if kind == "stt":
            console.line(f"[{utt}] heard: {kw['text']}")
        elif kind == "llm":
            lang = kw.get("lang")
            tag = f" ({lang})" if lang else ""
            console.line(f"[{utt}] reply{tag}: {kw['text']}")
        elif kind == "tts":
            console.line(f"[{utt}] synth {kw.get('audio_s')}s "
                         f"in {kw.get('elapsed_s')}s")
        elif kind == "latency":
            if console.log_latency:
                console.line(
                    f"[{utt}] latency stt={kw['stt_ms']}ms llm={kw['llm_ms']}ms "
                    f"tts={kw['tts_ms']}ms total={kw['total_ms']}ms"
                )
        elif kind == "stt_empty":
            console.line(f"[{utt}] no speech recognized")
        elif kind == "echo_dropped":
            console.line(f"[{utt}] echo of own output, dropped: {kw['text']}")
        elif kind == "barge_in_confirmed":
            console.line(f"[{utt}] interrupted by you — reply dropped")
        elif kind == "barge_in_rejected":
            # The reply was cut off by its own bleed. It does not come back, so
            # this is a mis-tuning warning, not just an FYI.
            console.line(
                f"[{utt}] ! reply was cut off by its own echo ({kw.get('reason')}). "
                f"Lower tts.normalize.target_lufs, or raise "
                f"capture.vad.barge_in_energy_multiplier / barge_in_debounce_ms."
            )
        elif kind == "barge_in_abandoned":
            console.line(f"  interrupt had no transcript ({kw.get('reason')}) — "
                         f"nothing sent to the model")
        elif kind == "playback_aborted":
            console.line(f"[{utt}] playback stopped")
        elif kind == "barge_in_provisional":
            pass  # already reported by the capture-side "barge_in" event
        elif kind == "tts_no_voice":
            # Piper has no voice for this language, so the reply is text-only.
            # Print it rather than dropping it silently — the answer is still
            # correct, it just cannot be spoken.
            console.line(f"[{utt}] no {kw.get('lang')} voice, text only: "
                         f"{kw.get('text')}")
        elif kind.endswith("_failed") or kind == "stage_error":
            console.line(f"  ! {kind}: {kw.get('error')}")

    def on_worker_log(name, line):
        console.line(f"  [{name}] {line}")

    # -- queues ------------------------------------------------------------

    q = cfg.get("queues", {})
    audio_q = queue.Queue(maxsize=q.get("audio_queue", {}).get("maxsize", 8))
    transcript_q = queue.Queue(maxsize=q.get("transcript_queue", {}).get("maxsize", 32))
    reply_q = queue.Queue(maxsize=q.get("reply_queue", {}).get("maxsize", 32))
    wav_q = queue.Queue(maxsize=q.get("wav_queue", {}).get("maxsize", 8))

    # Echo reduction and barge-in are the same problem seen from two sides, so
    # they share state. `speaking_event` marks the assistant as audible; with
    # barge-in off that means "drop mic frames" (airtight, uninterruptible), and
    # with it on it means "apply the strict VAD gate" (interruptible, with the
    # text guard keeping bleed from reaching the model as a user turn).
    echo_cfg = cfg.get("echo", {})
    barge_cfg = cfg.get("barge_in", {})
    barge_in_enabled = bool(barge_cfg.get("enabled", False))

    recent_speech = RecentSpeech(
        echo_cfg.get("window", 6),
        ttl_s=float(echo_cfg.get("ttl_s", 30)),
        ngram=int(echo_cfg.get("ngram", 5)),
    ) if echo_cfg.get("guard", True) else None
    echo_threshold = float(echo_cfg.get("threshold", 0.6))
    mute_tail_s = float(echo_cfg.get("mute_tail_ms", 0)) / 1000.0

    if barge_in_enabled and recent_speech is None:
        # Tier 1 stops playback on acoustics alone; the text guard is the only
        # thing that can tell whether it was the user. Without it every bleed
        # burst that survives the strict gate reaches the model as a user turn
        # and flushes the pipeline — the self-reply loop, with extra steps.
        print("barge_in.enabled requires echo.guard — enable echo.guard or "
              "disable barge_in.", file=sys.stderr)
        raise SystemExit(1)

    # The mic must stay live during playback for anything to be interruptible,
    # so barge-in implies the speaking flag is a gate selector, not a mute.
    speaking_event = threading.Event() if (
        barge_in_enabled or echo_cfg.get("mute_capture_while_replying", True)
    ) else None

    interrupt = InterruptController(on_event=lambda k, **kw: on_stage(k, **kw)) \
        if barge_in_enabled else None

    capture = CaptureThread(cfg["capture"], audio_q, stop_event, on_capture,
                            speaking_event=speaking_event, interrupt=interrupt)

    # -- capture-only mode (build order step 1) ----------------------------

    if args.capture_only:
        import numpy as np
        import soundfile as sf

        print("=" * 60)
        print("CAPTURE + VAD ONLY — utterances written to", spill_dir)
        print("=" * 60)
        capture.start()
        try:
            while True:
                try:
                    utt = audio_q.get(timeout=0.25)
                except queue.Empty:
                    continue
                path = spill_dir / f"{utt.utt_id}.wav"
                sf.write(path, utt.pcm, utt.sample_rate)
                console.line(f"  -> {path.name} ({utt.duration_s:.1f}s)")
        except KeyboardInterrupt:
            console.line("\nStopping...")
        finally:
            stop_event.set()
            capture.join(timeout=2)
        return

    # -- model subprocesses ------------------------------------------------

    print("=" * 60)
    print("REAL-TIME SPEECH PIPELINE")
    print("=" * 60)

    timeout_s = int(runtime.get("startup_timeout_s", 300))
    workers = []

    def spawn(key, label):
        section = dict(cfg[key])
        # Config paths are repo-relative; the child resolves them against its
        # own cwd, so make them absolute here.
        for path_key in ("prompt_file", "voices_dir"):
            if section.get(path_key):
                section[path_key] = str(ROOT / section[path_key])
        wp = WorkerProcess(
            name=label,
            python=ROOT / section["python"],
            script=ROOT / section["worker"],
            config=section,
            cwd=ROOT,
            on_log=on_worker_log,
        )
        print(f"Starting {label}...", flush=True)
        info = wp.start(timeout_s=timeout_s)
        detail = f"  {label} ready in {info.get('load_s')}s"
        if info.get("vram_gb"):
            detail += f" ({info['vram_gb']} GB VRAM)"
        print(detail, flush=True)
        workers.append(wp)
        return wp

    try:
        stt_worker = spawn("stt", "stt")
        llm_worker = spawn("llm", "llm")
        tts_worker = spawn("tts", "tts")
    except Exception as exc:
        print(f"\nStartup failed: {exc}", file=sys.stderr)
        for w in workers:
            w.stop()
        raise SystemExit(1)

    player = Player(cfg.get("playback", {}).get("device"))
    keep_wavs = bool(cfg.get("playback", {}).get("keep_wavs", False))

    def on_echo(utt_id, text):
        console.line(
            f"  ! echo detected [{utt_id}] — the assistant heard its own output.\n"
            f"    If this repeats, lower tts.normalize.target_lufs or raise "
            f"echo.mute_tail_ms."
        )

    def flush_downstream(utt_id):
        """Drop replies for turns the user has interrupted past.

        Called only on a *confirmed* barge-in. Anything already in the reply or
        wav queues answers a question the user has moved on from; playing it
        after they have started a new turn is worse than dropping it.

        transcript_queue is deliberately left alone: an item there is a user
        turn that has not been answered yet, not a stale reply.
        """
        dropped = 0
        for q_ in (reply_q, wav_q):
            while True:
                try:
                    item = q_.get_nowait()
                except queue.Empty:
                    break
                dropped += 1
                path = getattr(item, "wav_path", None)
                if path is not None and not keep_wavs:
                    path.unlink(missing_ok=True)
        if dropped:
            console.line(f"  flushed {dropped} stale repl{'y' if dropped == 1 else 'ies'}")

    stages = [
        STTStage(stt_worker, spill_dir, audio_q, transcript_q, stop_event, on_stage,
                 recent_speech=recent_speech, echo_threshold=echo_threshold,
                 on_echo=on_echo, interrupt=interrupt, on_flush=flush_downstream),
        LLMStage(llm_worker, transcript_q, reply_q, stop_event, on_stage,
                 recent_speech=recent_speech),
        TTSStage(tts_worker, spill_dir, reply_q, wav_q, stop_event, on_stage,
                 recent_speech=recent_speech),
        PlaybackStage(player, keep_wavs, wav_q, None, stop_event, on_stage,
                      speaking_event=speaking_event, mute_tail_s=mute_tail_s,
                      interrupt=interrupt, recent_speech=recent_speech,
                      verdict_timeout_s=float(
                          barge_cfg.get("verdict_timeout_ms", 6000)) / 1000.0),
    ]

    for s in stages:
        s.start()
    capture.start()

    # -- run until interrupted ---------------------------------------------

    def handle_sigint(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()

    # -- clean shutdown (spec §11) -----------------------------------------
    # Capture first so no new work enters, then let the stages drain, then
    # close the models. Leaving PortAudio open locks the mic until the
    # process is killed.
    console.line("\nShutting down...")
    capture.join(timeout=2)
    for s in stages:
        s.join(timeout=5)
    player.stop()
    for w in workers:
        w.stop()
    console.line("Stopped.")


if __name__ == "__main__":
    main()
