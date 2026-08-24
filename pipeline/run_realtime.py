"""Always-listening speech -> reasoning -> speech pipeline.

    mic -> VAD -> [LID] -> [SraVaani STT ] -> [Gemma 3 4B] -> [Indic-Mio  ] -> speaker
                           [ElevenLabs   ]                    [ElevenLabs ]

Each model runs as a subprocess in its own venv (their transformers pins are
mutually incompatible); this process owns capture, the queues, scheduling and
playback.

The two ears and the two voices are chosen per utterance. The local models are
Indic by construction, so a turn in Spanish or Japanese is heard and spoken by
ElevenLabs instead — decided from the waveform by the language-ID gate in the
STT worker, and from the reply's language at TTS. Everything downstream of
that choice is identical: one WAV, one player, the same VAD gate, echo guard
and barge-in.

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
from pipeline.realtime.session import Session

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "pipeline" / "config" / "realtime.yaml"

# Capture and session events render differently from stage events, and the
# stage set cannot be enumerated (every "<stage>_failed" is one), so the small
# closed set is the one named here and everything else falls through.
_CAPTURE_EVENTS = frozenset({
    "level", "calibrated", "listening", "speech_start", "utterance", "muted",
    "unmuted", "barge_in", "dropped", "frame_drops", "audio", "too_short",
    "worker_starting", "worker_ready", "flushed", "echo_warning",
})

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
    eleven_cfg = cfg.get("elevenlabs") or {}
    playback_cfg = cfg.get("playback", {})

    # -- status routing ----------------------------------------------------

    threshold_holder = {"value": 0.0}

    def on_event(kind, *args, **kw):
        """Every capture, stage and session event lands here."""
        if kind not in _CAPTURE_EVENTS:
            return on_stage(kind, **kw)
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
        elif kind == "too_short":
            pass  # too brief to transcribe; the VAD already moved on
        elif kind == "worker_starting":
            print(f"Starting {kw['worker']}...", flush=True)
        elif kind == "worker_ready":
            _report_ready(kw)
        elif kind == "flushed":
            n = kw["count"]
            console.line(f"  flushed {n} stale repl{'y' if n == 1 else 'ies'}")
        elif kind == "echo_warning":
            console.line(
                f"  ! echo detected [{kw['utt_id']}] — the assistant heard its "
                f"own output.\n    If this repeats, lower "
                f"tts.normalize.target_lufs or raise echo.mute_tail_ms."
            )

    def _report_ready(info):
        label = info["worker"]
        detail = f"  {label} ready in {info.get('load_s')}s"
        if info.get("vram_gb"):
            detail += f" ({info['vram_gb']} GB VRAM)"
        print(detail, flush=True)

        # The international path is optional and fails soft — the pipeline runs
        # without it, just local-only. That has to be visible here rather than
        # discovered halfway through a Spanish sentence, so both halves report
        # whether they actually came up.
        if info.get("lid") is False:
            print(f"    ! language ID off ({info.get('lid_error') or 'disabled'})"
                  f" — every turn routes to the local Indic models", flush=True)
        if info.get("elevenlabs") is False:
            missing = "transcribe" if label == "stt" else "speak"
            print(f"    ! ElevenLabs off — cannot {missing} languages outside "
                  f"the local set. Set "
                  f"${eleven_cfg.get('api_key_env', 'ELEVENLABS_API_KEY')}.",
                  flush=True)

    def on_stage(kind, **kw):
        utt = kw.get("utt_id", "")
        if kind == "stt":
            # The backend is named only when it is not the local one. A tag on
            # every line would be noise; a tag on the international ones is the
            # only visible sign that a paid call was made.
            tag = ""
            if kw.get("backend") == "elevenlabs":
                tag = f" ({kw.get('lang') or 'international'} via elevenlabs)"
            console.line(f"[{utt}] heard{tag}: {kw['text']}")
        elif kind == "llm":
            lang = kw.get("lang")
            tag = f" ({lang})" if lang else ""
            console.line(f"[{utt}] reply{tag}: {kw['text']}")
        elif kind == "tts":
            tag = " via elevenlabs" if kw.get("backend") == "elevenlabs" else ""
            console.line(f"[{utt}] synth {kw.get('audio_s')}s "
                         f"in {kw.get('elapsed_s')}s{tag}")
        elif kind == "latency":
            if console.log_latency:
                console.line(
                    f"[{utt}] latency stt={kw['stt_ms']}ms llm={kw['llm_ms']}ms "
                    f"tts={kw['tts_ms']}ms total={kw['total_ms']}ms"
                )
        elif kind == "stt_empty":
            console.line(f"[{utt}] no speech recognized")
        elif kind == "stt_no_international":
            # Identified as a language the local ear cannot hear, with nowhere
            # to send it. Transcribing it locally anyway would return confident
            # gibberish, so the turn is dropped and the reason named.
            console.line(
                f"[{utt}] ! {kw.get('lang') or 'international'} speech, but "
                f"ElevenLabs is not configured — turn dropped. Set "
                f"${cfg.get('elevenlabs', {}).get('api_key_env', 'ELEVENLABS_API_KEY')}"
                f" and restart."
            )
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
            # Neither voice speaks this language, so the reply is text-only.
            # Print it rather than dropping it silently — the answer is still
            # correct, it just cannot be spoken.
            why = f" ({kw['reason']})" if kw.get("reason") else ""
            console.line(f"[{utt}] no {kw.get('lang')} voice{why}, text only: "
                         f"{kw.get('text')}")
        elif kind.endswith("_failed") or kind == "stage_error":
            console.line(f"  ! {kind}: {kw.get('error')}")

    def on_worker_log(name, line):
        console.line(f"  [{name}] {line}")

    # -- capture-only mode (build order step 1) ----------------------------

    if args.capture_only:
        import soundfile as sf

        # Capture and VAD only — no models, no queues past the first one.
        q = cfg.get("queues", {})
        audio_q = queue.Queue(maxsize=q.get("audio_queue", {}).get("maxsize", 8))
        capture = CaptureThread(cfg["capture"], audio_q, stop_event, on_event)

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

    # -- the pipeline ------------------------------------------------------

    print("=" * 60)
    print("REAL-TIME SPEECH PIPELINE")
    print("=" * 60)

    try:
        session = Session(cfg, ROOT, on_event, Player(playback_cfg.get("device")),
                          on_worker_log=on_worker_log)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

    try:
        session.start()
    except Exception as exc:
        print(f"\nStartup failed: {exc}", file=sys.stderr)
        session.stop()
        raise SystemExit(1)

    # -- run until interrupted ---------------------------------------------

    def handle_sigint(signum, frame):
        session.stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not session.stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        session.stop_event.set()

    console.line("\nShutting down...")
    session.stop()
    console.line("Stopped.")


if __name__ == "__main__":
    main()
