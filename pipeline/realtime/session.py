"""One assembled pipeline: workers, queues, stages, capture.

`run_realtime.py` and the web server need exactly the same graph — the same
echo guard shared between four places, the same barge-in validation, the same
flush-on-confirm, the same shutdown order. That wiring is subtle enough that
two copies of it would drift, and the ways it drifts are all silent: a stage
that does not get the echo guard, a queue flushed in the wrong order, a
PortAudio stream never closed.

So it lives here once, and the two entrypoints differ only in what they plug
into the ends:

    terminal  MicSource     -> ... -> Player      (sounddevice)
    web       StreamSource  -> ... -> WebPlayer   (a browser)

Everything between those two is identical.
"""
import queue
import threading

from .capture import CaptureThread
from .echo_guard import RecentInput, RecentSpeech
from .interrupt import InterruptController
from .proc import WorkerProcess
from .workers import LLMStage, PlaybackStage, STTStage, TTSStage

# Config sections whose values are repo-relative paths the child resolves
# against its own cwd.
_PATH_KEYS = ("prompt_file", "voices_dir")

# The same, one level down: the Set B models hang off their stage's section
# (stt.whisper, tts.mms_tts) rather than the top level, so their model_dir
# needs the same treatment as the resident models' — a relative path here
# resolves against the *child's* cwd, and the child is started elsewhere.
_NESTED_PATH_KEYS = {
    "stt": ("lid", "whisper"),
    "tts": ("mms_tts",),
}


def spawn_worker(cfg, root, key, label, on_event=None, on_worker_log=None,
                 timeout_s=300):
    """Start one model subprocess and wait for it to report ready."""
    on_event = on_event or (lambda *a, **k: None)
    section = dict(cfg[key])
    # Config paths are repo-relative; the child resolves them against its own
    # cwd, so make them absolute here.
    for path_key in _PATH_KEYS:
        if section.get(path_key):
            section[path_key] = str(root / section[path_key])
    for sub in _NESTED_PATH_KEYS.get(key, ()):
        # Copied before mutating: `cfg` is the caller's parsed config and may
        # be reused (the server spawns workers more than once), so writing an
        # absolute path back into it would be a one-way change.
        block = dict(section.get(sub) or {})
        if block.get("model_dir"):
            block["model_dir"] = str(root / block["model_dir"])
            section[sub] = block
    wp = WorkerProcess(
        name=label,
        python=root / section["python"],
        script=root / section["worker"],
        config=section,
        cwd=root,
        on_log=on_worker_log or (lambda *a, **k: None),
    )
    on_event("worker_starting", worker=label)
    info = wp.start(timeout_s=timeout_s)
    on_event("worker_ready", worker=label, **info)
    return wp, info


def spawn_models(cfg, root, on_event=None, on_worker_log=None, timeout_s=None):
    """Start all three models. Returns ({label: WorkerProcess}, {label: info}).

    Standalone rather than a Session method because the server loads models
    long before any browser connects, and at that point there is no capture
    thread, no VAD and no player to construct — nor any reason to require an
    audio stack on the machine to bring the models up.
    """
    if timeout_s is None:
        timeout_s = int((cfg.get("runtime") or {}).get("startup_timeout_s", 300))
    workers, info = {}, {}
    try:
        for key in ("stt", "llm", "tts"):
            workers[key], info[key] = spawn_worker(
                cfg, root, key, key, on_event, on_worker_log, timeout_s)
    except Exception:
        for wp in workers.values():
            try:
                wp.stop()
            except Exception:
                pass
        raise
    return workers, info


class Session:
    """The pipeline, from a frame source to a player.

    `on_event(kind, **kw)` receives every stage and capture event; the caller
    decides whether that becomes a console line or a WebSocket frame.
    """

    def __init__(self, cfg, root, on_event, player, source=None,
                 on_worker_log=None, models=None):
        self.cfg = cfg
        self.root = root
        self.on_event = on_event
        self.player = player
        self.source = source
        self.on_worker_log = on_worker_log or (lambda *a, **k: None)

        # Models loaded elsewhere and shared across sessions. Loading them
        # takes minutes, so the server does it once at startup and hands the
        # same three processes to every browser that connects; only the queues,
        # the stage threads and the capture thread are per-session. A session
        # never stops models it did not start.
        self.models = dict(models) if models else None
        self._owns_models = models is None

        self.stop_event = threading.Event()
        self.workers = []
        self.stages = []
        self.capture = None
        self.ready_info = {}

        runtime = cfg.get("runtime", {})
        self.spill_dir = root / runtime.get("spill_dir", "pipeline/spill")
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout_s = int(runtime.get("startup_timeout_s", 300))

        q = cfg.get("queues", {})
        self.audio_q = queue.Queue(maxsize=q.get("audio_queue", {}).get("maxsize", 8))
        self.transcript_q = queue.Queue(
            maxsize=q.get("transcript_queue", {}).get("maxsize", 32))
        self.reply_q = queue.Queue(maxsize=q.get("reply_queue", {}).get("maxsize", 32))
        self.wav_q = queue.Queue(maxsize=q.get("wav_queue", {}).get("maxsize", 8))

        # Echo reduction and barge-in are the same problem seen from two sides,
        # so they share state. `speaking_event` marks the assistant as audible;
        # with barge-in off that means "drop mic frames" (airtight,
        # uninterruptible), and with it on it means "apply the strict VAD gate"
        # (interruptible, with the text guard keeping bleed from reaching the
        # model as a user turn).
        echo_cfg = cfg.get("echo", {})
        barge_cfg = cfg.get("barge_in", {})
        self.barge_in_enabled = bool(barge_cfg.get("enabled", False))

        self.recent_speech = RecentSpeech(
            echo_cfg.get("window", 6),
            ttl_s=float(echo_cfg.get("ttl_s", 30)),
            ngram=int(echo_cfg.get("ngram", 5)),
        ) if echo_cfg.get("guard", True) else None
        # The other loop: the same *input* re-entering the pipeline. Separate
        # from the reply window because it holds user turns, not replies, and
        # expires on the shorter clock of a turn in flight rather than the
        # longer one of audible playback.
        input_cfg = echo_cfg.get("input", {})
        self.recent_input = RecentInput(
            input_cfg.get("window", 4),
            ttl_s=float(input_cfg.get("ttl_s", 20)),
            threshold=float(input_cfg.get("threshold", 0.85)),
        ) if input_cfg.get("guard", True) else None
        self.echo_threshold = float(echo_cfg.get("threshold", 0.6))
        self.mute_tail_s = float(echo_cfg.get("mute_tail_ms", 0)) / 1000.0
        self.keep_wavs = bool(cfg.get("playback", {}).get("keep_wavs", False))

        if self.barge_in_enabled and self.recent_speech is None:
            # Tier 1 stops playback on acoustics alone; the text guard is the
            # only thing that can tell whether it was the user. Without it every
            # bleed burst that survives the strict gate reaches the model as a
            # user turn and flushes the pipeline — the self-reply loop, with
            # extra steps.
            raise ValueError(
                "barge_in.enabled requires echo.guard — enable echo.guard or "
                "disable barge_in."
            )

        # The mic must stay live during playback for anything to be
        # interruptible, so barge-in implies the speaking flag is a gate
        # selector, not a mute.
        self.speaking_event = threading.Event() if (
            self.barge_in_enabled
            or echo_cfg.get("mute_capture_while_replying", True)
        ) else None

        self.interrupt = InterruptController(
            on_event=lambda k, **kw: self.on_event(k, **kw)
        ) if self.barge_in_enabled else None

        self.capture = CaptureThread(
            cfg["capture"], self.audio_q, self.stop_event,
            lambda kind, *a, **kw: self.on_event(kind, *a, **kw),
            speaking_event=self.speaking_event, interrupt=self.interrupt,
            source=source,
        )

    # -- lifecycle ---------------------------------------------------------

    def start_models(self):
        """Load the three models. Blocks; raises if any fails to come up.

        A no-op when models were handed in — that is the server's path, where
        the same three processes outlive every browser connection.
        """
        if self.models is None:
            self.models, self.ready_info = spawn_models(
                self.cfg, self.root, self.on_event, self.on_worker_log,
                self.startup_timeout_s)
            self.workers = list(self.models.values())
        return (self.models["stt"], self.models["llm"], self.models["tts"])

    def _flush_downstream(self, utt_id):
        """Drop replies for turns the user has interrupted past.

        Called only on a *confirmed* barge-in. Anything already in the reply or
        wav queues answers a question the user has moved on from; playing it
        after they have started a new turn is worse than dropping it.

        transcript_queue is deliberately left alone: an item there is a user
        turn that has not been answered yet, not a stale reply.
        """
        dropped = 0
        for q_ in (self.reply_q, self.wav_q):
            while True:
                try:
                    item = q_.get_nowait()
                except queue.Empty:
                    break
                dropped += 1
                path = getattr(item, "wav_path", None)
                if path is not None and not self.keep_wavs:
                    path.unlink(missing_ok=True)
        if dropped:
            self.on_event("flushed", count=dropped)

    def start(self):
        """Bring up models, stages and capture. Blocks on model load."""
        stt, llm, tts = self.start_models()

        def on_echo(utt_id, text):
            self.on_event("echo_warning", utt_id=utt_id, text=text)

        barge_cfg = self.cfg.get("barge_in", {})
        self.stages = [
            STTStage(stt, self.spill_dir, self.audio_q, self.transcript_q,
                     self.stop_event, self.on_event,
                     recent_speech=self.recent_speech,
                     echo_threshold=self.echo_threshold, on_echo=on_echo,
                     interrupt=self.interrupt, on_flush=self._flush_downstream,
                     recent_input=self.recent_input),
            LLMStage(llm, self.transcript_q, self.reply_q, self.stop_event,
                     self.on_event, recent_speech=self.recent_speech),
            TTSStage(tts, self.spill_dir, self.reply_q, self.wav_q,
                     self.stop_event, self.on_event,
                     recent_speech=self.recent_speech),
            PlaybackStage(self.player, self.keep_wavs, self.wav_q, None,
                          self.stop_event, self.on_event,
                          speaking_event=self.speaking_event,
                          mute_tail_s=self.mute_tail_s, interrupt=self.interrupt,
                          recent_speech=self.recent_speech,
                          verdict_timeout_s=float(
                              barge_cfg.get("verdict_timeout_ms", 6000)) / 1000.0),
        ]
        for s in self.stages:
            s.start()
        self.capture.start()

    def stop(self, drain_timeout=5):
        """Shut down in order: capture, stages, player, models.

        Capture first so no new work enters, then let the stages drain, then
        close the models. Leaving PortAudio open locks the mic until the
        process is killed.
        """
        self.stop_event.set()
        if self.capture is not None:
            self.capture.join(timeout=2)
        for s in self.stages:
            s.join(timeout=drain_timeout)
        self.stages = []
        try:
            self.player.stop()
        except Exception:
            pass
        # Only models this session started. A server's shared workers must
        # survive a browser closing its tab.
        if self._owns_models:
            for w in self.workers:
                w.stop()
            self.workers = []
            self.models = None
