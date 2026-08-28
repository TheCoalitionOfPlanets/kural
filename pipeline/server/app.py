"""WebSocket server: a browser is the microphone and the speakers.

The pipeline was written around a local sounddevice mic and a local player.
Nothing about the tuned part of it — the VAD endpointing, the barge-in tiers,
the echo guard, the language gate — actually depends on that; they depend on
*frames arriving* and on *knowing when the assistant is audible*. So the two
ends are swapped and everything between is untouched:

    browser mic ─ WS binary ─▶ StreamSource ─▶ [ the whole pipeline ] ─▶ WebPlayer ─ WS ─▶ browser

Two things about the shape here matter.

**Models load once, sessions are cheap.** The three subprocesses take minutes
to come up and hold ~7 GB of VRAM, so they are started with the server and
shared. A browser connecting builds only queues, four stage threads and a
capture thread — all of which start in milliseconds and are torn down when the
tab closes.

**One browser at a time.** There is one GPU and one conversation history; a
second connection is refused with a reason rather than silently interleaved
into the first one's turns.

Run it with the root venv, which is where the orchestrator's deps live:

    venv/bin/python -m pipeline.server            # or venv\\Scripts\\python.exe
"""
import asyncio
import contextlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.realtime.capture import StreamSource  # noqa: E402
from pipeline.realtime.session import Session, spawn_models  # noqa: E402
from pipeline.realtime.web_player import WebPlayer  # noqa: E402

DEFAULT_CONFIG = ROOT / "pipeline" / "config" / "realtime.yaml"

# Capture-side events the browser does not need. `level` is the exception that
# proves the rule — it drives the orb, so it is forwarded, but at 50/s it is
# the one event worth thinking about before adding more.
_SILENT_EVENTS = frozenset({"too_short", "worker_starting"})


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Bridge:
    """Thread-safe outbound side of one WebSocket.

    Everything that produces events — the capture thread, four stage threads,
    the interrupt controller — runs on threads, while the socket lives on the
    event loop. `call_soon_threadsafe` is the only legal crossing, so all of
    them funnel through here into an asyncio queue that one writer task drains.

    The queue is bounded. A browser that stops reading must not be allowed to
    grow this without limit; dropping the oldest *status* frame is harmless,
    and audio is never dropped because losing it would wedge the playback stage
    waiting for a report that can never come.
    """

    def __init__(self, loop, maxsize=512):
        self.loop = loop
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.closed = False

    def _put(self, item):
        if self.closed:
            return
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            kind, _ = item
            if kind == "bytes":
                return  # never dropped; see the class docstring
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self.queue.put_nowait(item)

    def send_json(self, obj):
        if not self.closed:
            self.loop.call_soon_threadsafe(self._put, ("json", obj))

    def send_bytes(self, data):
        if not self.closed:
            self.loop.call_soon_threadsafe(self._put, ("bytes", data))

    def close(self):
        self.closed = True
        self.loop.call_soon_threadsafe(self._put, ("close", None))


class Hub:
    """Owns the shared models and whichever browser currently holds them."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.models = None
        self.model_info = {}
        self.startup_error = None
        self._lock = threading.Lock()
        self._holder = None

    # -- models ------------------------------------------------------------

    def load_models(self, on_event=None, on_worker_log=None):
        """Start the three subprocesses. Blocks for as long as they take."""
        self.models, self.model_info = spawn_models(
            self.cfg, ROOT, on_event, on_worker_log)
        return self.models

    def shutdown(self):
        for wp in (self.models or {}).values():
            with contextlib.suppress(Exception):
                wp.stop()
        self.models = None

    # -- exclusive use -----------------------------------------------------

    def acquire(self, who):
        with self._lock:
            if self._holder is not None:
                return False
            self._holder = who
            return True

    def release(self, who):
        with self._lock:
            if self._holder is who:
                self._holder = None

    @property
    def busy(self):
        return self._holder is not None


class Connection:
    """One browser: its session, its player, its frame source."""

    def __init__(self, hub, websocket, loop):
        self.hub = hub
        self.ws = websocket
        self.loop = loop
        self.bridge = Bridge(loop)
        self.source = StreamSource()
        self.player = WebPlayer(
            self.bridge.send_json, self.bridge.send_bytes,
            on_notice=lambda kind, **kw: self.emit("notice", event=kind, **kw),
        )
        self.session = None
        self.sample_rate = int(cfg_get(hub.cfg, "capture", "sample_rate", 16000))
        self.frame_ms = int(cfg_get(hub.cfg, "capture", "frame_ms", 20))
        # `level` fires every frame — 50/s. The orb cannot use more than a
        # handful, and each one is a WebSocket frame.
        self._last_level = 0.0
        self._level_interval = 1 / 20.0

    # -- outbound ----------------------------------------------------------

    def emit(self, kind, *args, **kw):
        """Every capture, stage and session event. Called on worker threads."""
        if kind in _SILENT_EVENTS:
            return
        if kind == "level":
            now = time.monotonic()
            if now - self._last_level < self._level_interval:
                return
            self._last_level = now
        if kind == "audio" and args:
            kw["message"] = str(args[0])
        self.bridge.send_json({"type": kind, **_jsonable(kw)})

    def worker_log(self, name, line):
        self.bridge.send_json({"type": "worker_log", "worker": name, "line": line})

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self.session = Session(
            self.hub.cfg, ROOT, self.emit, self.player,
            source=self.source, on_worker_log=self.worker_log,
            models=self.hub.models,
        )
        self.session.start()

    def stop(self):
        self.player.abandon()
        if self.session is not None:
            self.session.stop()
            self.session = None
        self.bridge.close()

    # -- inbound -----------------------------------------------------------

    def on_audio(self, data):
        """Raw int16 little-endian mono PCM at the capture rate."""
        if not data:
            return
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self.source.push(pcm)

    def on_message(self, msg):
        if self.player.note(msg):
            return
        kind = msg.get("type")
        if kind == "ping":
            self.bridge.send_json({"type": "pong", "t": msg.get("t")})


def cfg_get(cfg, section, key, default=None):
    return (cfg.get(section) or {}).get(key, default)


def _jsonable(kw):
    """Events carry Paths and numpy scalars; JSON carries neither."""
    out = {}
    for k, v in kw.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (str, int, float, bool, type(None), list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def create_app(config_path=DEFAULT_CONFIG):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    cfg = load_config(config_path)
    hub = Hub(cfg)
    server_cfg = cfg.get("server") or {}

    @contextlib.asynccontextmanager
    async def lifespan(app):
        # Loading blocks for minutes, so it runs in a thread and the HTTP
        # server answers /health with "loading" in the meantime — the browser
        # shows real progress instead of a refused connection.
        def _load():
            try:
                hub.load_models(
                    on_event=lambda k, **kw: print(f"  {k}: {kw}", flush=True),
                    on_worker_log=lambda n, l: print(f"  [{n}] {l}", flush=True),
                )
                print("models ready", flush=True)
            except Exception as exc:
                hub.startup_error = str(exc)
                print(f"model startup failed: {exc}", file=sys.stderr, flush=True)

        thread = threading.Thread(target=_load, name="model-load", daemon=True)
        thread.start()
        try:
            yield
        finally:
            hub.shutdown()

    app = FastAPI(title="kural voice pipeline", lifespan=lifespan)
    # Only /health is subject to CORS — WebSockets are not — but /health is
    # what the UI polls to decide whether its start button works, so getting
    # this wrong shows up as a permanent "Server offline" with a healthy
    # server behind it. Any loopback port is allowed by default because the
    # dev port is not knowable: Next picks another one whenever 3000 is taken.
    # `server.allow_origins` replaces this entirely when the server is exposed
    # somewhere real.
    explicit = server_cfg.get("allow_origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=explicit or [],
        allow_origin_regex=None if explicit
        else r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "error" if hub.startup_error
            else "ready" if hub.models else "loading",
            "error": hub.startup_error,
            "busy": hub.busy,
            "models": hub.model_info,
            "sample_rate": cfg_get(cfg, "capture", "sample_rate", 16000),
            "frame_ms": cfg_get(cfg, "capture", "frame_ms", 20),
        }

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()

        if hub.startup_error:
            await websocket.send_json({"type": "fatal", "message": hub.startup_error})
            await websocket.close()
            return
        if hub.models is None:
            await websocket.send_json({
                "type": "fatal",
                "message": "models are still loading — try again in a moment",
                "retry": True,
            })
            await websocket.close()
            return

        conn = Connection(hub, websocket, loop)
        if not hub.acquire(conn):
            await websocket.send_json({
                "type": "fatal",
                "message": "another session is already using the pipeline",
            })
            await websocket.close()
            return

        writer = asyncio.create_task(_writer(websocket, conn.bridge))
        try:
            await websocket.send_json({
                "type": "hello",
                "sample_rate": conn.sample_rate,
                "frame_ms": conn.frame_ms,
                "models": hub.model_info,
                "barge_in": bool((cfg.get("barge_in") or {}).get("enabled", False)),
            })
            # Stage threads start instantly; the capture thread then spends
            # capture.vad.calibration_s measuring this room's noise floor from
            # the browser's own frames, and reports it as `calibrated`.
            await asyncio.to_thread(conn.start)

            while True:
                packet = await websocket.receive()
                if packet.get("type") == "websocket.disconnect":
                    break
                if (data := packet.get("bytes")) is not None:
                    conn.on_audio(data)
                elif (text := packet.get("text")) is not None:
                    with contextlib.suppress(json.JSONDecodeError):
                        conn.on_message(json.loads(text))
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 — the socket is going away regardless
            print(f"session error: {exc!r}", file=sys.stderr, flush=True)
        finally:
            # Nothing here may await. A client disconnecting cancels this task,
            # and a cancelled task resumes its `finally` only as far as the
            # first await — everything after it is skipped. Awaiting the
            # teardown here meant `hub.release` never ran, and the pipeline
            # stayed locked to a browser that had already closed its tab.
            #
            # So teardown runs on its own thread and releases the hub when it
            # is genuinely finished. A connection arriving in the meantime is
            # told the pipeline is busy, which is the truth: the previous
            # session's capture and stage threads are still winding down.
            writer.cancel()
            threading.Thread(
                target=_teardown, args=(conn, hub), name="session-teardown",
                daemon=True,
            ).start()

    return app


def _teardown(conn, hub):
    """Stop one session, then release the pipeline for the next browser.

    `conn.stop()` abandons the player first, which unblocks a playback stage
    sitting on a report from a browser that has already gone.
    """
    try:
        conn.stop()
    except Exception as exc:  # noqa: BLE001 — must still release
        print(f"teardown error: {exc!r}", file=sys.stderr, flush=True)
    finally:
        hub.release(conn)


async def _writer(websocket, bridge):
    """Drain the outbound queue onto the socket."""
    while True:
        kind, payload = await bridge.queue.get()
        if kind == "close":
            return
        try:
            if kind == "json":
                await websocket.send_json(payload)
            else:
                await websocket.send_bytes(payload)
        except Exception:
            return


def main(argv=None):
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    server_cfg = cfg.get("server") or {}
    host = args.host or server_cfg.get("host", "127.0.0.1")
    port = args.port or int(server_cfg.get("port", 8000))

    print("=" * 60)
    print("KURAL VOICE SERVER")
    print("=" * 60)
    print(f"  ws://{host}:{port}/ws")
    print("  models load in the background; /health reports progress")
    print("=" * 60, flush=True)

    uvicorn.run(create_app(args.config), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
