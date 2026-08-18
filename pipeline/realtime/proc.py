"""Subprocess model host.

The three model stacks pin mutually incompatible transformers versions, so each
runs in its own venv as a child process. The protocol is JSON-lines over
stdin/stdout:

    parent -> child   {"cmd": "run", ...}      one line
    child  -> parent  {"event": "ready"}       once, after model load
    child  -> parent  {"ok": true, ...}        one line per request

Anything the child writes to stderr is surfaced as a log line, so tracebacks
and HF progress bars are visible instead of silently swallowed.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


def _child_env(cwd):
    """Environment for a model subprocess.

    HF_HOME may point at a drive that is not currently mounted (transformers
    then dies creating its module cache, before any model is touched). Every
    model here is loaded from a local directory, so fall back to an in-repo
    cache rather than failing the run.
    """
    env = os.environ.copy()
    hf_home = env.get("HF_HOME")
    if hf_home and not os.path.isdir(os.path.splitdrive(hf_home)[0] + os.sep):
        local = str(Path(cwd) / ".hf_cache")
        env["HF_HOME"] = local
        env["HF_HUB_CACHE"] = str(Path(local) / "hub")
        env["HF_HUB_OFFLINE"] = "1"
    return env


class WorkerProcess:
    """One model, resident in a child process."""

    def __init__(self, name, python, script, config, cwd, on_log=None):
        self.name = name
        self.python = str(python)
        self.script = str(script)
        self.config = config
        self.cwd = str(cwd)
        self.on_log = on_log or (lambda *a, **k: None)
        self.proc = None
        self._lock = threading.Lock()
        self._stderr_thread = None

    def start(self, timeout_s=300):
        self.proc = subprocess.Popen(
            [self.python, self.script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=_child_env(self.cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name=f"{self.name}-stderr", daemon=True
        )
        self._stderr_thread.start()

        self._send({"cmd": "init", "config": self.config})

        # Wait for the model to finish loading. A child that dies during load
        # closes stdout, so readline returns "" rather than hanging forever.
        deadline = threading.Event()
        result = {}

        def _wait():
            line = self.proc.stdout.readline()
            result["line"] = line
            deadline.set()

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
        if not deadline.wait(timeout_s):
            raise TimeoutError(f"{self.name}: model load exceeded {timeout_s}s")

        line = result.get("line", "")
        if not line:
            raise RuntimeError(f"{self.name}: worker exited during startup")
        msg = json.loads(line)
        if msg.get("event") != "ready":
            raise RuntimeError(f"{self.name}: unexpected startup message: {msg}")
        return msg

    def _drain_stderr(self):
        for line in self.proc.stderr:
            line = line.rstrip()
            if line:
                self.on_log(self.name, line)

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def run(self, payload):
        """Send one request, block for its reply. Serialized per worker."""
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError(f"{self.name}: worker is not running")
            self._send({"cmd": "run", **payload})
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name}: worker died mid-request")
            return json.loads(line)

    def stop(self, timeout=10):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self._send({"cmd": "shutdown"})
                self.proc.wait(timeout=timeout)
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
