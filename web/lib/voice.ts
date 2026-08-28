/* The browser half of the pipeline: microphone in, reply audio out.
 *
 * Kept as a plain class rather than living inside React state because it deals
 * in things React is bad at — a 16 kHz audio graph, a socket, and an amplitude
 * that has to be read at 60 fps. Re-rendering a component tree for every
 * microphone frame would be the whole cost of the app. The hook around it
 * subscribes to a handful of coarse events; the fast path never touches React.
 *
 * Two AudioContexts, deliberately:
 *
 *   capture  16 kHz  — the rate the server's VAD and both ASR models expect.
 *   output   default — replies come back at 24 kHz or 44.1 kHz, and decoding
 *                      them into a 16 kHz context would resample them down to
 *                      telephone quality on the way to the speakers.
 */
import type {
  Latency,
  ServerEvent,
  SessionState,
  Turn,
  VoiceHandlers,
} from "./types";

const DEFAULT_URL =
  process.env.NEXT_PUBLIC_PIPELINE_URL || "http://127.0.0.1:8000";

/** Events that are the pipeline working normally and need no user-facing note. */
const QUIET_EVENTS = new Set([
  "level", "calibrated", "listening", "speech_start", "utterance", "stt",
  "llm", "tts", "audio", "latency", "muted", "unmuted", "worker_log",
  "worker_ready", "pong", "hello", "flushed", "barge_in_provisional",
]);

let turnSeq = 0;

export class VoiceClient {
  private handlers: VoiceHandlers;
  private baseUrl: string;

  private ws: WebSocket | null = null;
  private captureCtx: AudioContext | null = null;
  private outputCtx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private worklet: AudioWorkletNode | null = null;
  private micAnalyser: AnalyserNode | null = null;
  private outAnalyser: AnalyserNode | null = null;
  private micBuf: Float32Array<ArrayBuffer> | null = null;
  private outBuf: Float32Array<ArrayBuffer> | null = null;

  private source: AudioBufferSourceNode | null = null;
  private playingUttId: string | null = null;
  private stoppedByServer = false;
  private pendingAudio: ServerEvent | null = null;

  private state: SessionState = "idle";
  private muted = false;
  private closing = false;

  /** Set once the server says the frame contract, so the worklet matches it. */
  sampleRate = 16000;
  frameMs = 20;

  constructor(handlers: VoiceHandlers = {}, baseUrl: string = DEFAULT_URL) {
    this.handlers = handlers;
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  // -- lifecycle ----------------------------------------------------------

  async connect(): Promise<void> {
    if (this.ws) return;
    this.closing = false;
    this.setState("connecting");

    // Ask before opening a socket: a refused microphone should not leave a
    // connection holding the pipeline that no one else can use.
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // The browser's own acoustic echo cancellation is far better than
          // anything the server can do after the fact — it has the reference
          // signal. The server-side echo guard stays as a second line, but
          // this is what actually keeps the assistant from hearing itself.
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
    } catch (err) {
      this.fail(
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Microphone permission was denied. Allow it and try again."
          : "No microphone available.",
      );
      return;
    }

    try {
      await this.buildAudioGraph();
    } catch (err) {
      this.fail(`Could not start audio: ${(err as Error).message}`);
      return;
    }

    const wsUrl = this.baseUrl.replace(/^http/, "ws") + "/ws";
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onmessage = (event) => this.onMessage(event);
    ws.onerror = () => {
      if (!this.closing) this.fail("Could not reach the pipeline server.");
    };
    ws.onclose = () => {
      if (!this.closing) {
        this.fail("The pipeline server closed the connection.");
      }
      this.teardownAudio();
      this.ws = null;
    };
  }

  disconnect(): void {
    this.closing = true;
    this.stopPlayback(false);
    try {
      this.ws?.close();
    } catch {
      /* already gone */
    }
    this.ws = null;
    this.teardownAudio();
    this.setState("idle");
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    // The worklet keeps sending frames, they are just silent — see the note
    // there. Stopping the stream instead would starve the server's capture
    // thread and freeze every status it drives.
    this.worklet?.port.postMessage({ type: "mute", value: muted });
  }

  isMuted(): boolean {
    return this.muted;
  }

  /** Peak amplitude, 0..1, of whichever side is currently making sound. */
  amplitude(): number {
    const analyser =
      this.state === "speaking" ? this.outAnalyser : this.micAnalyser;
    const buf = this.state === "speaking" ? this.outBuf : this.micBuf;
    if (!analyser || !buf) return 0;
    analyser.getFloatTimeDomainData(buf);
    let peak = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = Math.abs(buf[i]);
      if (v > peak) peak = v;
    }
    return Math.min(1, peak);
  }

  // -- audio graph --------------------------------------------------------

  private async buildAudioGraph(): Promise<void> {
    const capture = new AudioContext({ sampleRate: this.sampleRate });
    // Autoplay policy: a context created outside a user gesture starts
    // suspended, and a suspended capture context delivers no frames at all.
    if (capture.state === "suspended") await capture.resume();
    await capture.audioWorklet.addModule("/worklets/capture-processor.js");

    const mic = capture.createMediaStreamSource(this.stream!);
    const worklet = new AudioWorkletNode(capture, "capture-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: {
        targetRate: this.sampleRate,
        frameSamples: Math.round((this.sampleRate * this.frameMs) / 1000),
      },
    });
    worklet.port.onmessage = (event) => {
      const ws = this.ws;
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(event.data);
    };

    const analyser = capture.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.6;
    mic.connect(analyser);
    mic.connect(worklet);

    const output = new AudioContext();
    if (output.state === "suspended") await output.resume();
    const outAnalyser = output.createAnalyser();
    outAnalyser.fftSize = 1024;
    outAnalyser.smoothingTimeConstant = 0.6;
    outAnalyser.connect(output.destination);

    this.captureCtx = capture;
    this.outputCtx = output;
    this.worklet = worklet;
    this.micAnalyser = analyser;
    this.outAnalyser = outAnalyser;
    this.micBuf = new Float32Array(analyser.fftSize);
    this.outBuf = new Float32Array(outAnalyser.fftSize);
    if (this.muted) this.setMuted(true);
  }

  private teardownAudio(): void {
    this.worklet?.port.close();
    this.worklet?.disconnect();
    this.worklet = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    void this.captureCtx?.close().catch(() => {});
    void this.outputCtx?.close().catch(() => {});
    this.captureCtx = null;
    this.outputCtx = null;
    this.micAnalyser = null;
    this.outAnalyser = null;
  }

  // -- inbound ------------------------------------------------------------

  private onMessage(event: MessageEvent): void {
    if (event.data instanceof ArrayBuffer) {
      this.onAudioBytes(event.data);
      return;
    }
    let msg: ServerEvent;
    try {
      msg = JSON.parse(event.data as string);
    } catch {
      return;
    }
    this.handlers.onEvent?.(msg);
    this.dispatch(msg);
  }

  private dispatch(msg: ServerEvent): void {
    switch (msg.type) {
      case "hello":
        this.sampleRate = (msg.sample_rate as number) ?? this.sampleRate;
        this.frameMs = (msg.frame_ms as number) ?? this.frameMs;
        this.setState("calibrating");
        return;

      case "fatal":
        this.fail(String(msg.message ?? "The server refused the session."));
        return;

      case "calibrated":
      case "listening":
        this.setState("listening");
        return;

      case "speech_start":
        // Only a transition from waiting. Bleed detected mid-reply is barge-in
        // and is reported separately.
        if (this.state === "listening") this.setState("hearing");
        return;

      case "utterance":
        this.setState("thinking");
        return;

      case "stt":
        this.emitTurn({
          uttId: String(msg.utt_id ?? ""),
          role: "user",
          text: String(msg.text ?? ""),
          lang: (msg.lang as string) ?? null,
          backend: (msg.backend as string) ?? null,
        });
        return;

      case "llm":
        this.emitTurn({
          uttId: String(msg.utt_id ?? ""),
          role: "assistant",
          text: String(msg.text ?? ""),
          lang: (msg.lang as string) ?? null,
        });
        return;

      case "tts_no_voice":
        // The answer is correct, it just cannot be spoken. Showing it as text
        // is the whole point of the server reporting this separately.
        this.emitTurn({
          uttId: String(msg.utt_id ?? ""),
          role: "assistant",
          text: String(msg.text ?? ""),
          lang: (msg.lang as string) ?? null,
          textOnly: true,
          reason: (msg.reason as string) ?? null,
        });
        this.setState("listening");
        return;

      case "audio":
        // The bytes follow as the next frame; nothing to do until they land.
        this.pendingAudio = msg;
        return;

      case "stop_audio":
        this.stopPlayback(true);
        return;

      case "barge_in":
        this.handlers.onBargeIn?.();
        return;

      case "latency":
        this.handlers.onLatency?.({
          uttId: String(msg.utt_id ?? ""),
          sttMs: Number(msg.stt_ms ?? 0),
          llmMs: Number(msg.llm_ms ?? 0),
          ttsMs: Number(msg.tts_ms ?? 0),
          totalMs: Number(msg.total_ms ?? 0),
        });
        return;

      case "stt_empty":
      case "echo_dropped":
        this.setState("listening");
        return;

      case "stt_no_international":
        this.handlers.onNotice?.({
          kind: "warn",
          message:
            `${msg.lang ?? "International"} speech was recognised, but ` +
            `ElevenLabs is not configured — the turn was dropped.`,
        });
        this.setState("listening");
        return;

      default:
        if (
          !QUIET_EVENTS.has(msg.type) &&
          (msg.type.endsWith("_failed") || msg.type === "stage_error" ||
            msg.type === "notice")
        ) {
          this.handlers.onNotice?.({
            kind: msg.type === "notice" ? "info" : "error",
            message: String(msg.error ?? msg.event ?? msg.type),
          });
          if (msg.type.endsWith("_failed")) this.setState("listening");
        }
    }
  }

  // -- playback -----------------------------------------------------------

  private async onAudioBytes(bytes: ArrayBuffer): Promise<void> {
    const meta = this.pendingAudio;
    this.pendingAudio = null;
    const output = this.outputCtx;
    if (!meta || !output) return;

    const uttId = String(meta.utt_id ?? "");
    let buffer: AudioBuffer;
    try {
      buffer = await output.decodeAudioData(bytes);
    } catch {
      this.send({ type: "playback_failed", utt_id: uttId, error: "decode" });
      return;
    }
    // A barge-in can arrive while decoding; do not start audio the server has
    // already cancelled.
    if (this.playingUttId === null && this.stoppedByServer) {
      this.stoppedByServer = false;
      return;
    }

    const src = output.createBufferSource();
    src.buffer = buffer;
    src.connect(this.outAnalyser!);
    src.onended = () => {
      if (this.playingUttId !== uttId) return;
      this.playingUttId = null;
      this.source = null;
      // `stop()` fires onended too, so the two endings are told apart by who
      // asked for it. Reporting a barge-in as a natural finish would let the
      // server believe the reply was heard in full.
      this.send({
        type: this.stoppedByServer ? "playback_stopped" : "playback_finished",
        utt_id: uttId,
      });
      this.stoppedByServer = false;
      this.setState("listening");
    };

    this.source = src;
    this.playingUttId = uttId;
    this.setState("speaking");
    this.send({ type: "playback_started", utt_id: uttId });
    src.start();
  }

  private stopPlayback(fromServer: boolean): void {
    if (!this.source) {
      if (fromServer) this.stoppedByServer = true;
      return;
    }
    this.stoppedByServer = fromServer;
    try {
      this.source.stop();
    } catch {
      /* already stopped */
    }
  }

  // -- plumbing -----------------------------------------------------------

  private send(obj: Record<string, unknown>): void {
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  private emitTurn(turn: Omit<Turn, "id" | "at">): void {
    if (!turn.text) return;
    this.handlers.onTurn?.({ ...turn, id: `t${++turnSeq}`, at: Date.now() });
  }

  private setState(state: SessionState): void {
    if (this.state === state) return;
    this.state = state;
    this.handlers.onState?.(state);
  }

  private fail(message: string): void {
    this.handlers.onNotice?.({ kind: "error", message });
    this.setState("error");
    this.teardownAudio();
  }
}

export async function pipelineHealth(
  baseUrl: string = DEFAULT_URL,
): Promise<{ status: string; error?: string; busy?: boolean } | null> {
  try {
    const res = await fetch(`${baseUrl.replace(/\/$/, "")}/health`, {
      cache: "no-store",
    });
    return await res.json();
  } catch {
    return null;
  }
}

export { DEFAULT_URL };
