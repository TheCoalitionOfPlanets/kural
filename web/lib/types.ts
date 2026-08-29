/** The wire protocol, as pipeline/server/app.py emits it. */

export type SessionState =
  | "idle"          // nothing connected
  | "connecting"    // socket opening, mic being granted
  | "loading"       // server has the socket but models are still coming up
  | "calibrating"   // measuring this room's noise floor from real frames
  | "listening"     // live, waiting for speech
  | "hearing"       // the user is speaking right now
  | "thinking"      // utterance closed, pipeline working
  | "speaking"      // a reply is playing
  | "error";

export type Role = "user" | "assistant";

export interface Turn {
  id: string;
  uttId: string;
  role: Role;
  text: string;
  /** Language decided from the audio, not from the words. */
  lang?: string | null;
  /** "sravaani" | "whisper" | "indic-mio" | "mms-tts" — shown when not Set A. */
  backend?: string | null;
  /** The reply could not be spoken; it is text only. */
  textOnly?: boolean;
  reason?: string | null;
  at: number;
}

export interface Latency {
  uttId: string;
  sttMs: number;
  llmMs: number;
  ttsMs: number;
  totalMs: number;
}

export interface Notice {
  id: string;
  kind: "info" | "warn" | "error";
  message: string;
  at: number;
}

/** Anything the server sends as JSON. `type` is the event name. */
export interface ServerEvent {
  type: string;
  [key: string]: unknown;
}

export interface VoiceHandlers {
  onState?: (state: SessionState) => void;
  onTurn?: (turn: Turn) => void;
  onLatency?: (latency: Latency) => void;
  onNotice?: (notice: Omit<Notice, "id" | "at">) => void;
  onEvent?: (event: ServerEvent) => void;
  onBargeIn?: () => void;
}
