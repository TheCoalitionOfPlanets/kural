"use client";

import styles from "./StatusRail.module.css";
import type { Latency, Notice, SessionState } from "@/lib/types";

interface Props {
  state: SessionState;
  notices: Notice[];
  muted: boolean;
  onDismiss: (id: string) => void;
}

const LABELS: Record<SessionState, string> = {
  idle: "Ready",
  connecting: "Connecting",
  loading: "Loading models",
  calibrating: "Calibrating",
  listening: "Listening",
  hearing: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  error: "Something went wrong",
};

const HINTS: Partial<Record<SessionState, string>> = {
  // No hint while idle. The row keeps its height either way (.hint has a
  // min-height), so nothing shifts when the first real hint appears.
  calibrating: "Measuring the room's noise floor. Stay quiet for a moment.",
  listening: "Go ahead — pause when you are done speaking.",
  hearing: "Listening…",
  thinking: "Working on it.",
  speaking: "Speak over it to interrupt.",
};

export function StatusRail({ state, notices, muted, onDismiss }: Props) {
  const live = state !== "idle" && state !== "error";
  const hint = muted && live ? "Microphone is muted." : HINTS[state];

  return (
    <>
      <div className={styles.rail}>
        <div className={styles.state}>
          <span
            className={`${styles.dot} ${
              state === "error" ? styles.bad : live ? styles.live : ""
            }`}
          />
          {muted && live ? "Muted" : LABELS[state]}
        </div>

        <p className={styles.hint}>{hint ?? ""}</p>
      </div>

      {notices.length > 0 && (
        <div className={styles.notices}>
          {notices.map((n) => (
            <div key={n.id} className={styles.notice} data-kind={n.kind} role="status">
              <span>{n.message}</span>
              <button
                className={styles.dismiss}
                onClick={() => onDismiss(n.id)}
                aria-label="Dismiss"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

/** Per-turn timings, shown just above the controls rather than in the rail:
 *  it is telemetry, and between the hint and the transcript it competed with
 *  both for the same few pixels. */
export function LatencyRow({ latency }: { latency: Latency | null }) {
  if (!latency) return null;
  return (
    <div
      className={styles.latency}
      title="End of speech to first audio, per stage"
    >
      <span>
        stt <b>{latency.sttMs}ms</b>
      </span>
      <span>
        llm <b>{latency.llmMs}ms</b>
      </span>
      <span>
        tts <b>{latency.ttsMs}ms</b>
      </span>
      <span>
        total <b>{latency.totalMs}ms</b>
      </span>
    </div>
  );
}
