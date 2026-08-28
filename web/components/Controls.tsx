"use client";

import styles from "./Controls.module.css";
import {
  ListIcon,
  MicIcon,
  MicOffIcon,
  MoonIcon,
  StopIcon,
  SunIcon,
} from "./icons";
import type { SessionState } from "@/lib/types";

interface Props {
  state: SessionState;
  active: boolean;
  muted: boolean;
  serverStatus: string;
  showTranscript: boolean;
  theme: "light" | "dark";
  onConnect: () => void;
  onDisconnect: () => void;
  onToggleMute: () => void;
  onToggleTranscript: () => void;
  onToggleTheme: () => void;
}

function startLabel(serverStatus: string): { text: string; ready: boolean } {
  // The button says what the server is actually doing. Models take minutes to
  // load, and a button that just fails on click looks broken rather than busy.
  if (serverStatus === "loading") return { text: "Loading models…", ready: false };
  if (serverStatus === "offline") return { text: "Server offline", ready: false };
  if (serverStatus === "error") return { text: "Server error", ready: false };
  return { text: "Start talking", ready: true };
}

export function Controls({
  state,
  active,
  muted,
  serverStatus,
  showTranscript,
  theme,
  onConnect,
  onDisconnect,
  onToggleMute,
  onToggleTranscript,
  onToggleTheme,
}: Props) {
  const start = startLabel(serverStatus);
  const connecting = state === "connecting";

  return (
    <div className={styles.bar}>
      <button
        className={`${styles.btn} ${showTranscript ? styles.on : ""}`}
        onClick={onToggleTranscript}
        aria-pressed={showTranscript}
        aria-label="Toggle transcript"
        title="Transcript"
      >
        <ListIcon className={styles.icon} />
      </button>

      {active ? (
        <>
          <button
            className={`${styles.btn} ${muted ? styles.on : ""}`}
            onClick={onToggleMute}
            aria-pressed={muted}
            aria-label={muted ? "Unmute microphone" : "Mute microphone"}
            title={muted ? "Unmute" : "Mute"}
          >
            {muted ? (
              <MicOffIcon className={styles.icon} />
            ) : (
              <MicIcon className={styles.icon} />
            )}
          </button>
          <button
            className={`${styles.btn} ${styles.danger}`}
            onClick={onDisconnect}
            aria-label="End session"
            title="End session"
          >
            <StopIcon className={styles.icon} />
          </button>
        </>
      ) : (
        <button
          className={styles.start}
          onClick={onConnect}
          disabled={!start.ready || connecting}
        >
          <MicIcon className={styles.icon} />
          {connecting ? "Connecting…" : start.text}
        </button>
      )}

      <button
        className={styles.btn}
        onClick={onToggleTheme}
        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        title="Theme"
      >
        {theme === "dark" ? (
          <SunIcon className={styles.icon} />
        ) : (
          <MoonIcon className={styles.icon} />
        )}
      </button>
    </div>
  );
}
