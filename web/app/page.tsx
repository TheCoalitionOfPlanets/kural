"use client";

import { useCallback, useEffect, useState } from "react";

import styles from "./page.module.css";
import { Controls } from "@/components/Controls";
import { Orb } from "@/components/Orb";
import { LatencyRow, StatusRail } from "@/components/StatusRail";
import { Transcript } from "@/components/Transcript";
import { useVoice } from "@/lib/useVoice";

type Theme = "light" | "dark";

function initialTheme(): Theme {
  if (typeof window === "undefined") return "light";
  const stored = window.localStorage.getItem("kural-theme");
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export default function Page() {
  const voice = useVoice();
  const [showTranscript, setShowTranscript] = useState(true);
  const [theme, setTheme] = useState<Theme>("light");

  // Read after mount: the inline script in layout.tsx has already applied the
  // stored theme to <html>, so this only syncs React's copy of it.
  useEffect(() => setTheme(initialTheme()), []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try {
        window.localStorage.setItem("kural-theme", next);
      } catch {
        /* private mode; the choice just will not persist */
      }
      return next;
    });
  }, []);

  // Space toggles the mic, the way a call app does. Ignored while typing so it
  // cannot fight a future text input.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement;
      if (el instanceof HTMLElement && (el.isContentEditable || el.tagName === "INPUT")) {
        return;
      }
      if (e.code === "Space" && voice.active) {
        e.preventDefault();
        voice.toggleMute();
      }
      if (e.key === "Escape" && voice.active) voice.disconnect();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [voice]);

  return (
    <main className={styles.shell}>
      <span className={styles.brand}>Kural</span>

      <div className={styles.stage}>
        <Orb
          state={voice.state}
          getAmplitude={voice.getAmplitude}
          bargeInAt={voice.bargeInAt}
        />
        <StatusRail
          state={voice.state}
          notices={voice.notices}
          muted={voice.muted}
          onDismiss={voice.dismissNotice}
        />
      </div>

      <div className={styles.body}>
        {showTranscript && (
          <Transcript turns={voice.turns} active={voice.active} />
        )}
      </div>

      <div className={styles.dock}>
        <LatencyRow latency={voice.latency} />
        <Controls
          state={voice.state}
          active={voice.active}
          muted={voice.muted}
          serverStatus={voice.serverStatus}
          showTranscript={showTranscript}
          theme={theme}
          onConnect={voice.connect}
          onDisconnect={voice.disconnect}
          onToggleMute={voice.toggleMute}
          onToggleTranscript={() => setShowTranscript((v) => !v)}
          onToggleTheme={toggleTheme}
        />
      </div>
    </main>
  );
}
