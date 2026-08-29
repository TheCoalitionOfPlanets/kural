"use client";

import { useEffect, useRef } from "react";

import styles from "./Transcript.module.css";
import type { Turn } from "@/lib/types";

interface Props {
  turns: Turn[];
  active: boolean;
}

/**
 * Set A — the Indic-specialised models that handle most turns. Anything else
 * was heard by Whisper or spoken by MMS-TTS, which is worth showing: it is the
 * only visible sign that the turn left the local Indic stack.
 */
const LOCAL_BACKENDS = new Set(["sravaani", "indic-mio"]);

export function Transcript({ turns, active }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  // Follow the conversation, but stop following the moment the user scrolls up
  // to read something — a transcript that yanks itself back down is unusable.
  useEffect(() => {
    const el = ref.current;
    if (!el || !pinned.current) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns]);

  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  };

  if (turns.length === 0) {
    return (
      <div className={styles.panel}>
        <p className={styles.empty}>
          {active
            ? "Listening. Just start talking — pause when you are done."
            : "Start a session and speak in any language."}
        </p>
      </div>
    );
  }

  return (
    <div
      ref={ref}
      className={styles.panel}
      onScroll={onScroll}
      role="log"
      aria-live="polite"
      aria-label="Conversation transcript"
    >
      {turns.map((turn) => {
        const setB = turn.backend != null && !LOCAL_BACKENDS.has(turn.backend);
        return (
          <div
            key={turn.id}
            className={`${styles.turn} ${styles[turn.role]}`}
          >
            <div>
              <div className={styles.bubble}>{turn.text}</div>
              <div className={styles.meta}>
                {turn.lang && <span className={styles.tag}>{turn.lang}</span>}
                {setB && (
                  <span className={`${styles.tag} ${styles.setB}`}>
                    {turn.backend}
                  </span>
                )}
                {turn.textOnly && (
                  <span
                    className={`${styles.tag} ${styles.textOnly}`}
                    title={turn.reason ?? undefined}
                  >
                    text only — no voice
                  </span>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
