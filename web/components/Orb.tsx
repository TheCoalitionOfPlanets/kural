"use client";

import { useEffect, useRef } from "react";

import styles from "./Orb.module.css";
import type { SessionState } from "@/lib/types";

interface Props {
  state: SessionState;
  getAmplitude: () => number;
  bargeInAt: number;
}

/** How fast the rendered amplitude chases the real one, per frame. */
const ATTACK = 0.35;
const RELEASE = 0.12;

export function Orb({ state, getAmplitude, bargeInAt }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    let raf = 0;
    let smoothed = 0;

    const tick = () => {
      const live =
        stateRef.current === "hearing" ||
        stateRef.current === "listening" ||
        stateRef.current === "speaking";
      const target = live ? getAmplitude() : 0;
      // Asymmetric smoothing: rise quickly so a syllable registers, fall
      // slowly so the orb does not flicker between words.
      smoothed += (target - smoothed) * (target > smoothed ? ATTACK : RELEASE);
      el.style.setProperty("--amp", smoothed.toFixed(4));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [getAmplitude]);

  // One ripple per barge-in. Keyed on the timestamp so React remounts the
  // element and the animation restarts even on consecutive interrupts.
  return (
    <div ref={wrapRef} className={styles.wrap} data-state={state}>
      <div className={styles.glow} />
      <div className={styles.ring} />
      <div className={styles.sweep} />
      {bargeInAt > 0 && <span key={bargeInAt} className={styles.pulse} />}
      <div className={styles.core} />
    </div>
  );
}
