"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { VoiceClient, pipelineHealth } from "./voice";
import type { Latency, Notice, SessionState, Turn } from "./types";

let noticeSeq = 0;

export function useVoice() {
  const [state, setState] = useState<SessionState>("idle");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [latency, setLatency] = useState<Latency | null>(null);
  const [notices, setNotices] = useState<Notice[]>([]);
  const [muted, setMutedState] = useState(false);
  const [serverStatus, setServerStatus] = useState<string>("unknown");
  const [bargeInAt, setBargeInAt] = useState(0);

  const clientRef = useRef<VoiceClient | null>(null);

  const pushNotice = useCallback((n: Omit<Notice, "id" | "at">) => {
    const notice: Notice = { ...n, id: `n${++noticeSeq}`, at: Date.now() };
    setNotices((prev) => [...prev.slice(-4), notice]);
    // Errors are the only kind worth keeping on screen until dismissed; the
    // rest are transient status the user should not have to clear.
    if (notice.kind !== "error") {
      setTimeout(
        () => setNotices((prev) => prev.filter((x) => x.id !== notice.id)),
        6000,
      );
    }
  }, []);

  const client = useMemo(() => {
    if (typeof window === "undefined") return null;
    const c = new VoiceClient({
      onState: setState,
      onLatency: setLatency,
      onNotice: pushNotice,
      onBargeIn: () => setBargeInAt(Date.now()),
      onTurn: (turn) =>
        setTurns((prev) => {
          // A reply with no voice arrives twice — once as the model's text,
          // once as the "cannot speak this" report carrying the same text. The
          // second is an annotation on the first, not another message.
          const i = prev.findIndex(
            (t) => t.uttId === turn.uttId && t.role === turn.role,
          );
          if (i === -1) return [...prev, turn];
          const merged = [...prev];
          merged[i] = { ...merged[i], ...turn, id: merged[i].id };
          return merged;
        }),
    });
    clientRef.current = c;
    return c;
  }, [pushNotice]);

  // Poll for readiness while the models load, so the button can say what is
  // actually happening instead of failing on click.
  useEffect(() => {
    let alive = true;
    const check = async () => {
      const health = await pipelineHealth();
      if (!alive) return;
      setServerStatus(health ? health.status : "offline");
    };
    void check();
    const id = setInterval(check, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => () => clientRef.current?.disconnect(), []);

  const connect = useCallback(async () => {
    setTurns([]);
    setLatency(null);
    setNotices([]);
    await client?.connect();
  }, [client]);

  const disconnect = useCallback(() => client?.disconnect(), [client]);

  const toggleMute = useCallback(() => {
    if (!client) return;
    const next = !client.isMuted();
    client.setMuted(next);
    setMutedState(next);
  }, [client]);

  const dismissNotice = useCallback(
    (id: string) => setNotices((prev) => prev.filter((n) => n.id !== id)),
    [],
  );

  // Deliberately a getter, not state: the orb reads this every animation frame
  // and re-rendering the tree 60 times a second would be the entire cost of
  // the page.
  const getAmplitude = useCallback(() => client?.amplitude() ?? 0, [client]);

  const active = state !== "idle" && state !== "error";

  return {
    state,
    turns,
    latency,
    notices,
    muted,
    active,
    serverStatus,
    bargeInAt,
    connect,
    disconnect,
    toggleMute,
    dismissNotice,
    getAmplitude,
  };
}
