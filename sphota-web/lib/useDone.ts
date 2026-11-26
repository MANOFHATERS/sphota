"use client";
import { useEffect, useState, useSyncExternalStore } from "react";
import { isDemo } from "./demo";
import { useJob } from "./store";
import { watchJob } from "./api";
import type { DoneEvent } from "./types";

// Mount-safe hooks: nothing here reads window while the server HTML is being matched, so hydration never differs.
// `ready` is true once the client has mounted (and, in demo mode, once the demo fallback has loaded), so pages
// never flash an empty state.
const noopSubscribe = () => () => {};
export const useMounted = () => useSyncExternalStore(noopSubscribe, () => true, () => false);

// Brings back the session saved by the store (after a refresh or a typed address). True once that has been tried.
function useRestored(): boolean {
  const mounted = useMounted();
  const [restored, setRestored] = useState(false);
  useEffect(() => {
    if (!mounted) return;
    useJob.getState().restore();
    setRestored(true);
  }, [mounted]);
  return restored;
}

// The finished job from the store; in demo mode, falls back to the last demo event so Trust works on its own.
// `waiting` is true while a session exists whose first decode has not finished: this hook listens to it, so the result arrives
// even when the user left the Compress page before the decode ended.
export function useDone(): { done?: DoneEvent; ready: boolean; waiting: boolean } {
  const done = useJob((s) => s.done);
  const jobId = useJob((s) => s.result?.job_id);
  const restored = useRestored();
  const mounted = useMounted();
  const [demo, setDemo] = useState<DoneEvent>();
  const [fetched, setFetched] = useState(false);
  const demoMode = mounted && isDemo();
  useEffect(() => {
    if (done || !demoMode) return;
    let alive = true;
    fetch("/demo/events.json")
      .then((r) => r.json())
      .then((ev: DoneEvent[]) => alive && setDemo(ev[ev.length - 1]))
      .finally(() => alive && setFetched(true));
    return () => { alive = false; };
  }, [done, demoMode]);
  useEffect(() => {
    if (!restored || done || !jobId || demoMode) return;
    return watchJob(jobId, (e) => {
      const s = useJob.getState();
      if (s.result?.job_id !== jobId) return;                    // a newer photo replaced this session
      if (e.stage === "denoise") s.set({ denoise: { step: e.step, total: e.total, preview_url: e.preview_url } });
      else if (e.stage === "done") s.set({ done: e });
    });
  }, [restored, done, jobId, demoMode]);
  return { done: done ?? demo, ready: mounted && restored && (!!done || !demoMode || fetched), waiting: !done && !!jobId && !demoMode };
}

export function useJobId(): { id?: string; ready: boolean } {
  const id = useJob((s) => s.result?.job_id);
  const mounted = useMounted();
  const restored = useRestored();
  return { id: id ?? (mounted && isDemo() ? "demo-001" : undefined), ready: mounted && restored };
}
