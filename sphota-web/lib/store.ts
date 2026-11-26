// The current job, shared by all four app pages (guide F15).
import { create } from "zustand";
import type { CompressResult, DoneEvent, Mode } from "./types";

export type StageState = { status: "waiting" | "running" | "done"; ms?: number };

type State = {
  file?: File;
  previewUrl?: string;
  result?: CompressResult;
  done?: DoneEvent;
  denoise?: { step: number; total: number; preview_url: string };
  stages: Record<string, StageState>;
  presetMode?: Mode;
  set: (p: Partial<State>) => void;
  mark: (stage: string, status: "running" | "done", ms?: number) => void;
  reset: () => void;
  restore: () => void;
};

// The session's result and its finished decode survive a refresh / a typed address (sessionStorage, this tab only). The server keeps
// the session for hours, so Compare, Trust and Explain keep working. Nothing is read during server rendering.
const KEY = "sphota-job";

export const useJob = create<State>((set, get) => ({
  stages: {},
  set: (p) => set(p),
  mark: (stage, status, ms) => set((s) => ({ stages: { ...s.stages, [stage]: { status, ms } } })),
  reset: () => set({ result: undefined, done: undefined, denoise: undefined, stages: {} }),
  restore: () => {
    if (get().result) return;
    try {
      const raw = sessionStorage.getItem(KEY);
      if (raw) {
        const p = JSON.parse(raw) as { result?: CompressResult; done?: DoneEvent };
        if (p.result) set({ result: p.result, done: p.done });
      }
    } catch {
      /* private mode or bad JSON: start empty */
    }
  },
}));

if (typeof window !== "undefined") {
  useJob.subscribe((s, prev) => {
    if (s.result === prev.result && s.done === prev.done) return;
    try {
      sessionStorage.setItem(KEY, JSON.stringify({ result: s.result, done: s.done }));
    } catch {
      /* storage full or blocked: the in-memory state still works */
    }
  });
}
