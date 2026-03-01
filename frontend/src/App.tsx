import { useMemo, useState, useEffect, useCallback } from "react";
import { CallStore, useConnectionStatus } from "./stores/callStore";
import { useEmergencyCall } from "./hooks/useEmergencyCall";
import { useDemoMode } from "./hooks/useDemoMode";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { UrgencyPanel } from "./components/UrgencyPanel";
import { BargeInIndicator } from "./components/BargeInIndicator";

function getInitialCallId(): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("call_id");
}

function isExplicitDemo(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).has("demo");
}

function ConnectionDot({ status }: { status: string }) {
  const color =
    status === "connected"
      ? "bg-green-500"
      : status === "connecting" || status === "reconnecting"
        ? "bg-amber-500 animate-pulse"
        : "bg-red-500";
  return <span className={`inline-block h-2 w-2 rounded-full ${color}`} />;
}

// -- Live Dashboard (connected to a real call) --

function Dashboard({ callId, store }: { callId: string; store: CallStore }) {
  const { connectionStatus, audioRef, endCall } = useEmergencyCall(callId, store);
  const storeConnectionStatus = useConnectionStatus(store);
  const displayStatus = connectionStatus || storeConnectionStatus;

  return (
    <div className="flex h-screen flex-col bg-dispatch-950 text-dispatch-100">
      <header className="flex items-center justify-between border-b border-dispatch-700 px-6 py-3">
        <div className="flex items-center gap-4">
          <h1 className="text-sm font-black uppercase tracking-[0.2em] text-dispatch-300">
            911 Dispatch
          </h1>
          <span className="rounded bg-dispatch-800 px-2 py-0.5 text-[11px] font-mono tabular-nums text-dispatch-400">
            {callId}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <ConnectionDot status={displayStatus} />
            <span className="text-[11px] uppercase tracking-wider text-dispatch-400">
              {displayStatus}
            </span>
          </div>
          <button
            onClick={endCall}
            className="rounded-lg bg-stress-high/20 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-stress-high ring-1 ring-stress-high/30 transition-colors hover:bg-stress-high/30"
          >
            End Call
          </button>
        </div>
      </header>

      <main className="flex flex-1 gap-4 overflow-hidden p-4">
        <div className="flex w-72 shrink-0 flex-col gap-4">
          <UrgencyPanel store={store} />
          <BargeInIndicator store={store} />
        </div>
        <div className="flex-1">
          <TranscriptPanel store={store} />
        </div>
      </main>

      <audio ref={audioRef} autoPlay className="hidden" />
    </div>
  );
}

// -- Demo Dashboard (scripted simulation) --

function DemoDashboard({ store }: { store: CallStore }) {
  useDemoMode(store);

  return (
    <div className="flex h-screen flex-col bg-dispatch-950 text-dispatch-100">
      <header className="flex items-center justify-between border-b border-dispatch-700 px-6 py-3">
        <div className="flex items-center gap-4">
          <h1 className="text-sm font-black uppercase tracking-[0.2em] text-dispatch-300">
            911 Dispatch
          </h1>
          <span className="rounded bg-amber-900/40 px-2 py-0.5 text-[11px] font-mono font-bold text-amber-400">
            DEMO MODE
          </span>
        </div>
        <div className="flex items-center gap-2">
          <ConnectionDot status="connected" />
          <span className="text-[11px] uppercase tracking-wider text-dispatch-400">
            simulated
          </span>
        </div>
      </header>

      <main className="flex flex-1 gap-4 overflow-hidden p-4">
        <div className="flex w-72 shrink-0 flex-col gap-4">
          <UrgencyPanel store={store} />
          <BargeInIndicator store={store} />
        </div>
        <div className="flex-1">
          <TranscriptPanel store={store} />
        </div>
      </main>
    </div>
  );
}

// -- Lobby (polls for incoming calls, auto-connects) --

const POLL_INTERVAL_MS = 2000;

function Lobby({ onCallDetected }: { onCallDetected: (callId: string) => void }) {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch("/api/active-calls");
        if (cancelled) return;
        if (!res.ok) {
          setBackendOnline(false);
          return;
        }
        setBackendOnline(true);
        const data: { calls: string[] } = await res.json();
        if (data.calls.length > 0) {
          onCallDetected(data.calls[0]);
        }
      } catch {
        if (!cancelled) setBackendOnline(false);
      }
    };

    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [onCallDetected]);

  return (
    <div className="flex h-screen flex-col items-center justify-center bg-dispatch-950 text-dispatch-100">
      <div className="mb-8 text-center">
        <h1 className="mb-2 text-lg font-black uppercase tracking-[0.25em] text-dispatch-300">
          911 Dispatch
        </h1>
        <p className="text-sm text-dispatch-500">Waiting for incoming call...</p>
      </div>

      <div className="mb-8 flex flex-col items-center gap-3">
        <div className="relative flex h-20 w-20 items-center justify-center">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent/20" />
          <span className="relative inline-flex h-12 w-12 items-center justify-center rounded-full bg-accent/30">
            <svg viewBox="0 0 24 24" fill="currentColor" className="h-6 w-6 text-accent">
              <path d="M6.62 10.79a15.053 15.053 0 006.59 6.59l2.2-2.2a1 1 0 011.01-.24c1.12.37 2.33.57 3.58.57a1 1 0 011 1V20a1 1 0 01-1 1A17 17 0 013 4a1 1 0 011-1h3.5a1 1 0 011 1c0 1.25.2 2.46.57 3.58a1 1 0 01-.24 1.01l-2.2 2.2z" />
            </svg>
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              backendOnline === true
                ? "bg-green-500"
                : backendOnline === false
                  ? "bg-red-500"
                  : "bg-dispatch-500 animate-pulse"
            }`}
          />
          <span className="text-[11px] uppercase tracking-wider text-dispatch-400">
            {backendOnline === true
              ? "Backend connected — polling"
              : backendOnline === false
                ? "Backend offline"
                : "Checking..."}
          </span>
        </div>
      </div>

      <a
        href="?demo"
        className="rounded-lg bg-dispatch-800 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-dispatch-400 ring-1 ring-dispatch-600 transition-colors hover:bg-dispatch-700 hover:text-dispatch-200"
      >
        Run Demo Instead
      </a>
    </div>
  );
}

// -- App root --

export default function App() {
  const store = useMemo(() => new CallStore(), []);
  const urlCallId = useMemo(getInitialCallId, []);
  const demo = useMemo(isExplicitDemo, []);

  const [activeCallId, setActiveCallId] = useState<string | null>(urlCallId);

  const handleCallDetected = useCallback((callId: string) => {
    setActiveCallId(callId);
    window.history.replaceState(null, "", `?call_id=${encodeURIComponent(callId)}`);
  }, []);

  if (demo) {
    return <DemoDashboard store={store} />;
  }

  if (activeCallId) {
    return <Dashboard callId={activeCallId} store={store} />;
  }

  return <Lobby onCallDetected={handleCallDetected} />;
}
