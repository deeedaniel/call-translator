import { memo, useEffect, useRef, useCallback } from "react";
import type { CallStore } from "../stores/callStore";
import {
  useStressLevel,
  useKeywords,
  useProsody,
} from "../stores/callStore";
import type { DetectedKeyword, ProsodySnapshot, StressLevel } from "../types/events";

const STRESS_CONFIG: Record<
  StressLevel,
  { bg: string; border: string; label: string; animation: string }
> = {
  low: {
    bg: "bg-stress-low/15",
    border: "border-stress-low/40",
    label: "LOW",
    animation: "",
  },
  medium: {
    bg: "bg-stress-medium/20",
    border: "border-stress-medium/50",
    label: "MEDIUM",
    animation: "animate-pulse-stress",
  },
  high: {
    bg: "bg-stress-high/25",
    border: "border-stress-high/60",
    label: "HIGH",
    animation: "animate-pulse-stress-high",
  },
};

const STRESS_TEXT_COLOR: Record<StressLevel, string> = {
  low: "text-stress-low",
  medium: "text-stress-medium",
  high: "text-stress-high",
};

const DESTRUCTIVE_KEYWORDS = new Set([
  "gun",
  "fire",
  "bomb",
  "knife",
  "weapon",
  "shoot",
  "shooting",
  "explosion",
  "stabbing",
  "blood",
  "dying",
  "dead",
  "kill",
]);

// --- StressBadge ---

const StressBadge = memo(function StressBadge({
  store,
}: {
  store: CallStore;
}) {
  const { level, confidence } = useStressLevel(store);
  const cfg = STRESS_CONFIG[level];

  return (
    <div
      data-testid="stress-badge"
      data-stress-level={level}
      className={`rounded-xl border-2 p-5 text-center transition-colors duration-300 ${cfg.bg} ${cfg.border} ${cfg.animation}`}
    >
      <div className="mb-1 text-[10px] font-bold uppercase tracking-widest text-dispatch-400">
        Caller Stress
      </div>
      <div
        className={`text-3xl font-black tracking-tight ${STRESS_TEXT_COLOR[level]}`}
      >
        {cfg.label}
      </div>
      <div className="mt-1 text-xs tabular-nums text-dispatch-400">
        {Math.round(confidence * 100)}% confidence
      </div>
    </div>
  );
});

// --- KeywordAlerts ---

const KEYWORD_DISMISS_MS = 10_000;

function KeywordPill({
  kw,
  onDismiss,
}: {
  kw: DetectedKeyword;
  onDismiss: (keyword: string) => void;
}) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isDestructive = DESTRUCTIVE_KEYWORDS.has(kw.keyword.toLowerCase());

  useEffect(() => {
    timerRef.current = setTimeout(() => onDismiss(kw.keyword), KEYWORD_DISMISS_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [kw.keyword, onDismiss]);

  return (
    <span
      className={`animate-flash-in inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-bold uppercase tracking-wide ${
        isDestructive
          ? "bg-stress-high/25 text-stress-high ring-1 ring-stress-high/40"
          : "bg-stress-medium/25 text-stress-medium ring-1 ring-stress-medium/40"
      }`}
    >
      {isDestructive && (
        <svg
          viewBox="0 0 16 16"
          fill="currentColor"
          className="h-3 w-3"
          aria-hidden="true"
        >
          <path d="M8 1l1.5 5H15l-4 3.5 1.5 5L8 11.5 3.5 14.5l1.5-5L1 6h5.5z" />
        </svg>
      )}
      {kw.keyword}
    </span>
  );
}

const KeywordAlerts = memo(function KeywordAlerts({
  store,
}: {
  store: CallStore;
}) {
  const keywords = useKeywords(store);

  const handleDismiss = useCallback(
    (keyword: string) => store.dismissKeyword(keyword),
    [store]
  );

  if (keywords.length === 0) return null;

  return (
    <div data-testid="keyword-alerts" className="mt-3">
      <div className="mb-2 text-[10px] font-bold uppercase tracking-widest text-dispatch-400">
        Detected Keywords
      </div>
      <div className="flex flex-wrap gap-2">
        {keywords.map((kw, i) => (
          <KeywordPill key={`${kw.keyword}-${i}`} kw={kw} onDismiss={handleDismiss} />
        ))}
      </div>
    </div>
  );
});

// --- ProsodyMeter ---

function meterWidth(
  value: number | null,
  max: number
): string {
  if (value === null) return "0%";
  const pct = Math.min(Math.max((value / max) * 100, 0), 100);
  return `${pct}%`;
}

const ProsodyMeter = memo(function ProsodyMeter({
  store,
}: {
  store: CallStore;
}) {
  const prosody: ProsodySnapshot = useProsody(store);

  return (
    <div className="mt-4 space-y-2" data-testid="prosody-meter">
      <div className="text-[10px] font-bold uppercase tracking-widest text-dispatch-400">
        Prosody
      </div>

      <div>
        <div className="mb-0.5 flex items-center justify-between text-[10px] text-dispatch-500">
          <span>Pitch</span>
          <span className="tabular-nums">
            {prosody.pitch_mean_hz !== null
              ? `${Math.round(prosody.pitch_mean_hz)} Hz`
              : "--"}
          </span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-dispatch-700">
          <div
            className="h-full rounded-full bg-accent transition-all duration-200"
            style={{ width: meterWidth(prosody.pitch_mean_hz, 500) }}
          />
        </div>
      </div>

      <div>
        <div className="mb-0.5 flex items-center justify-between text-[10px] text-dispatch-500">
          <span>Loudness</span>
          <span className="tabular-nums">
            {prosody.rms_energy_db !== null
              ? `${prosody.rms_energy_db.toFixed(1)} dB`
              : "--"}
          </span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-dispatch-700">
          <div
            className="h-full rounded-full bg-stress-medium transition-all duration-200"
            style={{ width: meterWidth(prosody.rms_energy_db, 1.0) }}
          />
        </div>
      </div>
    </div>
  );
});

// --- Composite panel ---

interface UrgencyPanelProps {
  store: CallStore;
}

export const UrgencyPanel = memo(function UrgencyPanel({
  store,
}: UrgencyPanelProps) {
  return (
    <div className="rounded-xl border border-dispatch-700 bg-dispatch-900 p-4">
      <StressBadge store={store} />
      <KeywordAlerts store={store} />
      <ProsodyMeter store={store} />
    </div>
  );
});
