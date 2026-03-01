import { memo } from "react";
import type { CallStore } from "../stores/callStore";
import { useBargeInState } from "../stores/callStore";

function PulseWave() {
  return (
    <span className="inline-flex items-center gap-0.5" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="inline-block h-3 w-0.5 rounded-full bg-accent"
          style={{
            animation: `pulse-stress 0.8s ease-in-out ${i * 0.15}s infinite`,
          }}
        />
      ))}
    </span>
  );
}

interface BargeInIndicatorProps {
  store: CallStore;
}

export const BargeInIndicator = memo(function BargeInIndicator({
  store,
}: BargeInIndicatorProps) {
  const state = useBargeInState(store);

  return (
    <div
      data-testid="barge-in-indicator"
      data-barge-state={state}
      className={`flex items-center gap-3 rounded-xl border px-4 py-3 transition-colors duration-200 ${
        state === "interrupted"
          ? "animate-shake border-stress-high/60 bg-stress-high/15"
          : state === "speaking"
            ? "border-accent/40 bg-accent/10"
            : "border-dispatch-700 bg-dispatch-900"
      }`}
    >
      {state === "idle" && (
        <>
          <span className="inline-block h-2 w-2 rounded-full bg-dispatch-500" />
          <span className="text-xs text-dispatch-500">Listening...</span>
        </>
      )}

      {state === "speaking" && (
        <>
          <PulseWave />
          <span className="text-xs font-semibold text-accent">
            AI Speaking...
          </span>
        </>
      )}

      {state === "interrupted" && (
        <>
          <svg
            viewBox="0 0 16 16"
            fill="currentColor"
            className="h-4 w-4 text-stress-high"
            aria-hidden="true"
          >
            <path
              fillRule="evenodd"
              d="M8 1a7 7 0 100 14A7 7 0 008 1zm-.75 4a.75.75 0 011.5 0v3a.75.75 0 01-1.5 0V5zm.75 6.25a.75.75 0 100-1.5.75.75 0 000 1.5z"
              clipRule="evenodd"
            />
          </svg>
          <span className="text-xs font-bold text-stress-high">
            Caller Interrupted!
          </span>
        </>
      )}
    </div>
  );
});
