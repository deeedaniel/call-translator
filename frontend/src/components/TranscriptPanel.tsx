import { memo, useRef, useEffect, useState, useCallback } from "react";
import type { CallStore } from "../stores/callStore";
import { useTranscripts } from "../stores/callStore";
import type { TranscriptEntry } from "../types/events";

const LANGUAGE_LABELS: Record<string, string> = {
  es: "ES",
  fr: "FR",
  de: "DE",
  zh: "ZH",
  ja: "JA",
  ko: "KO",
  ar: "AR",
  pt: "PT",
  ru: "RU",
  hi: "HI",
  en: "EN",
};

function ConfidenceDot({ confidence }: { confidence: number }) {
  const color =
    confidence >= 0.85
      ? "bg-green-500"
      : confidence >= 0.6
        ? "bg-amber-500"
        : "bg-red-500";
  return (
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full ${color}`}
      title={`${Math.round(confidence * 100)}% confidence`}
    />
  );
}

function TranscriptRow({ entry }: { entry: TranscriptEntry }) {
  const langLabel = LANGUAGE_LABELS[entry.language] ?? entry.language.toUpperCase();

  return (
    <div
      className={`grid grid-cols-2 gap-4 rounded-lg px-4 py-3 transition-opacity duration-200 ${
        entry.isFinal
          ? "bg-dispatch-800/60"
          : "bg-dispatch-800/30 opacity-60"
      }`}
    >
      <div className="min-w-0">
        <div className="mb-1 flex items-center gap-2">
          <span className="inline-flex items-center rounded bg-dispatch-600 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-dispatch-200">
            {langLabel}
          </span>
          <ConfidenceDot confidence={entry.confidence} />
        </div>
        <p className="break-words text-sm leading-relaxed text-dispatch-100">
          {entry.original}
          {!entry.isFinal && (
            <span className="ml-1 text-dispatch-400 animate-pulse">...</span>
          )}
        </p>
      </div>

      <div className="min-w-0 border-l border-dispatch-700 pl-4">
        <div className="mb-1">
          <span className="inline-flex items-center rounded bg-accent/20 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-accent">
            EN
          </span>
        </div>
        {entry.translated ? (
          <p className="break-words text-sm leading-relaxed text-dispatch-100">
            {entry.translated}
          </p>
        ) : (
          <div className="h-4 w-3/4 rounded animate-shimmer bg-dispatch-700/50" />
        )}
      </div>
    </div>
  );
}

interface TranscriptPanelProps {
  store: CallStore;
}

export const TranscriptPanel = memo(function TranscriptPanel({
  store,
}: TranscriptPanelProps) {
  const transcripts = useTranscripts(store);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isAutoScroll, setIsAutoScroll] = useState(true);
  const isUserScrollingRef = useRef(false);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    isUserScrollingRef.current = !atBottom;
    setIsAutoScroll(atBottom);
  }, []);

  useEffect(() => {
    if (isAutoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcripts, isAutoScroll]);

  const jumpToLatest = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      setIsAutoScroll(true);
    }
  }, []);

  return (
    <div className="relative flex h-full flex-col overflow-hidden rounded-xl border border-dispatch-700 bg-dispatch-900">
      <div className="flex items-center border-b border-dispatch-700 px-4 py-2.5">
        <h2 className="text-xs font-bold uppercase tracking-widest text-dispatch-400">
          Live Transcript
        </h2>
        <span className="ml-auto text-[10px] tabular-nums text-dispatch-500">
          {transcripts.length} segments
        </span>
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 space-y-2 overflow-y-auto p-3"
      >
        {transcripts.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-dispatch-500">
              Waiting for caller audio...
            </p>
          </div>
        ) : (
          transcripts.map((entry) => (
            <TranscriptRow key={`${entry.seq}-${entry.isFinal}`} entry={entry} />
          ))
        )}
      </div>

      {!isAutoScroll && transcripts.length > 0 && (
        <button
          onClick={jumpToLatest}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-accent px-4 py-1.5 text-xs font-semibold text-white shadow-lg transition-transform hover:scale-105"
        >
          Jump to latest
        </button>
      )}
    </div>
  );
});
