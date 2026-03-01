import { useSyncExternalStore, useRef, useCallback } from "react";
import type {
  CallState,
  ServerEvent,
  TranscriptEntry,
  AnalyticsSnapshot,
  BargeInState,
  StressLevel,
  DetectedKeyword,
  ProsodySnapshot,
  ConnectionStatus,
} from "../types/events";

const INITIAL_ANALYTICS: AnalyticsSnapshot = {
  stressLevel: "low",
  stressConfidence: 0,
  prosody: {
    pitch_mean_hz: null,
    pitch_std_hz: null,
    rms_energy_db: null,
    mfcc_1_mean: null,
  },
  keywords: [],
};

function createInitialState(): CallState {
  return {
    transcripts: [],
    analytics: { ...INITIAL_ANALYTICS },
    bargeIn: "idle",
    connectionStatus: "disconnected",
  };
}

const PARTIAL_FLUSH_MS = 150;
const BARGE_IN_CLEAR_MS = 3000;
const SPEAKING_TIMEOUT_MS = 10000;

export class CallStore {
  private _state: CallState;
  private _listeners = new Set<() => void>();

  private _pendingPartialSeq: number | null = null;
  private _partialTimer: ReturnType<typeof setTimeout> | null = null;
  private _bargeInTimer: ReturnType<typeof setTimeout> | null = null;
  private _speakingTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this._state = createInitialState();
  }

  getSnapshot = (): CallState => this._state;

  subscribe = (listener: () => void): (() => void) => {
    this._listeners.add(listener);
    return () => this._listeners.delete(listener);
  };

  private _notify(): void {
    this._state = { ...this._state };
    for (const l of this._listeners) l();
  }

  setConnectionStatus(status: ConnectionStatus): void {
    if (this._state.connectionStatus === status) return;
    this._state = { ...this._state, connectionStatus: status };
    this._notify();
  }

  dispatch(event: ServerEvent): void {
    switch (event.event) {
      case "transcript":
        this._handleTranscript(event);
        break;
      case "translation":
        this._handleTranslation(event);
        break;
      case "analytics":
        this._handleAnalytics(event);
        break;
      case "barge_in":
        this._handleBargeIn();
        break;
    }
  }

  private _handleTranscript(event: ServerEvent & { event: "transcript" }): void {
    const entry: TranscriptEntry = {
      seq: event.seq,
      original: event.original_transcript,
      translated: null,
      language: event.detected_language,
      confidence: event.asr_confidence,
      isFinal: event.is_final,
      timestamp: event.timestamp_utc,
    };

    if (event.is_final) {
      this._clearPartialTimer();

      if (this._pendingPartialSeq === event.seq) {
        const newTranscripts = [...this._state.transcripts];
        newTranscripts[newTranscripts.length - 1] = entry;
        this._state = { ...this._state, transcripts: newTranscripts };
      } else {
        this._state = {
          ...this._state,
          transcripts: [...this._state.transcripts, entry],
        };
      }
      this._pendingPartialSeq = null;

      if (this._state.bargeIn === "speaking") {
        this._clearSpeakingTimer();
        this._state = { ...this._state, bargeIn: "idle" };
      }

      this._notify();
    } else {
      if (this._pendingPartialSeq === event.seq) {
        const newTranscripts = [...this._state.transcripts];
        newTranscripts[newTranscripts.length - 1] = entry;
        this._state = { ...this._state, transcripts: newTranscripts };
      } else {
        this._pendingPartialSeq = event.seq;
        this._state = {
          ...this._state,
          transcripts: [...this._state.transcripts, entry],
        };
      }

      if (!this._partialTimer) {
        this._partialTimer = setTimeout(() => {
          this._partialTimer = null;
          this._notify();
        }, PARTIAL_FLUSH_MS);
      }
    }
  }

  private _handleTranslation(
    event: ServerEvent & { event: "translation" }
  ): void {
    const idx = this._state.transcripts.findIndex((t) => t.seq === event.seq);
    if (idx === -1) return;

    const newTranscripts = [...this._state.transcripts];
    newTranscripts[idx] = {
      ...newTranscripts[idx],
      translated: event.translated_text,
    };
    this._state = { ...this._state, transcripts: newTranscripts };

    this._clearSpeakingTimer();
    this._speakingTimer = setTimeout(() => {
      this._speakingTimer = null;
      if (this._state.bargeIn === "speaking") {
        this._state = { ...this._state, bargeIn: "idle" };
        this._notify();
      }
    }, SPEAKING_TIMEOUT_MS);
    this._state = { ...this._state, bargeIn: "speaking" };

    this._notify();
  }

  private _handleAnalytics(
    event: ServerEvent & { event: "analytics" }
  ): void {
    const prev = this._state.analytics;
    const levelChanged = prev.stressLevel !== event.stress_level;
    const newKeywords = event.keywords_detected.length > 0;

    const analytics: AnalyticsSnapshot = {
      stressLevel: event.stress_level,
      stressConfidence: event.stress_confidence,
      prosody: event.prosody,
      keywords: newKeywords
        ? [...prev.keywords, ...event.keywords_detected]
        : prev.keywords,
    };

    this._state = { ...this._state, analytics };

    if (levelChanged || newKeywords) {
      this._notify();
    }
  }

  private _handleBargeIn(): void {
    this._clearSpeakingTimer();
    this._clearBargeInTimer();

    this._state = { ...this._state, bargeIn: "interrupted" };
    this._notify();

    this._bargeInTimer = setTimeout(() => {
      this._bargeInTimer = null;
      this._state = { ...this._state, bargeIn: "idle" };
      this._notify();
    }, BARGE_IN_CLEAR_MS);
  }

  private _clearPartialTimer(): void {
    if (this._partialTimer) {
      clearTimeout(this._partialTimer);
      this._partialTimer = null;
    }
  }

  private _clearBargeInTimer(): void {
    if (this._bargeInTimer) {
      clearTimeout(this._bargeInTimer);
      this._bargeInTimer = null;
    }
  }

  private _clearSpeakingTimer(): void {
    if (this._speakingTimer) {
      clearTimeout(this._speakingTimer);
      this._speakingTimer = null;
    }
  }

  reset(): void {
    this._clearPartialTimer();
    this._clearBargeInTimer();
    this._clearSpeakingTimer();
    this._pendingPartialSeq = null;
    this._state = createInitialState();
    this._notify();
  }

  dismissKeyword(keyword: string): void {
    const filtered = this._state.analytics.keywords.filter(
      (k) => k.keyword !== keyword
    );
    if (filtered.length === this._state.analytics.keywords.length) return;
    this._state = {
      ...this._state,
      analytics: { ...this._state.analytics, keywords: filtered },
    };
    this._notify();
  }
}

// --- Selector hooks ---

export function useTranscripts(store: CallStore): TranscriptEntry[] {
  return useSyncExternalStore(store.subscribe, () => store.getSnapshot().transcripts);
}

export function useStressLevel(
  store: CallStore
): { level: StressLevel; confidence: number } {
  const selectRef = useRef<{ level: StressLevel; confidence: number } | null>(null);

  const getSnapshot = useCallback(() => {
    const { stressLevel, stressConfidence } = store.getSnapshot().analytics;
    const prev = selectRef.current;
    if (prev && prev.level === stressLevel && prev.confidence === stressConfidence) {
      return prev;
    }
    const next = { level: stressLevel, confidence: stressConfidence };
    selectRef.current = next;
    return next;
  }, [store]);

  return useSyncExternalStore(store.subscribe, getSnapshot);
}

export function useKeywords(store: CallStore): DetectedKeyword[] {
  return useSyncExternalStore(
    store.subscribe,
    () => store.getSnapshot().analytics.keywords
  );
}

export function useProsody(store: CallStore): ProsodySnapshot {
  return useSyncExternalStore(
    store.subscribe,
    () => store.getSnapshot().analytics.prosody
  );
}

export function useBargeInState(store: CallStore): BargeInState {
  return useSyncExternalStore(
    store.subscribe,
    () => store.getSnapshot().bargeIn
  );
}

export function useConnectionStatus(store: CallStore): ConnectionStatus {
  return useSyncExternalStore(
    store.subscribe,
    () => store.getSnapshot().connectionStatus
  );
}
