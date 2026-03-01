import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { CallStore } from "../stores/callStore";
import type {
  TranscriptEvent,
  TranslationEvent,
  AnalyticsEvent,
  BargeInEvent,
} from "../types/events";

function makeTranscript(overrides: Partial<TranscriptEvent> = {}): TranscriptEvent {
  return {
    event: "transcript",
    call_id: "test",
    timestamp_utc: new Date().toISOString(),
    seq: 0,
    original_transcript: "Hola",
    detected_language: "es",
    asr_confidence: 0.9,
    word_timestamps: [],
    is_final: true,
    ...overrides,
  };
}

function makeTranslation(overrides: Partial<TranslationEvent> = {}): TranslationEvent {
  return {
    event: "translation",
    call_id: "test",
    timestamp_utc: new Date().toISOString(),
    seq: 0,
    source_text: "Hola",
    translated_text: "Hello",
    source_language: "es",
    target_language: "en",
    translation_model: "riva",
    ...overrides,
  };
}

function makeAnalytics(overrides: Partial<AnalyticsEvent> = {}): AnalyticsEvent {
  return {
    event: "analytics",
    call_id: "test",
    timestamp_utc: new Date().toISOString(),
    seq: 0,
    stress_level: "low",
    stress_confidence: 0.7,
    prosody: {
      pitch_mean_hz: null,
      pitch_std_hz: null,
      rms_energy_db: null,
      mfcc_1_mean: null,
    },
    keywords_detected: [],
    ...overrides,
  };
}

function makeBargeIn(): BargeInEvent {
  return {
    event: "barge_in",
    call_id: "test",
    timestamp_utc: new Date().toISOString(),
    priority: "high",
    reason: "caller_speech_detected_during_playback",
  };
}

describe("CallStore", () => {
  let store: CallStore;

  beforeEach(() => {
    vi.useFakeTimers();
    store = new CallStore();
  });

  afterEach(() => {
    store.reset();
    vi.useRealTimers();
  });

  it("starts with empty initial state", () => {
    const state = store.getSnapshot();
    expect(state.transcripts).toEqual([]);
    expect(state.analytics.stressLevel).toBe("low");
    expect(state.bargeIn).toBe("idle");
    expect(state.connectionStatus).toBe("disconnected");
  });

  describe("transcript handling", () => {
    it("appends final transcripts immediately", () => {
      const listener = vi.fn();
      store.subscribe(listener);

      store.dispatch(makeTranscript({ seq: 0, original_transcript: "Hola" }));

      expect(store.getSnapshot().transcripts).toHaveLength(1);
      expect(store.getSnapshot().transcripts[0].original).toBe("Hola");
      expect(store.getSnapshot().transcripts[0].isFinal).toBe(true);
      expect(listener).toHaveBeenCalledTimes(1);
    });

    it("batches partial transcripts with 150ms timer", () => {
      const listener = vi.fn();
      store.subscribe(listener);

      store.dispatch(
        makeTranscript({ seq: 1, is_final: false, original_transcript: "Ho" })
      );
      expect(listener).not.toHaveBeenCalled();

      vi.advanceTimersByTime(150);
      expect(listener).toHaveBeenCalledTimes(1);
      expect(store.getSnapshot().transcripts).toHaveLength(1);
      expect(store.getSnapshot().transcripts[0].isFinal).toBe(false);
    });

    it("finalizes a pending partial in-place", () => {
      store.dispatch(
        makeTranscript({ seq: 5, is_final: false, original_transcript: "Hol" })
      );
      vi.advanceTimersByTime(150);

      store.dispatch(
        makeTranscript({ seq: 5, is_final: true, original_transcript: "Hola" })
      );

      expect(store.getSnapshot().transcripts).toHaveLength(1);
      expect(store.getSnapshot().transcripts[0].original).toBe("Hola");
      expect(store.getSnapshot().transcripts[0].isFinal).toBe(true);
    });
  });

  describe("translation handling", () => {
    it("sets translated field on matching seq", () => {
      store.dispatch(makeTranscript({ seq: 0 }));
      store.dispatch(makeTranslation({ seq: 0, translated_text: "Hello" }));

      expect(store.getSnapshot().transcripts[0].translated).toBe("Hello");
    });

    it("ignores translation with no matching seq", () => {
      store.dispatch(makeTranscript({ seq: 0 }));
      const listener = vi.fn();
      store.subscribe(listener);

      store.dispatch(makeTranslation({ seq: 99 }));
      expect(listener).not.toHaveBeenCalled();
    });

    it("transitions barge-in state to speaking", () => {
      store.dispatch(makeTranscript({ seq: 0 }));
      store.dispatch(makeTranslation({ seq: 0 }));

      expect(store.getSnapshot().bargeIn).toBe("speaking");
    });
  });

  describe("analytics handling", () => {
    it("notifies when stress level changes", () => {
      const listener = vi.fn();
      store.subscribe(listener);

      store.dispatch(makeAnalytics({ stress_level: "high" }));
      expect(listener).toHaveBeenCalledTimes(1);
      expect(store.getSnapshot().analytics.stressLevel).toBe("high");
    });

    it("does not notify when stress level stays the same and no keywords", () => {
      store.dispatch(makeAnalytics({ stress_level: "low" }));
      const listener = vi.fn();
      store.subscribe(listener);

      store.dispatch(makeAnalytics({ stress_level: "low", seq: 1 }));
      expect(listener).not.toHaveBeenCalled();
    });

    it("notifies when new keywords are detected", () => {
      const listener = vi.fn();
      store.subscribe(listener);

      store.dispatch(
        makeAnalytics({
          keywords_detected: [
            {
              keyword: "fire",
              detected_at_utc: new Date().toISOString(),
              source: "transcript",
              transcript_seq: 0,
            },
          ],
        })
      );
      expect(listener).toHaveBeenCalledTimes(1);
      expect(store.getSnapshot().analytics.keywords).toHaveLength(1);
    });

    it("accumulates keywords across events", () => {
      store.dispatch(
        makeAnalytics({
          keywords_detected: [
            {
              keyword: "fire",
              detected_at_utc: new Date().toISOString(),
              source: "transcript",
              transcript_seq: 0,
            },
          ],
        })
      );
      store.dispatch(
        makeAnalytics({
          seq: 1,
          stress_level: "high",
          keywords_detected: [
            {
              keyword: "gun",
              detected_at_utc: new Date().toISOString(),
              source: "transcript",
              transcript_seq: 1,
            },
          ],
        })
      );

      expect(store.getSnapshot().analytics.keywords).toHaveLength(2);
    });
  });

  describe("barge-in handling", () => {
    it("sets state to interrupted on barge_in event", () => {
      store.dispatch(makeBargeIn());
      expect(store.getSnapshot().bargeIn).toBe("interrupted");
    });

    it("auto-clears interrupted state after 3 seconds", () => {
      store.dispatch(makeBargeIn());
      expect(store.getSnapshot().bargeIn).toBe("interrupted");

      vi.advanceTimersByTime(3000);
      expect(store.getSnapshot().bargeIn).toBe("idle");
    });

    it("clears speaking state when barge_in arrives", () => {
      store.dispatch(makeTranscript({ seq: 0 }));
      store.dispatch(makeTranslation({ seq: 0 }));
      expect(store.getSnapshot().bargeIn).toBe("speaking");

      store.dispatch(makeBargeIn());
      expect(store.getSnapshot().bargeIn).toBe("interrupted");
    });
  });

  describe("connection status", () => {
    it("updates connection status", () => {
      store.setConnectionStatus("connected");
      expect(store.getSnapshot().connectionStatus).toBe("connected");
    });

    it("does not notify when status unchanged", () => {
      const listener = vi.fn();
      store.subscribe(listener);
      store.setConnectionStatus("disconnected");
      expect(listener).not.toHaveBeenCalled();
    });
  });
});
