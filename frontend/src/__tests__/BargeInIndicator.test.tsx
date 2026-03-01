import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { CallStore } from "../stores/callStore";
import { BargeInIndicator } from "../components/BargeInIndicator";
import type {
  TranscriptEvent,
  TranslationEvent,
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

function makeBargeIn(): BargeInEvent {
  return {
    event: "barge_in",
    call_id: "test",
    timestamp_utc: new Date().toISOString(),
    priority: "high",
    reason: "caller_speech_detected_during_playback",
  };
}

describe("BargeInIndicator", () => {
  let store: CallStore;

  beforeEach(() => {
    vi.useFakeTimers();
    store = new CallStore();
  });

  afterEach(() => {
    store.reset();
    vi.useRealTimers();
  });

  it("shows idle state by default", () => {
    render(<BargeInIndicator store={store} />);
    expect(screen.getByText("Listening...")).toBeInTheDocument();
    expect(screen.getByTestId("barge-in-indicator")).toHaveAttribute(
      "data-barge-state",
      "idle"
    );
  });

  it("shows speaking state after translation event", () => {
    store.dispatch(makeTranscript({ seq: 0 }));
    store.dispatch(makeTranslation({ seq: 0 }));
    render(<BargeInIndicator store={store} />);
    expect(screen.getByText("AI Speaking...")).toBeInTheDocument();
    expect(screen.getByTestId("barge-in-indicator")).toHaveAttribute(
      "data-barge-state",
      "speaking"
    );
  });

  it("shows interrupted state after barge_in event", () => {
    store.dispatch(makeBargeIn());
    render(<BargeInIndicator store={store} />);
    expect(screen.getByText("Caller Interrupted!")).toBeInTheDocument();
    expect(screen.getByTestId("barge-in-indicator")).toHaveAttribute(
      "data-barge-state",
      "interrupted"
    );
  });

  it("auto-clears interrupted state after 3 seconds", () => {
    store.dispatch(makeBargeIn());
    render(<BargeInIndicator store={store} />);
    expect(screen.getByText("Caller Interrupted!")).toBeInTheDocument();

    act(() => { vi.advanceTimersByTime(3000); });
    expect(screen.getByText("Listening...")).toBeInTheDocument();
  });
});
