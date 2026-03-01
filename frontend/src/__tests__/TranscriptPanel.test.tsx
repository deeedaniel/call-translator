import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { CallStore } from "../stores/callStore";
import { TranscriptPanel } from "../components/TranscriptPanel";
import type { TranscriptEvent, TranslationEvent } from "../types/events";

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

describe("TranscriptPanel", () => {
  let store: CallStore;

  beforeEach(() => {
    vi.useFakeTimers();
    store = new CallStore();
  });

  afterEach(() => {
    store.reset();
    vi.useRealTimers();
  });

  it("shows empty state when no transcripts", () => {
    render(<TranscriptPanel store={store} />);
    expect(screen.getByText("Waiting for caller audio...")).toBeInTheDocument();
  });

  it("renders a final transcript entry", () => {
    store.dispatch(
      makeTranscript({ seq: 0, original_transcript: "Necesito ayuda" })
    );
    render(<TranscriptPanel store={store} />);
    expect(screen.getByText("Necesito ayuda")).toBeInTheDocument();
    expect(screen.getByText("ES")).toBeInTheDocument();
  });

  it("renders translated text when available", () => {
    store.dispatch(
      makeTranscript({ seq: 0, original_transcript: "Hola" })
    );
    store.dispatch(
      makeTranslation({ seq: 0, translated_text: "Hello" })
    );
    render(<TranscriptPanel store={store} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });

  it("shows segment count", () => {
    store.dispatch(makeTranscript({ seq: 0 }));
    store.dispatch(makeTranscript({ seq: 1 }));
    render(<TranscriptPanel store={store} />);
    expect(screen.getByText("2 segments")).toBeInTheDocument();
  });
});
