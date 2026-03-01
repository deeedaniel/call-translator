import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { CallStore } from "../stores/callStore";
import { UrgencyPanel } from "../components/UrgencyPanel";
import type { AnalyticsEvent } from "../types/events";

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

describe("UrgencyPanel", () => {
  let store: CallStore;

  beforeEach(() => {
    vi.useFakeTimers();
    store = new CallStore();
  });

  afterEach(() => {
    store.reset();
    vi.useRealTimers();
  });

  it("shows LOW stress by default", () => {
    render(<UrgencyPanel store={store} />);
    const badge = screen.getByTestId("stress-badge");
    expect(badge).toHaveAttribute("data-stress-level", "low");
    expect(screen.getByText("LOW")).toBeInTheDocument();
  });

  it("updates to HIGH when analytics event arrives", () => {
    store.dispatch(makeAnalytics({ stress_level: "high", stress_confidence: 0.92 }));
    render(<UrgencyPanel store={store} />);
    const badge = screen.getByTestId("stress-badge");
    expect(badge).toHaveAttribute("data-stress-level", "high");
    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText("92% confidence")).toBeInTheDocument();
  });

  it("shows keyword pills when keywords detected", () => {
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
    render(<UrgencyPanel store={store} />);
    expect(screen.getByText("fire")).toBeInTheDocument();
  });

  it("auto-dismisses keywords after 10 seconds", () => {
    store.dispatch(
      makeAnalytics({
        keywords_detected: [
          {
            keyword: "gun",
            detected_at_utc: new Date().toISOString(),
            source: "transcript",
            transcript_seq: 0,
          },
        ],
      })
    );
    render(<UrgencyPanel store={store} />);
    expect(screen.getByText("gun")).toBeInTheDocument();

    act(() => { vi.advanceTimersByTime(10_000); });
    expect(screen.queryByText("gun")).not.toBeInTheDocument();
  });

  it("renders prosody section with placeholder values", () => {
    render(<UrgencyPanel store={store} />);
    expect(screen.getByTestId("prosody-meter")).toBeInTheDocument();
    expect(screen.getByText("Pitch")).toBeInTheDocument();
    expect(screen.getByText("Loudness")).toBeInTheDocument();
  });
});
