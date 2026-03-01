import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useEmergencyCall } from "../hooks/useEmergencyCall";
import { CallStore } from "../stores/callStore";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  readyState = 0; // CONNECTING
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];

  constructor(_url: string) {
    MockWebSocket.instances.push(this);
    setTimeout(() => {
      this.readyState = 1; // OPEN
      this.onopen?.();
    }, 0);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.readyState = 3; // CLOSED
  }

  simulateMessage(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }

  static readonly OPEN = 1;
}

class MockRTCPeerConnection {
  static instances: MockRTCPeerConnection[] = [];
  ontrack: ((ev: unknown) => void) | null = null;
  onicecandidate: ((ev: unknown) => void) | null = null;
  localDescription = { sdp: "mock-offer-sdp", type: "offer" };

  constructor() {
    MockRTCPeerConnection.instances.push(this);
  }

  addTransceiver() {}
  async createOffer() {
    return { type: "offer", sdp: "mock-offer-sdp" };
  }
  async setLocalDescription() {}
  async setRemoteDescription() {}
  close() {}
}

describe("useEmergencyCall", () => {
  let store: CallStore;

  beforeEach(() => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    MockRTCPeerConnection.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket);
    vi.stubGlobal("RTCPeerConnection", MockRTCPeerConnection);
    vi.stubGlobal("RTCSessionDescription", class {
      constructor(public init: unknown) {}
    });
    store = new CallStore();
  });

  afterEach(() => {
    store.reset();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("opens a WebSocket on mount and sends SDP offer", async () => {
    renderHook(() => useEmergencyCall("call-123", store));

    await vi.advanceTimersByTimeAsync(10);

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockRTCPeerConnection.instances).toHaveLength(1);

    const ws = MockWebSocket.instances[0];
    const offerMsg = ws.sent.find((s) => {
      const parsed = JSON.parse(s);
      return parsed.kind === "sdp";
    });
    expect(offerMsg).toBeDefined();
    const parsed = JSON.parse(offerMsg!);
    expect(parsed.sdp.type).toBe("offer");
  });

  it("dispatches pipeline events to the store", async () => {
    renderHook(() => useEmergencyCall("call-123", store));

    await vi.advanceTimersByTimeAsync(10);

    const ws = MockWebSocket.instances[0];
    const dispatchSpy = vi.spyOn(store, "dispatch");

    await act(async () => {
      ws.simulateMessage({
        event: "transcript",
        call_id: "call-123",
        timestamp_utc: new Date().toISOString(),
        seq: 0,
        original_transcript: "Hola",
        detected_language: "es",
        asr_confidence: 0.9,
        word_timestamps: [],
        is_final: true,
      });
    });

    expect(dispatchSpy).toHaveBeenCalledTimes(1);
    expect(dispatchSpy).toHaveBeenCalledWith(
      expect.objectContaining({ event: "transcript" })
    );
  });

  it("sends bye on endCall and sets disconnected", async () => {
    const { result } = renderHook(() => useEmergencyCall("call-123", store));

    await vi.advanceTimersByTimeAsync(10);

    const ws = MockWebSocket.instances[0];
    ws.readyState = 1;

    act(() => {
      result.current.endCall();
    });

    const byeMsg = ws.sent.find((s) => JSON.parse(s).kind === "bye");
    expect(byeMsg).toBeDefined();
    expect(result.current.connectionStatus).toBe("disconnected");
  });
});
