import { useEffect, useRef, useState, useCallback } from "react";
import type { CallStore } from "../stores/callStore";
import type { ConnectionStatus, ServerEvent } from "../types/events";

interface UseEmergencyCallReturn {
  connectionStatus: ConnectionStatus;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  endCall: () => void;
}

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000];

function getWsUrl(callId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/webrtc?call_id=${encodeURIComponent(callId)}`;
}

function isSignaling(data: unknown): data is { kind: string } {
  return typeof data === "object" && data !== null && "kind" in data;
}

function isPipelineEvent(data: unknown): data is ServerEvent {
  return typeof data === "object" && data !== null && "event" in data;
}

export function useEmergencyCall(
  callId: string,
  store: CallStore
): UseEmergencyCallReturn {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closedIntentionallyRef = useRef(false);

  const cleanup = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    cleanup();

    const status: ConnectionStatus =
      reconnectAttemptRef.current > 0 ? "reconnecting" : "connecting";
    setConnectionStatus(status);
    store.setConnectionStatus(status);

    const ws = new WebSocket(getWsUrl(callId));
    wsRef.current = ws;

    ws.onopen = async () => {
      reconnectAttemptRef.current = 0;
      setConnectionStatus("connected");
      store.setConnectionStatus("connected");

      try {
        const pc = new RTCPeerConnection({
          iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        });
        pcRef.current = pc;

        pc.addTransceiver("audio", { direction: "recvonly" });

        pc.ontrack = (ev) => {
          if (audioRef.current && ev.streams[0]) {
            audioRef.current.srcObject = ev.streams[0];
            audioRef.current.play().catch(() => {});
          }
        };

        pc.onicecandidate = (ev) => {
          if (ev.candidate && ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                kind: "ice",
                ice: {
                  candidate: ev.candidate.candidate,
                  sdpMid: ev.candidate.sdpMid,
                  sdpMLineIndex: ev.candidate.sdpMLineIndex,
                },
              })
            );
          }
        };

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        ws.send(
          JSON.stringify({
            kind: "sdp",
            sdp: { type: "offer", sdp: offer.sdp },
          })
        );
      } catch (err) {
        console.error("WebRTC setup failed:", err);
      }
    };

    ws.onmessage = async (ev) => {
      let data: unknown;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return;
      }

      if (isSignaling(data)) {
        if (data.kind === "sdp" && pcRef.current) {
          const sdpData = data as { kind: "sdp"; sdp: { type: string; sdp: string } };
          await pcRef.current.setRemoteDescription(
            new RTCSessionDescription({
              type: sdpData.sdp.type as RTCSdpType,
              sdp: sdpData.sdp.sdp,
            })
          );
        }
      } else if (isPipelineEvent(data)) {
        store.dispatch(data);
      }
    };

    ws.onclose = () => {
      if (closedIntentionallyRef.current) {
        setConnectionStatus("disconnected");
        store.setConnectionStatus("disconnected");
        return;
      }

      const attempt = reconnectAttemptRef.current;
      const delay = RECONNECT_DELAYS[Math.min(attempt, RECONNECT_DELAYS.length - 1)];
      reconnectAttemptRef.current = attempt + 1;
      setConnectionStatus("reconnecting");
      store.setConnectionStatus("reconnecting");

      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [callId, store, cleanup]);

  const endCall = useCallback(() => {
    closedIntentionallyRef.current = true;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ kind: "bye" }));
    }
    cleanup();
    setConnectionStatus("disconnected");
    store.setConnectionStatus("disconnected");
  }, [cleanup, store]);

  useEffect(() => {
    closedIntentionallyRef.current = false;
    connect();
    return () => {
      closedIntentionallyRef.current = true;
      cleanup();
    };
  }, [connect, cleanup]);

  return { connectionStatus, audioRef, endCall };
}
