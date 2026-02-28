from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from starlette.testclient import TestClient

from main import app, get_audio_bus


def _make_start_msg(stream_sid: str = "MZ123", call_sid: str = "CA456") -> dict:
    return {
        "event": "start",
        "sequenceNumber": "1",
        "streamSid": stream_sid,
        "start": {
            "streamSid": stream_sid,
            "accountSid": "AC000",
            "callSid": call_sid,
            "tracks": ["inbound"],
            "customParameters": {},
            "mediaFormat": {
                "encoding": "audio/x-mulaw",
                "sampleRate": 8000,
                "channels": 1,
            },
        },
    }


def _make_media_msg(
    payload_bytes: bytes,
    seq: int = 2,
    stream_sid: str = "MZ123",
) -> dict:
    return {
        "event": "media",
        "sequenceNumber": str(seq),
        "streamSid": stream_sid,
        "media": {
            "track": "inbound",
            "chunk": str(seq),
            "timestamp": str(seq * 20),
            "payload": base64.b64encode(payload_bytes).decode(),
        },
    }


def _make_stop_msg(stream_sid: str = "MZ123") -> dict:
    return {
        "event": "stop",
        "sequenceNumber": "99",
        "streamSid": stream_sid,
    }


def test_full_twilio_stream_lifecycle():
    """Simulate connected -> start -> media*3 -> stop and verify AudioBus output."""
    bus = get_audio_bus()
    client = TestClient(app)

    # 160 bytes of mu-law silence (0xFF)
    silence_mulaw = bytes([0xFF] * 160)

    import asyncio

    loop = asyncio.new_event_loop()
    q = loop.run_until_complete(bus.subscribe("CA456"))

    with client.websocket_connect("/ws/twilio") as ws:
        ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0"}))
        ws.send_text(json.dumps(_make_start_msg()))

        for i in range(3):
            ws.send_text(json.dumps(_make_media_msg(silence_mulaw, seq=i + 2)))

        ws.send_text(json.dumps(_make_stop_msg()))

    received: list[bytes] = []
    while not q.empty():
        item = q.get_nowait()
        if item is None:
            break
        received.append(item)

    assert len(received) == 3

    for frame in received:
        pcm = np.frombuffer(frame, dtype=np.int16)
        # 160 mu-law samples -> 320 PCM16 samples at 16kHz
        assert len(pcm) == 320
        # Silence should produce near-zero values
        assert np.max(np.abs(pcm)) < 100

    loop.run_until_complete(bus.unsubscribe("CA456", q))
    loop.close()


def test_malformed_message_does_not_crash():
    """The WebSocket handler should survive unparseable JSON payloads."""
    client = TestClient(app)

    with client.websocket_connect("/ws/twilio") as ws:
        ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0"}))
        ws.send_text(json.dumps({"event": "unknown_event", "foo": "bar"}))
        ws.send_text(json.dumps(_make_start_msg()))
        ws.send_text(json.dumps(_make_stop_msg()))
    # If we get here without exception, the test passes.


def test_media_before_start_is_ignored():
    """Media events received before a start event should be silently dropped."""
    bus = get_audio_bus()
    client = TestClient(app)

    import asyncio
    loop = asyncio.new_event_loop()
    q = loop.run_until_complete(bus.subscribe("CA456"))

    silence_mulaw = bytes([0xFF] * 160)

    with client.websocket_connect("/ws/twilio") as ws:
        ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0"}))
        # Send media BEFORE start
        ws.send_text(json.dumps(_make_media_msg(silence_mulaw)))
        ws.send_text(json.dumps(_make_stop_msg()))

    assert q.empty()

    loop.run_until_complete(bus.unsubscribe("CA456", q))
    loop.close()
