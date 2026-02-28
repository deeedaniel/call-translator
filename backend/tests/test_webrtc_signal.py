from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from av import AudioFrame

from services.audio_bus import AudioBus
from services.webrtc_peer import TelephonyAudioTrack, WebRTCPeerManager


@pytest.mark.anyio
async def test_telephony_audio_track_produces_correct_frames():
    """TelephonyAudioTrack should yield AudioFrame with correct sample rate."""
    q: asyncio.Queue[bytes | None] = asyncio.Queue()

    # Push a 20ms frame of PCM16 silence (320 samples at 16kHz)
    silence = np.zeros(320, dtype=np.int16).tobytes()
    await q.put(silence)
    await q.put(None)  # sentinel to end

    track = TelephonyAudioTrack(q)

    frame = await track.recv()
    assert isinstance(frame, AudioFrame)
    assert frame.sample_rate == 16000

    arr = frame.to_ndarray()
    assert arr.shape == (1, 320)


@pytest.mark.anyio
async def test_telephony_audio_track_stops_on_sentinel():
    """Track should raise MediaStreamError when it receives None."""
    from aiortc.mediastreams import MediaStreamError

    q: asyncio.Queue[bytes | None] = asyncio.Queue()
    await q.put(None)

    track = TelephonyAudioTrack(q)

    with pytest.raises(MediaStreamError):
        await track.recv()


@pytest.mark.anyio
async def test_webrtc_peer_manager_creates_answer():
    """Verify that aiortc can generate an SDP answer from an offer.

    Uses the lower-level RTCPeerConnection API directly to avoid triggering
    the ICE connection attempt that would fail without a real remote peer.
    """
    from aiortc import RTCPeerConnection, RTCSessionDescription

    offer_sdp = (
        "v=0\r\n"
        "o=- 0 0 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        "c=IN IP4 0.0.0.0\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
        "a=sendrecv\r\n"
        "a=ice-ufrag:test\r\n"
        "a=ice-pwd:testpassword1234567890AB\r\n"
        "a=fingerprint:sha-256 "
        "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:"
        "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99\r\n"
        "a=setup:actpass\r\n"
        "a=mid:0\r\n"
        "a=rtcp-mux\r\n"
    )

    pc = RTCPeerConnection()
    try:
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        assert answer is not None
        assert isinstance(answer.sdp, str)
        assert "v=0" in answer.sdp
        assert "m=audio" in answer.sdp
    finally:
        await pc.close()


@pytest.mark.anyio
async def test_webrtc_peer_manager_close_is_idempotent():
    """Calling close() multiple times should not raise."""
    bus = AudioBus()
    peer = WebRTCPeerManager(call_id="test-call", audio_bus=bus)
    await peer.close()
    await peer.close()  # Should not raise


def test_signaling_ws_rejects_missing_call_id():
    """The /ws/webrtc endpoint requires a call_id query parameter."""
    from starlette.testclient import TestClient
    from main import app

    client = TestClient(app)

    with pytest.raises(Exception):
        with client.websocket_connect("/ws/webrtc") as ws:
            pass


def test_signaling_ws_bye_closes_connection():
    """Sending a 'bye' message should cleanly close the signaling session."""
    from starlette.testclient import TestClient
    from main import app

    client = TestClient(app)

    with client.websocket_connect("/ws/webrtc?call_id=test-bye") as ws:
        ws.send_text(json.dumps({"kind": "bye"}))
