from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from models.webrtc_models import SignalMessage
from services.audio_bus import AudioBus
from services.webrtc_peer import WebRTCPeerManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webrtc"])

# Injected at app startup; see main.py
audio_bus: AudioBus | None = None
_peers: dict[str, WebRTCPeerManager] = {}


def set_audio_bus(bus: AudioBus) -> None:
    global audio_bus
    audio_bus = bus


@router.websocket("/ws/webrtc")
async def webrtc_signaling(ws: WebSocket, call_id: str = Query(...)) -> None:
    """WebSocket endpoint for WebRTC signaling between browser and server.

    Query params:
        call_id: The call identifier to subscribe to on the AudioBus.

    Protocol (JSON messages with "kind" discriminator):
        -> {kind: "sdp", sdp: {type: "offer", sdp: "..."}}
        <- {kind: "sdp", sdp: {type: "answer", sdp: "..."}}
        -> {kind: "ice", ice: {candidate: "...", sdpMid: "...", sdpMLineIndex: 0}}
        -> {kind: "bye"}
    """
    assert audio_bus is not None, "AudioBus not initialised"

    await ws.accept()
    logger.info("WebRTC signaling WS connected for call %s", call_id)

    peer = WebRTCPeerManager(call_id=call_id, audio_bus=audio_bus)
    _peers[call_id] = peer

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg = SignalMessage.model_validate(data)

            if msg.kind == "sdp" and msg.sdp and msg.sdp.type == "offer":
                answer_sdp = await peer.create_answer(
                    offer_sdp=msg.sdp.sdp, offer_type=msg.sdp.type
                )
                await ws.send_text(
                    json.dumps(
                        {
                            "kind": "sdp",
                            "sdp": {"type": "answer", "sdp": answer_sdp},
                        }
                    )
                )

            elif msg.kind == "ice" and msg.ice:
                await peer.add_ice_candidate(msg.ice.model_dump(by_alias=True))

            elif msg.kind == "bye":
                break

    except WebSocketDisconnect:
        logger.info("WebRTC signaling WS disconnected for call %s", call_id)
    except Exception:
        logger.exception("Error in WebRTC signaling handler for call %s", call_id)
    finally:
        await peer.close()
        _peers.pop(call_id, None)
