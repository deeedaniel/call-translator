from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.audio_bus import AudioBus
from services.twilio_stream import TwilioStreamSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["twilio"])

# Injected at app startup via lifespan; see main.py
audio_bus: AudioBus | None = None


def set_audio_bus(bus: AudioBus) -> None:
    global audio_bus
    audio_bus = bus


@router.websocket("/ws/twilio")
async def twilio_media_stream(ws: WebSocket) -> None:
    """Handle an inbound Twilio Media Stream WebSocket connection."""
    assert audio_bus is not None, "AudioBus not initialised"

    await ws.accept()
    session = TwilioStreamSession(audio_bus)
    logger.info("Twilio WebSocket connected")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            await session.handle_message(data)
    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception:
        logger.exception("Error in Twilio WebSocket handler")
    finally:
        await session.cleanup()
