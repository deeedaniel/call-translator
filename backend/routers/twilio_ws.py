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


async def _on_call_start(call_id: str) -> None:
    """Start all processing pipelines when Twilio streams audio for a call."""
    from main import (
        start_asr_pipeline,
        start_analytics_pipeline,
        start_translation_pipeline,
    )

    logger.info("Starting pipelines for call %s", call_id)
    await start_asr_pipeline(call_id)
    await start_translation_pipeline(call_id, source_language="es", target_language="en")
    await start_analytics_pipeline(call_id)


async def _on_call_stop(call_id: str) -> None:
    """Tear down all processing pipelines when a Twilio call ends."""
    from main import (
        stop_asr_pipeline,
        stop_analytics_pipeline,
        stop_translation_pipeline,
    )

    logger.info("Stopping pipelines for call %s", call_id)
    await stop_analytics_pipeline(call_id)
    await stop_translation_pipeline(call_id)
    await stop_asr_pipeline(call_id)


@router.websocket("/ws/twilio")
async def twilio_media_stream(ws: WebSocket) -> None:
    """Handle an inbound Twilio Media Stream WebSocket connection."""
    assert audio_bus is not None, "AudioBus not initialised"

    await ws.accept()
    session = TwilioStreamSession(
        audio_bus,
        on_start=_on_call_start,
        on_stop=_on_call_stop,
    )
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
