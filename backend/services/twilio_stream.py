from __future__ import annotations

import base64
import logging
from typing import Awaitable, Callable

from models.twilio_events import (
    MediaEvent,
    StartEvent,
    StopEvent,
    parse_twilio_event,
)
from services.audio_bus import AudioBus
from services.audio_transcoder import AudioTranscoder

logger = logging.getLogger(__name__)

OnStartCallback = Callable[[str], Awaitable[None]]
OnStopCallback = Callable[[str], Awaitable[None]]


class TwilioStreamSession:
    """Manages the state of a single Twilio Media Stream connection.

    Lifecycle: connected -> start -> N x media -> stop
    """

    def __init__(
        self,
        audio_bus: AudioBus,
        on_start: OnStartCallback | None = None,
        on_stop: OnStopCallback | None = None,
    ) -> None:
        self._bus = audio_bus
        self._transcoder = AudioTranscoder()
        self._on_start_cb = on_start
        self._on_stop_cb = on_stop
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self._sequence: int = 0
        self._started = False

    @property
    def call_id(self) -> str:
        """Stable identifier for AudioBus keying. Falls back to stream_sid."""
        return self.call_sid or self.stream_sid or "unknown"

    async def handle_message(self, data: dict) -> None:
        """Dispatch a raw Twilio WebSocket JSON message."""
        try:
            event = parse_twilio_event(data)
        except (ValueError, Exception) as exc:
            logger.warning("Ignoring unparseable Twilio message: %s", exc)
            return

        if isinstance(event, StartEvent):
            await self._on_start(event)
        elif isinstance(event, MediaEvent):
            await self._on_media(event)
        elif isinstance(event, StopEvent):
            await self._on_stop(event)
        else:
            logger.debug("Twilio event %s (no-op)", data.get("event"))

    async def _on_start(self, event: StartEvent) -> None:
        self.stream_sid = event.stream_sid
        self.call_sid = event.start.call_sid
        self._started = True
        logger.info(
            "Twilio stream started: stream=%s call=%s",
            self.stream_sid,
            self.call_sid,
        )
        if self._on_start_cb:
            await self._on_start_cb(self.call_id)

    async def _on_media(self, event: MediaEvent) -> None:
        if not self._started:
            logger.warning("Received media before start event, ignoring")
            return

        self._sequence = int(event.sequence_number)
        mulaw_bytes = base64.b64decode(event.media.payload)
        pcm16_bytes = self._transcoder.transcode(mulaw_bytes)
        await self._bus.publish(self.call_id, pcm16_bytes)

    async def _on_stop(self, event: StopEvent) -> None:
        logger.info("Twilio stream stopped: stream=%s", self.stream_sid)
        if self._on_stop_cb:
            await self._on_stop_cb(self.call_id)
        await self._bus.close_channel(self.call_id)
        self._started = False

    async def cleanup(self) -> None:
        """Called when the WebSocket disconnects (possibly without a stop event)."""
        if self._started:
            if self._on_stop_cb:
                await self._on_stop_cb(self.call_id)
            await self._bus.close_channel(self.call_id)
            self._started = False
