from __future__ import annotations

import asyncio
import logging

import numpy as np

from models.analytics_events import AnalyticsEvent, DetectedKeyword, ProsodySnapshot
from services.analytics.audio_chunk_buffer import AudioChunkBuffer
from services.analytics.keyword_spotter import KeywordSpotter
from services.analytics.prosody_extractor import ProsodyExtractor
from services.analytics.stress_classifier import StressClassifier
from services.audio_bus import AudioBus
from services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)


class AnalyticsWorker:
    """Orchestrates prosody analysis + transcript KWS and emits AnalyticsEvents.

    Runs three concurrent async tasks:
    1. **Prosody loop** -- consumes AudioChunkBuffer, extracts features,
       classifies stress.
    2. **KWS loop** -- KeywordSpotter matching against the EventEmitter
       transcript stream.
    3. **Tick emitter** -- every *emit_interval_ms* milliseconds, publishes
       an ``AnalyticsEvent`` to the ``EventEmitter``.
    """

    def __init__(
        self,
        chunk_buffer: AudioChunkBuffer,
        prosody_extractor: ProsodyExtractor,
        stress_classifier: StressClassifier,
        keyword_spotter: KeywordSpotter,
        emit_interval_ms: int = 500,
    ) -> None:
        self._chunk_buffer = chunk_buffer
        self._prosody = prosody_extractor
        self._classifier = stress_classifier
        self._kws = keyword_spotter
        self._emit_interval_s = emit_interval_ms / 1000.0

        self._current_stress: tuple[str, float] = ("low", 0.5)
        self._current_prosody = ProsodySnapshot()
        self._pending_keywords: list[DetectedKeyword] = []
        self._seq = 0
        self._done = asyncio.Event()

    async def run(
        self,
        call_id: str,
        audio_bus: AudioBus,
        emitter: EventEmitter,
    ) -> None:
        """Start all tasks; returns when the audio stream ends."""
        chunk_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=50)
        kw_queue: asyncio.Queue[DetectedKeyword] = asyncio.Queue(maxsize=200)

        prosody_task = asyncio.create_task(
            self._prosody_loop(call_id, audio_bus, chunk_queue),
            name=f"analytics-prosody-{call_id}",
        )
        kws_task = asyncio.create_task(
            self._kws.run(call_id, emitter, kw_queue),
            name=f"analytics-kws-{call_id}",
        )
        ticker_task = asyncio.create_task(
            self._tick_loop(call_id, emitter, kw_queue),
            name=f"analytics-tick-{call_id}",
        )

        try:
            await prosody_task
        finally:
            self._done.set()
            kws_task.cancel()
            await asyncio.gather(kws_task, ticker_task, return_exceptions=True)

            await self._emit_final(call_id, emitter, kw_queue)
            logger.info("AnalyticsWorker finished for call %s", call_id)

    async def _prosody_loop(
        self,
        call_id: str,
        audio_bus: AudioBus,
        chunk_queue: asyncio.Queue[np.ndarray | None],
    ) -> None:
        """Buffer audio into chunks, extract prosody, classify stress."""
        buffer_task = asyncio.create_task(
            self._chunk_buffer.run(call_id, audio_bus, chunk_queue),
            name=f"analytics-buffer-{call_id}",
        )
        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    return

                snapshot = await self._prosody.extract(chunk)
                level, conf = self._classifier.classify(snapshot)
                self._current_stress = (level, conf)
                self._current_prosody = snapshot
        finally:
            buffer_task.cancel()
            await asyncio.gather(buffer_task, return_exceptions=True)

    async def _tick_loop(
        self,
        call_id: str,
        emitter: EventEmitter,
        kw_queue: asyncio.Queue[DetectedKeyword],
    ) -> None:
        """Emit an AnalyticsEvent every *emit_interval_s*."""
        while not self._done.is_set():
            await asyncio.sleep(self._emit_interval_s)
            self._drain_keywords(kw_queue)
            await self._emit(call_id, emitter)

    def _drain_keywords(self, kw_queue: asyncio.Queue[DetectedKeyword]) -> None:
        while not kw_queue.empty():
            try:
                self._pending_keywords.append(kw_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

    async def _emit(self, call_id: str, emitter: EventEmitter) -> None:
        level, conf = self._current_stress
        event = AnalyticsEvent(
            call_id=call_id,
            seq=self._seq,
            stress_level=level,
            stress_confidence=conf,
            prosody=self._current_prosody,
            keywords_detected=list(self._pending_keywords),
        )
        self._pending_keywords.clear()
        self._seq += 1
        await emitter.emit(event)

    async def _emit_final(
        self,
        call_id: str,
        emitter: EventEmitter,
        kw_queue: asyncio.Queue[DetectedKeyword],
    ) -> None:
        """Emit one last event with any remaining keywords."""
        self._drain_keywords(kw_queue)
        await self._emit(call_id, emitter)
