from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from services.analytics.analytics_worker import AnalyticsWorker
from services.analytics.audio_chunk_buffer import AudioChunkBuffer
from services.analytics.keyword_spotter import KeywordSpotter
from services.analytics.prosody_extractor import ProsodyExtractor
from services.analytics.stress_classifier import (
    MLStressClassifier,
    RuleBasedStressClassifier,
    StressClassifier,
)
from services.audio_bus import AudioBus
from services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)


class AnalyticsPipeline:
    """Lifecycle wrapper for the ML Audio Analytics microservice.

    Mirrors the ``ASRPipeline`` pattern: a single ``asyncio.Task``
    running an ``AnalyticsWorker`` for a given call.
    """

    def __init__(
        self,
        call_id: str,
        audio_bus: AudioBus,
        emitter: EventEmitter,
        *,
        keywords: list[str] | None = None,
        keywords_path: str | None = None,
        keyword_cooldown_s: float = 5.0,
        chunk_ms: int = 500,
        emit_interval_ms: int = 500,
        stress_classifier_type: str = "rule_based",
        ml_model_path: str | None = None,
    ) -> None:
        self.call_id = call_id
        self._audio_bus = audio_bus
        self._emitter = emitter

        classifier: StressClassifier
        if stress_classifier_type == "ml" and ml_model_path:
            classifier = MLStressClassifier(Path(ml_model_path))
        else:
            classifier = RuleBasedStressClassifier()

        self._worker = AnalyticsWorker(
            chunk_buffer=AudioChunkBuffer(chunk_ms=chunk_ms),
            prosody_extractor=ProsodyExtractor(),
            stress_classifier=classifier,
            keyword_spotter=KeywordSpotter(
                keywords=keywords,
                keywords_path=keywords_path,
                cooldown_s=keyword_cooldown_s,
            ),
            emit_interval_ms=emit_interval_ms,
        )
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        logger.info("Starting analytics pipeline for call %s", self.call_id)
        self._task = asyncio.create_task(
            self._worker.run(self.call_id, self._audio_bus, self._emitter),
            name=f"analytics-{self.call_id}",
        )

    async def stop(self) -> None:
        logger.info("Stopping analytics pipeline for call %s", self.call_id)
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()
