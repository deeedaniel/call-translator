from __future__ import annotations

import asyncio
import json
import logging

from models.asr_events import TranscriptEvent
from models.translation_events import TranslationEvent
from services.event_emitter import EventEmitter
from services.punctuated_buffer import PunctuatedBufferStreamer
from services.translation_worker import TranslationWorker

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """Orchestrates EventEmitter -> Buffer -> Translate -> EventEmitter for a single call.

    Subscribes to the shared ``EventEmitter`` for ``TranscriptEvent`` payloads,
    feeds them through a ``PunctuatedBufferStreamer``, translates each clause
    via ``TranslationWorker``, and emits ``TranslationEvent`` back to the emitter
    for downstream TTS consumption.
    """

    def __init__(
        self,
        call_id: str,
        source_language: str,
        target_language: str,
        emitter: EventEmitter,
        worker: TranslationWorker,
    ) -> None:
        self._call_id = call_id
        self._source_language = source_language
        self._target_language = target_language
        self._emitter = emitter
        self._worker = worker
        self._buffer = PunctuatedBufferStreamer()
        self._seq = 0
        self._queue: asyncio.Queue[str] | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Subscribe to the emitter and launch the translation loop."""
        logger.info("Starting translation pipeline for call %s", self._call_id)
        self._queue = await self._emitter.subscribe(self._call_id)
        self._task = asyncio.create_task(
            self._run(), name=f"translate-{self._call_id}"
        )

    async def _run(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()

        while True:
            raw = await self._queue.get()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON on translation queue: %s", raw[:120])
                continue

            if data.get("event") != "transcript":
                continue

            transcript = TranscriptEvent.model_validate(data)
            for chunk in self._buffer.feed(transcript):
                try:
                    translated = await loop.run_in_executor(
                        None,
                        self._worker.translate_sync,
                        chunk.text,
                        chunk.source_language,
                        self._target_language,
                    )
                except Exception:
                    logger.exception(
                        "Translation failed for call %s", self._call_id
                    )
                    continue

                self._seq += 1
                event = TranslationEvent(
                    call_id=self._call_id,
                    seq=self._seq,
                    source_text=chunk.text,
                    translated_text=translated,
                    source_language=chunk.source_language,
                    target_language=self._target_language,
                )
                await self._emitter.emit(event)

    async def stop(self) -> None:
        """Flush the buffer, unsubscribe, and cancel the task."""
        logger.info("Stopping translation pipeline for call %s", self._call_id)
        self._buffer.flush(self._call_id)

        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

        if self._queue is not None:
            await self._emitter.unsubscribe(self._call_id, self._queue)
            self._queue = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()
