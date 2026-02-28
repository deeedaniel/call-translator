from __future__ import annotations

import asyncio
import logging

from services.asr_worker import ASRWorker
from services.audio_bus import AudioBus
from services.event_emitter import EventEmitter
from services.vad_worker import SpeechChunk, VADWorker

logger = logging.getLogger(__name__)


class ASRPipeline:
    """Orchestrates VAD → ASR → EventEmitter for a single call.

    Launched as a pair of ``asyncio.Task`` objects when a call begins.
    Cancels cleanly when the AudioBus channel closes (``None`` sentinel).
    """

    def __init__(
        self,
        call_id: str,
        audio_bus: AudioBus,
        emitter: EventEmitter,
        asr_worker: ASRWorker,
        vad_worker: VADWorker,
        speech_queue_size: int = 50,
    ) -> None:
        self.call_id = call_id
        self._audio_bus = audio_bus
        self._emitter = emitter
        self._asr_worker = asr_worker
        self._vad_worker = vad_worker
        self._speech_queue: asyncio.Queue[SpeechChunk | None] = asyncio.Queue(
            maxsize=speech_queue_size
        )
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Launch the VAD and ASR worker tasks."""
        logger.info("Starting ASR pipeline for call %s", self.call_id)

        vad_task = asyncio.create_task(
            self._vad_worker.run(
                self.call_id,
                self._audio_bus,
                self._speech_queue,
                self._emitter,
            ),
            name=f"vad-{self.call_id}",
        )
        asr_task = asyncio.create_task(
            self._asr_worker.run(
                self.call_id,
                self._speech_queue,
                self._emitter,
            ),
            name=f"asr-{self.call_id}",
        )
        self._tasks = [vad_task, asr_task]

    async def stop(self) -> None:
        """Cancel running tasks and wait for cleanup."""
        logger.info("Stopping ASR pipeline for call %s", self.call_id)
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    @property
    def is_running(self) -> bool:
        return any(not t.done() for t in self._tasks)
