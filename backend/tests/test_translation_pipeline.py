from __future__ import annotations

import asyncio
import json

import pytest

from models.asr_events import TranscriptEvent
from services.event_emitter import EventEmitter
from services.translation_pipeline import TranslationPipeline
from services.translation_worker import TranslationWorker


class FakeTranslationWorker(TranslationWorker):
    """Deterministic stub that prepends 'TRANSLATED: ' to input text."""

    def __init__(self) -> None:
        super().__init__(nvidia_api_key="fake-key")

    def _ensure_connection(self) -> None:
        pass

    def translate_sync(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        return f"TRANSLATED: {text}"


@pytest.fixture
def emitter() -> EventEmitter:
    return EventEmitter(max_queue_size=50)


@pytest.fixture
def fake_worker() -> FakeTranslationWorker:
    return FakeTranslationWorker()


@pytest.mark.anyio
async def test_pipeline_translates_transcript(
    emitter: EventEmitter, fake_worker: FakeTranslationWorker
):
    call_id = "test-translate"
    downstream_q = await emitter.subscribe(call_id)

    pipeline = TranslationPipeline(
        call_id=call_id,
        source_language="es",
        target_language="en",
        emitter=emitter,
        worker=fake_worker,
    )
    await pipeline.start()

    transcript = TranscriptEvent(
        call_id=call_id,
        seq=1,
        original_transcript="El paciente tiene dolor.",
        detected_language="es",
        asr_confidence=0.95,
        is_final=True,
    )
    await emitter.emit(transcript)

    await asyncio.sleep(0.3)

    events = []
    while not downstream_q.empty():
        events.append(json.loads(downstream_q.get_nowait()))

    translation_events = [e for e in events if e["event"] == "translation"]
    assert len(translation_events) >= 1
    assert translation_events[0]["translated_text"] == "TRANSLATED: El paciente tiene dolor."
    assert translation_events[0]["source_language"] == "es"
    assert translation_events[0]["target_language"] == "en"
    assert translation_events[0]["seq"] == 1

    await pipeline.stop()


@pytest.mark.anyio
async def test_pipeline_ignores_non_transcript_events(
    emitter: EventEmitter, fake_worker: FakeTranslationWorker
):
    from models.asr_events import BargeInEvent

    call_id = "test-filter"
    downstream_q = await emitter.subscribe(call_id)

    pipeline = TranslationPipeline(
        call_id=call_id,
        source_language="es",
        target_language="en",
        emitter=emitter,
        worker=fake_worker,
    )
    await pipeline.start()

    await emitter.emit(BargeInEvent(call_id=call_id))
    await asyncio.sleep(0.1)

    events = []
    while not downstream_q.empty():
        events.append(json.loads(downstream_q.get_nowait()))

    translation_events = [e for e in events if e["event"] == "translation"]
    assert len(translation_events) == 0

    await pipeline.stop()


@pytest.mark.anyio
async def test_pipeline_stop_cancels_task(
    emitter: EventEmitter, fake_worker: FakeTranslationWorker
):
    call_id = "test-stop"
    pipeline = TranslationPipeline(
        call_id=call_id,
        source_language="es",
        target_language="en",
        emitter=emitter,
        worker=fake_worker,
    )
    await pipeline.start()
    assert pipeline.is_running

    await pipeline.stop()
    assert not pipeline.is_running


@pytest.mark.anyio
async def test_pipeline_handles_absolute_transcripts(
    emitter: EventEmitter, fake_worker: FakeTranslationWorker
):
    """Verify absolute (accumulated) transcripts don't produce duplicates."""
    call_id = "test-absolute"
    downstream_q = await emitter.subscribe(call_id)

    pipeline = TranslationPipeline(
        call_id=call_id,
        source_language="es",
        target_language="en",
        emitter=emitter,
        worker=fake_worker,
    )
    await pipeline.start()

    for seq, text, final in [
        (0, "El paciente", False),
        (1, "El paciente tiene", False),
        (2, "El paciente tiene dolor.", True),
    ]:
        event = TranscriptEvent(
            call_id=call_id,
            seq=seq,
            original_transcript=text,
            detected_language="es",
            asr_confidence=0.9,
            is_final=final,
        )
        await emitter.emit(event)
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.3)

    events = []
    while not downstream_q.empty():
        events.append(json.loads(downstream_q.get_nowait()))

    translation_events = [e for e in events if e["event"] == "translation"]
    assert len(translation_events) == 1
    assert translation_events[0]["translated_text"] == "TRANSLATED: El paciente tiene dolor."

    await pipeline.stop()
