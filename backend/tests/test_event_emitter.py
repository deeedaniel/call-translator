from __future__ import annotations

import asyncio
import json

import pytest

from models.asr_events import BargeInEvent, TranscriptEvent, WordTimestamp
from services.event_emitter import EventEmitter


@pytest.fixture
def emitter() -> EventEmitter:
    return EventEmitter(max_queue_size=50)


@pytest.mark.anyio
async def test_emit_and_subscribe_roundtrip(emitter: EventEmitter):
    q = await emitter.subscribe("call-1")
    event = BargeInEvent(call_id="call-1")
    await emitter.emit(event)

    msg = q.get_nowait()
    data = json.loads(msg)
    assert data["event"] == "barge_in"
    assert data["call_id"] == "call-1"


@pytest.mark.anyio
async def test_multiple_subscribers(emitter: EventEmitter):
    q1 = await emitter.subscribe("call-1")
    q2 = await emitter.subscribe("call-1")

    event = BargeInEvent(call_id="call-1")
    await emitter.emit(event)

    assert q1.qsize() == 1
    assert q2.qsize() == 1

    msg1 = json.loads(q1.get_nowait())
    msg2 = json.loads(q2.get_nowait())
    assert msg1["event"] == msg2["event"] == "barge_in"


@pytest.mark.anyio
async def test_no_cross_talk(emitter: EventEmitter):
    qa = await emitter.subscribe("call-A")
    qb = await emitter.subscribe("call-B")

    await emitter.emit(BargeInEvent(call_id="call-A"))
    await emitter.emit(BargeInEvent(call_id="call-B"))

    assert qa.qsize() == 1
    assert qb.qsize() == 1

    assert json.loads(qa.get_nowait())["call_id"] == "call-A"
    assert json.loads(qb.get_nowait())["call_id"] == "call-B"


@pytest.mark.anyio
async def test_unsubscribe_stops_delivery(emitter: EventEmitter):
    q = await emitter.subscribe("call-1")
    await emitter.emit(BargeInEvent(call_id="call-1"))
    await emitter.unsubscribe("call-1", q)
    await emitter.emit(BargeInEvent(call_id="call-1"))

    assert q.qsize() == 1  # only the first one


@pytest.mark.anyio
async def test_transcript_event_json_format(emitter: EventEmitter):
    q = await emitter.subscribe("call-1")
    event = TranscriptEvent(
        call_id="call-1",
        seq=1,
        original_transcript="hello world",
        asr_confidence=0.95,
        word_timestamps=[
            WordTimestamp(word="hello", start_s=0.0, end_s=0.3, confidence=0.96),
            WordTimestamp(word="world", start_s=0.35, end_s=0.7, confidence=0.94),
        ],
    )
    await emitter.emit(event)

    data = json.loads(q.get_nowait())
    assert data["event"] == "transcript"
    assert data["original_transcript"] == "hello world"
    assert len(data["word_timestamps"]) == 2


@pytest.mark.anyio
async def test_backpressure_drops_oldest():
    emitter = EventEmitter(max_queue_size=3)
    q = await emitter.subscribe("call-1")

    for i in range(5):
        event = TranscriptEvent(
            call_id="call-1",
            seq=i,
            original_transcript=f"msg-{i}",
            asr_confidence=0.9,
        )
        await emitter.emit(event)

    items = []
    while not q.empty():
        items.append(json.loads(q.get_nowait()))

    assert len(items) == 3
    # Most recent message should be the last one emitted
    assert items[-1]["seq"] == 4
