from __future__ import annotations

import asyncio
import json

import pytest

from models.analytics_events import DetectedKeyword
from services.analytics.keyword_spotter import KeywordSpotter, DEFAULT_KEYWORDS
from services.event_emitter import EventEmitter


@pytest.fixture
def emitter() -> EventEmitter:
    return EventEmitter()


def _transcript_json(text: str, seq: int = 0, call_id: str = "call-1") -> str:
    return json.dumps({
        "event": "transcript",
        "call_id": call_id,
        "seq": seq,
        "original_transcript": text,
        "asr_confidence": 0.9,
        "is_final": True,
    })


class TestKeywordSpotter:
    @pytest.mark.asyncio
    async def test_detection_fires(self, emitter: EventEmitter):
        """A transcript containing a keyword produces a DetectedKeyword."""
        hit_queue: asyncio.Queue[DetectedKeyword] = asyncio.Queue()
        spotter = KeywordSpotter(keywords=["fire", "gun"], cooldown_s=0.0)

        call_id = "call-1"
        eq = await emitter.subscribe(call_id)

        async def _emit():
            await asyncio.sleep(0.01)
            eq_internal = emitter._subscribers[call_id]
            msg = _transcript_json("there is a fire in the building", seq=1)
            for q in eq_internal:
                q.put_nowait(msg)
            await asyncio.sleep(0.05)

        task = asyncio.create_task(_emit())
        spotter_task = asyncio.create_task(
            spotter.run(call_id, emitter, hit_queue)
        )

        await task
        await asyncio.sleep(0.05)
        spotter_task.cancel()
        await asyncio.gather(spotter_task, return_exceptions=True)

        assert not hit_queue.empty()
        kw = hit_queue.get_nowait()
        assert kw.keyword == "fire"
        assert kw.source == "transcript"
        assert kw.transcript_seq == 1

    @pytest.mark.asyncio
    async def test_case_insensitive(self, emitter: EventEmitter):
        """Keywords are matched case-insensitively."""
        hit_queue: asyncio.Queue[DetectedKeyword] = asyncio.Queue()
        spotter = KeywordSpotter(keywords=["fire"], cooldown_s=0.0)

        call_id = "call-ci"

        async def _emit():
            await asyncio.sleep(0.01)
            subs = emitter._subscribers[call_id]
            msg = _transcript_json("There is a FIRE!", call_id=call_id)
            for q in subs:
                q.put_nowait(msg)
            await asyncio.sleep(0.05)

        eq = await emitter.subscribe(call_id)
        task = asyncio.create_task(_emit())
        spotter_task = asyncio.create_task(
            spotter.run(call_id, emitter, hit_queue)
        )

        await task
        await asyncio.sleep(0.05)
        spotter_task.cancel()
        await asyncio.gather(spotter_task, return_exceptions=True)

        assert not hit_queue.empty()
        kw = hit_queue.get_nowait()
        assert kw.keyword == "fire"

    @pytest.mark.asyncio
    async def test_cooldown_deduplication(self, emitter: EventEmitter):
        """Repeated keyword within cooldown window is suppressed."""
        hit_queue: asyncio.Queue[DetectedKeyword] = asyncio.Queue()
        spotter = KeywordSpotter(keywords=["fire"], cooldown_s=999.0)

        call_id = "call-cd"

        async def _emit():
            await asyncio.sleep(0.01)
            subs = emitter._subscribers[call_id]
            for i in range(3):
                msg = _transcript_json("fire", seq=i, call_id=call_id)
                for q in subs:
                    q.put_nowait(msg)
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)

        eq = await emitter.subscribe(call_id)
        task = asyncio.create_task(_emit())
        spotter_task = asyncio.create_task(
            spotter.run(call_id, emitter, hit_queue)
        )

        await task
        await asyncio.sleep(0.05)
        spotter_task.cancel()
        await asyncio.gather(spotter_task, return_exceptions=True)

        assert hit_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_no_match(self, emitter: EventEmitter):
        """Transcripts without keywords produce no hits."""
        hit_queue: asyncio.Queue[DetectedKeyword] = asyncio.Queue()
        spotter = KeywordSpotter(keywords=["fire", "gun"], cooldown_s=0.0)

        call_id = "call-nm"

        async def _emit():
            await asyncio.sleep(0.01)
            subs = emitter._subscribers[call_id]
            msg = _transcript_json("the weather is nice today", call_id=call_id)
            for q in subs:
                q.put_nowait(msg)
            await asyncio.sleep(0.05)

        eq = await emitter.subscribe(call_id)
        task = asyncio.create_task(_emit())
        spotter_task = asyncio.create_task(
            spotter.run(call_id, emitter, hit_queue)
        )

        await task
        await asyncio.sleep(0.05)
        spotter_task.cancel()
        await asyncio.gather(spotter_task, return_exceptions=True)

        assert hit_queue.empty()

    @pytest.mark.asyncio
    async def test_multi_word_phrase(self, emitter: EventEmitter):
        """Multi-word phrases like 'chest pain' are matched."""
        hit_queue: asyncio.Queue[DetectedKeyword] = asyncio.Queue()
        spotter = KeywordSpotter(keywords=["chest pain"], cooldown_s=0.0)

        call_id = "call-mp"

        async def _emit():
            await asyncio.sleep(0.01)
            subs = emitter._subscribers[call_id]
            msg = _transcript_json(
                "patient has chest pain and difficulty breathing",
                call_id=call_id,
            )
            for q in subs:
                q.put_nowait(msg)
            await asyncio.sleep(0.05)

        eq = await emitter.subscribe(call_id)
        task = asyncio.create_task(_emit())
        spotter_task = asyncio.create_task(
            spotter.run(call_id, emitter, hit_queue)
        )

        await task
        await asyncio.sleep(0.05)
        spotter_task.cancel()
        await asyncio.gather(spotter_task, return_exceptions=True)

        assert not hit_queue.empty()
        kw = hit_queue.get_nowait()
        assert kw.keyword == "chest pain"

    @pytest.mark.asyncio
    async def test_multiple_keywords_in_one_transcript(self, emitter: EventEmitter):
        """Multiple keywords in a single transcript are all detected."""
        hit_queue: asyncio.Queue[DetectedKeyword] = asyncio.Queue()
        spotter = KeywordSpotter(
            keywords=["fire", "trapped"], cooldown_s=0.0
        )

        call_id = "call-mk"

        async def _emit():
            await asyncio.sleep(0.01)
            subs = emitter._subscribers[call_id]
            msg = _transcript_json(
                "there is a fire and people are trapped",
                call_id=call_id,
            )
            for q in subs:
                q.put_nowait(msg)
            await asyncio.sleep(0.05)

        eq = await emitter.subscribe(call_id)
        task = asyncio.create_task(_emit())
        spotter_task = asyncio.create_task(
            spotter.run(call_id, emitter, hit_queue)
        )

        await task
        await asyncio.sleep(0.05)
        spotter_task.cancel()
        await asyncio.gather(spotter_task, return_exceptions=True)

        assert hit_queue.qsize() == 2
        kws = {hit_queue.get_nowait().keyword, hit_queue.get_nowait().keyword}
        assert kws == {"fire", "trapped"}

    @pytest.mark.asyncio
    async def test_non_transcript_events_ignored(self, emitter: EventEmitter):
        """Events other than 'transcript' are silently skipped."""
        hit_queue: asyncio.Queue[DetectedKeyword] = asyncio.Queue()
        spotter = KeywordSpotter(keywords=["fire"], cooldown_s=0.0)

        call_id = "call-nt"

        async def _emit():
            await asyncio.sleep(0.01)
            subs = emitter._subscribers[call_id]
            msg = json.dumps({"event": "vad_state", "speaking": True})
            for q in subs:
                q.put_nowait(msg)
            await asyncio.sleep(0.05)

        eq = await emitter.subscribe(call_id)
        task = asyncio.create_task(_emit())
        spotter_task = asyncio.create_task(
            spotter.run(call_id, emitter, hit_queue)
        )

        await task
        await asyncio.sleep(0.05)
        spotter_task.cancel()
        await asyncio.gather(spotter_task, return_exceptions=True)

        assert hit_queue.empty()

    def test_default_keywords_populated(self):
        """Default keyword list contains expected emergency terms."""
        assert "fire" in DEFAULT_KEYWORDS
        assert "chest pain" in DEFAULT_KEYWORDS
        assert "not breathing" in DEFAULT_KEYWORDS
        assert len(DEFAULT_KEYWORDS) >= 20
