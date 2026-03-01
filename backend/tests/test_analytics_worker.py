from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from models.analytics_events import (
    AnalyticsEvent,
    DetectedKeyword,
    ProsodySnapshot,
)
from services.analytics.analytics_worker import AnalyticsWorker
from services.analytics.audio_chunk_buffer import AudioChunkBuffer
from services.analytics.prosody_extractor import ProsodyExtractor
from services.analytics.stress_classifier import RuleBasedStressClassifier
from services.analytics.keyword_spotter import KeywordSpotter
from services.audio_bus import AudioBus
from services.event_emitter import EventEmitter

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = SAMPLE_RATE // 2  # 500 ms


def _make_pcm_bytes(n_samples: int, amplitude: int = 1000) -> bytes:
    return (np.ones(n_samples, dtype=np.int16) * amplitude).tobytes()


class _FakeProsodyExtractor(ProsodyExtractor):
    """Returns a fixed snapshot without calling openSMILE."""

    def __init__(self, snapshot: ProsodySnapshot | None = None) -> None:
        self._fixed = snapshot or ProsodySnapshot(
            pitch_mean_hz=10.0, rms_energy_db=0.6
        )

    async def extract(self, chunk: np.ndarray) -> ProsodySnapshot:
        return self._fixed


class _FakeKeywordSpotter(KeywordSpotter):
    """Pushes a canned detection immediately, then waits for cancellation."""

    def __init__(self, detections: list[DetectedKeyword] | None = None) -> None:
        super().__init__(keywords=["fire"])
        self._detections = detections or []

    async def run(
        self,
        call_id: str,
        emitter: EventEmitter,
        hit_queue: asyncio.Queue[DetectedKeyword],
    ) -> None:
        for kw in self._detections:
            hit_queue.put_nowait(kw)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return


@pytest.fixture
def audio_bus() -> AudioBus:
    return AudioBus()


@pytest.fixture
def emitter() -> EventEmitter:
    return EventEmitter()


class TestAnalyticsWorker:
    @pytest.mark.asyncio
    async def test_emits_analytics_events(
        self, audio_bus: AudioBus, emitter: EventEmitter
    ):
        """Worker emits at least one AnalyticsEvent before the stream ends."""
        call_id = "test-emit"

        worker = AnalyticsWorker(
            chunk_buffer=AudioChunkBuffer(chunk_ms=500),
            prosody_extractor=_FakeProsodyExtractor(),
            stress_classifier=RuleBasedStressClassifier(),
            keyword_spotter=_FakeKeywordSpotter(),
            emit_interval_ms=100,
        )

        eq = await emitter.subscribe(call_id)

        async def _publish():
            await asyncio.sleep(0.05)
            await audio_bus.publish(call_id, _make_pcm_bytes(CHUNK_SAMPLES))
            await asyncio.sleep(0.25)
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await worker.run(call_id, audio_bus, emitter)

        events: list[dict] = []
        while not eq.empty():
            raw = eq.get_nowait()
            events.append(json.loads(raw))

        analytics = [e for e in events if e.get("event") == "analytics"]
        assert len(analytics) >= 1

        first = analytics[0]
        assert first["call_id"] == call_id
        assert first["stress_level"] in ("low", "medium", "high")
        assert 0.0 <= first["stress_confidence"] <= 1.0
        assert "prosody" in first

    @pytest.mark.asyncio
    async def test_keywords_appear_in_event(
        self, audio_bus: AudioBus, emitter: EventEmitter
    ):
        """Keywords detected by KWS show up in the emitted AnalyticsEvent."""
        call_id = "test-kw"

        canned = [DetectedKeyword(keyword="fire")]
        worker = AnalyticsWorker(
            chunk_buffer=AudioChunkBuffer(chunk_ms=500),
            prosody_extractor=_FakeProsodyExtractor(),
            stress_classifier=RuleBasedStressClassifier(),
            keyword_spotter=_FakeKeywordSpotter(detections=canned),
            emit_interval_ms=100,
        )

        eq = await emitter.subscribe(call_id)

        async def _publish():
            await asyncio.sleep(0.05)
            await audio_bus.publish(call_id, _make_pcm_bytes(CHUNK_SAMPLES))
            await asyncio.sleep(0.25)
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await worker.run(call_id, audio_bus, emitter)

        events: list[dict] = []
        while not eq.empty():
            events.append(json.loads(eq.get_nowait()))

        analytics = [e for e in events if e.get("event") == "analytics"]
        all_kw = []
        for e in analytics:
            all_kw.extend(e.get("keywords_detected", []))

        assert any(k["keyword"] == "fire" for k in all_kw)

    @pytest.mark.asyncio
    async def test_seq_increments(
        self, audio_bus: AudioBus, emitter: EventEmitter
    ):
        """Each emitted event has a monotonically increasing seq number."""
        call_id = "test-seq"

        worker = AnalyticsWorker(
            chunk_buffer=AudioChunkBuffer(chunk_ms=500),
            prosody_extractor=_FakeProsodyExtractor(),
            stress_classifier=RuleBasedStressClassifier(),
            keyword_spotter=_FakeKeywordSpotter(),
            emit_interval_ms=80,
        )

        eq = await emitter.subscribe(call_id)

        async def _publish():
            await asyncio.sleep(0.05)
            for _ in range(4):
                await audio_bus.publish(call_id, _make_pcm_bytes(CHUNK_SAMPLES))
                await asyncio.sleep(0.1)
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await worker.run(call_id, audio_bus, emitter)

        seqs = []
        while not eq.empty():
            data = json.loads(eq.get_nowait())
            if data.get("event") == "analytics":
                seqs.append(data["seq"])

        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

    @pytest.mark.asyncio
    async def test_silence_yields_low_stress(
        self, audio_bus: AudioBus, emitter: EventEmitter
    ):
        """Silence input should classify as low stress."""
        call_id = "test-silence"

        worker = AnalyticsWorker(
            chunk_buffer=AudioChunkBuffer(chunk_ms=500),
            prosody_extractor=_FakeProsodyExtractor(snapshot=ProsodySnapshot()),
            stress_classifier=RuleBasedStressClassifier(),
            keyword_spotter=_FakeKeywordSpotter(),
            emit_interval_ms=100,
        )

        eq = await emitter.subscribe(call_id)

        async def _publish():
            await asyncio.sleep(0.05)
            silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16).tobytes()
            await audio_bus.publish(call_id, silence)
            await asyncio.sleep(0.2)
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await worker.run(call_id, audio_bus, emitter)

        while not eq.empty():
            data = json.loads(eq.get_nowait())
            if data.get("event") == "analytics":
                assert data["stress_level"] == "low"
