from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.audio_bus import AudioBus
from services.event_emitter import EventEmitter
from services.vad_worker import FRAME_SAMPLES, SAMPLE_RATE, SpeechChunk, VADWorker


def _generate_sine(duration_s: float, freq: float = 400.0) -> bytes:
    """Generate PCM16 sine wave bytes at 16 kHz."""
    t = np.arange(0, duration_s, 1 / SAMPLE_RATE)
    samples = (16000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return samples.tobytes()


def _generate_silence(duration_s: float) -> bytes:
    """Generate silent PCM16 bytes at 16 kHz."""
    n_samples = int(duration_s * SAMPLE_RATE)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


@pytest.fixture
def bus() -> AudioBus:
    return AudioBus(max_queue_size=500)


@pytest.fixture
def emitter() -> EventEmitter:
    return EventEmitter()


@pytest.fixture
def speech_queue() -> asyncio.Queue[SpeechChunk | None]:
    return asyncio.Queue(maxsize=50)


@pytest.mark.anyio
async def test_vad_detects_speech_and_produces_chunk(
    bus: AudioBus, emitter: EventEmitter, speech_queue: asyncio.Queue
):
    """Speech (sine) followed by silence should produce one SpeechChunk."""
    call_id = "test-call-1"

    # Mock TenVad so it returns 1 for non-zero audio, 0 for silence
    with patch("services.vad_worker.TenVad") as MockVad:
        mock_instance = MagicMock()

        def process_side_effect(frame: np.ndarray):
            # If any sample is non-zero, it's speech
            return 1 if np.any(frame != 0) else 0

        mock_instance.process = MagicMock(side_effect=process_side_effect)
        MockVad.return_value = mock_instance

        vad = VADWorker(min_speech_ms=50, min_silence_ms=100)

        # Publish audio: 300ms speech + 200ms silence + sentinel
        speech_audio = _generate_sine(0.3)
        silence_audio = _generate_silence(0.2)

        async def publish():
            # Feed in small increments matching frame size
            for audio in [speech_audio, silence_audio]:
                pcm = np.frombuffer(audio, dtype=np.int16)
                # Publish in frame-sized chunks
                for i in range(0, len(pcm), FRAME_SAMPLES):
                    chunk = pcm[i : i + FRAME_SAMPLES]
                    if len(chunk) == FRAME_SAMPLES:
                        await bus.publish(call_id, chunk.tobytes())
                        await asyncio.sleep(0)  # let event loop run
            await bus.close_channel(call_id)

        publish_task = asyncio.create_task(publish())
        vad_task = asyncio.create_task(
            vad.run(call_id, bus, speech_queue, emitter)
        )

        await asyncio.wait_for(
            asyncio.gather(publish_task, vad_task), timeout=5.0
        )

        # Should have at least one SpeechChunk followed by None sentinel
        items = []
        while not speech_queue.empty():
            items.append(speech_queue.get_nowait())

        # Last item should be the None sentinel
        assert items[-1] is None
        chunks = [i for i in items if i is not None]
        assert len(chunks) >= 1
        assert isinstance(chunks[0], SpeechChunk)
        assert len(chunks[0].audio) > 0


@pytest.mark.anyio
async def test_silence_only_produces_no_chunk(
    bus: AudioBus, emitter: EventEmitter, speech_queue: asyncio.Queue
):
    """Pure silence should produce no SpeechChunks (only the sentinel)."""
    call_id = "test-call-2"

    with patch("services.vad_worker.TenVad") as MockVad:
        mock_instance = MagicMock()
        mock_instance.process = MagicMock(return_value=0)
        MockVad.return_value = mock_instance

        vad = VADWorker(min_speech_ms=50, min_silence_ms=100)
        silence_audio = _generate_silence(0.5)

        async def publish():
            pcm = np.frombuffer(silence_audio, dtype=np.int16)
            for i in range(0, len(pcm), FRAME_SAMPLES):
                chunk = pcm[i : i + FRAME_SAMPLES]
                if len(chunk) == FRAME_SAMPLES:
                    await bus.publish(call_id, chunk.tobytes())
                    await asyncio.sleep(0)
            await bus.close_channel(call_id)

        publish_task = asyncio.create_task(publish())
        vad_task = asyncio.create_task(
            vad.run(call_id, bus, speech_queue, emitter)
        )

        await asyncio.wait_for(
            asyncio.gather(publish_task, vad_task), timeout=5.0
        )

        items = []
        while not speech_queue.empty():
            items.append(speech_queue.get_nowait())

        # Only the sentinel None
        assert items == [None]


@pytest.mark.anyio
async def test_barge_in_emitted_during_tts(
    bus: AudioBus, speech_queue: asyncio.Queue
):
    """When tts_playing is True and speech is detected, a BargeInEvent is emitted."""
    call_id = "test-call-3"
    emitter = EventEmitter()
    eq = await emitter.subscribe(call_id)

    with patch("services.vad_worker.TenVad") as MockVad:
        mock_instance = MagicMock()
        mock_instance.process = MagicMock(return_value=1)
        MockVad.return_value = mock_instance

        vad = VADWorker(min_speech_ms=50, min_silence_ms=100)
        vad.tts_playing = True  # Simulate TTS playing

        speech_audio = _generate_sine(0.15)

        async def publish():
            pcm = np.frombuffer(speech_audio, dtype=np.int16)
            for i in range(0, len(pcm), FRAME_SAMPLES):
                chunk = pcm[i : i + FRAME_SAMPLES]
                if len(chunk) == FRAME_SAMPLES:
                    await bus.publish(call_id, chunk.tobytes())
                    await asyncio.sleep(0)
            # Give time for tasks to run
            await asyncio.sleep(0.1)
            await bus.close_channel(call_id)

        publish_task = asyncio.create_task(publish())
        vad_task = asyncio.create_task(
            vad.run(call_id, bus, speech_queue, emitter)
        )

        await asyncio.wait_for(
            asyncio.gather(publish_task, vad_task), timeout=5.0
        )

        # Allow background tasks to finish
        await asyncio.sleep(0.1)

        # Check that a barge_in event was emitted
        events = []
        while not eq.empty():
            events.append(json.loads(eq.get_nowait()))

        barge_in_events = [e for e in events if e["event"] == "barge_in"]
        assert len(barge_in_events) >= 1
        assert barge_in_events[0]["priority"] == "high"
