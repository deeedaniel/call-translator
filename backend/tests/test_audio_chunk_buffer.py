from __future__ import annotations

import asyncio

import numpy as np
import pytest

from services.analytics.audio_chunk_buffer import AudioChunkBuffer, SAMPLE_RATE
from services.audio_bus import AudioBus


@pytest.fixture
def audio_bus() -> AudioBus:
    return AudioBus()


@pytest.fixture
def chunk_buffer() -> AudioChunkBuffer:
    return AudioChunkBuffer(chunk_ms=500)


def _make_pcm_bytes(n_samples: int) -> bytes:
    """Generate n_samples of silent PCM16 as bytes (non-zero to distinguish from padding)."""
    return (np.ones(n_samples, dtype=np.int16) * 1000).tobytes()


class TestAudioChunkBuffer:
    @pytest.mark.asyncio
    async def test_exact_chunk_boundary(self, audio_bus: AudioBus, chunk_buffer: AudioChunkBuffer):
        """Publishing exactly 8000 samples yields exactly one 8000-sample chunk."""
        out: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        call_id = "test-exact"

        samples_per_chunk = int(SAMPLE_RATE * 0.5)  # 8000

        async def _publish():
            await audio_bus.publish(call_id, _make_pcm_bytes(samples_per_chunk))
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await chunk_buffer.run(call_id, audio_bus, out)

        chunk = await out.get()
        assert isinstance(chunk, np.ndarray)
        assert len(chunk) == samples_per_chunk
        assert np.all(chunk == 1000)

        sentinel = await out.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_multiple_small_frames(self, audio_bus: AudioBus, chunk_buffer: AudioChunkBuffer):
        """Many small frames are accumulated into complete chunks."""
        out: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        call_id = "test-small"
        frame_size = 160  # 10 ms
        samples_per_chunk = int(SAMPLE_RATE * 0.5)
        frames_needed = samples_per_chunk // frame_size  # 50 frames = 500ms

        async def _publish():
            for _ in range(frames_needed * 2):  # 2 full chunks
                await audio_bus.publish(call_id, _make_pcm_bytes(frame_size))
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await chunk_buffer.run(call_id, audio_bus, out)

        c1 = await out.get()
        c2 = await out.get()
        assert len(c1) == samples_per_chunk
        assert len(c2) == samples_per_chunk

        sentinel = await out.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_partial_flush_zero_padded(self, audio_bus: AudioBus, chunk_buffer: AudioChunkBuffer):
        """Leftover samples at end-of-stream are zero-padded to chunk size."""
        out: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        call_id = "test-partial"
        samples_per_chunk = int(SAMPLE_RATE * 0.5)
        partial = 1000  # less than 8000

        async def _publish():
            await audio_bus.publish(call_id, _make_pcm_bytes(partial))
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await chunk_buffer.run(call_id, audio_bus, out)

        chunk = await out.get()
        assert len(chunk) == samples_per_chunk
        assert np.all(chunk[:partial] == 1000)
        assert np.all(chunk[partial:] == 0)

        sentinel = await out.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_empty_stream(self, audio_bus: AudioBus, chunk_buffer: AudioChunkBuffer):
        """Closing the channel immediately sends only the sentinel."""
        out: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        call_id = "test-empty"

        async def _publish():
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await chunk_buffer.run(call_id, audio_bus, out)

        sentinel = await out.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_custom_chunk_ms(self, audio_bus: AudioBus):
        """A 1000ms buffer yields 16000-sample chunks."""
        buf = AudioChunkBuffer(chunk_ms=1000)
        out: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        call_id = "test-1s"
        expected_samples = SAMPLE_RATE  # 16000

        async def _publish():
            await audio_bus.publish(call_id, _make_pcm_bytes(expected_samples))
            await audio_bus.close_channel(call_id)

        asyncio.create_task(_publish())
        await buf.run(call_id, audio_bus, out)

        chunk = await out.get()
        assert len(chunk) == expected_samples
