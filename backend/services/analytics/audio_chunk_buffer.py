from __future__ import annotations

import asyncio
import logging

import numpy as np

from services.audio_bus import AudioBus

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2  # PCM16


class AudioChunkBuffer:
    """Subscribes to an AudioBus and yields fixed-size PCM16 numpy chunks.

    Parameters
    ----------
    chunk_ms : int
        Duration of each output chunk in milliseconds (default 500).
    """

    def __init__(self, chunk_ms: int = 500) -> None:
        self._chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self._chunk_bytes = self._chunk_samples * BYTES_PER_SAMPLE
        self._buf = bytearray()

    async def run(
        self,
        call_id: str,
        audio_bus: AudioBus,
        out_queue: asyncio.Queue[np.ndarray | None],
    ) -> None:
        """Consume PCM16 frames and emit fixed-size numpy chunks."""
        async with audio_bus.listen(call_id) as bus_queue:
            logger.info(
                "AudioChunkBuffer started for call %s (%d ms chunks)",
                call_id,
                self._chunk_samples * 1000 // SAMPLE_RATE,
            )
            while True:
                pcm_bytes = await bus_queue.get()
                if pcm_bytes is None:
                    self._flush(out_queue)
                    out_queue.put_nowait(None)
                    logger.info("AudioChunkBuffer finished for call %s", call_id)
                    return

                self._buf.extend(pcm_bytes)
                while len(self._buf) >= self._chunk_bytes:
                    raw = bytes(self._buf[: self._chunk_bytes])
                    del self._buf[: self._chunk_bytes]
                    chunk = np.frombuffer(raw, dtype=np.int16).copy()
                    try:
                        out_queue.put_nowait(chunk)
                    except asyncio.QueueFull:
                        logger.warning("Chunk queue full, dropping oldest chunk")
                        try:
                            out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        out_queue.put_nowait(chunk)

    def _flush(self, out_queue: asyncio.Queue[np.ndarray | None]) -> None:
        """Flush any remaining bytes as a zero-padded chunk."""
        if not self._buf:
            return
        padded = bytearray(self._chunk_bytes)
        padded[: len(self._buf)] = self._buf
        self._buf.clear()
        chunk = np.frombuffer(bytes(padded), dtype=np.int16).copy()
        try:
            out_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass
