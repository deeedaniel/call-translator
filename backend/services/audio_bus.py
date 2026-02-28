from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AudioBus:
    """In-process async pub/sub for PCM16 audio frames, keyed by call_id.

    One publisher (e.g. Twilio stream) pushes frames; multiple subscribers
    (e.g. STT engine, WebRTC track) each receive every frame independently.
    """

    def __init__(self, max_queue_size: int = 200) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[bytes | None]]] = defaultdict(
            list
        )
        self._max_queue_size = max_queue_size
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def publish(self, call_id: str, frame: bytes) -> None:
        """Fan-out a PCM16 frame to every subscriber for this call_id."""
        async with self._get_lock():
            queues = self._subscribers.get(call_id, [])
        for q in queues:
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                # Drop oldest frame to prevent unbounded backpressure
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(frame)

    async def subscribe(self, call_id: str) -> asyncio.Queue[bytes | None]:
        """Create and register a new subscriber queue for a call."""
        q: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=self._max_queue_size
        )
        async with self._get_lock():
            self._subscribers[call_id].append(q)
        return q

    async def unsubscribe(self, call_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue. Safe to call multiple times."""
        async with self._get_lock():
            subs = self._subscribers.get(call_id, [])
            if queue in subs:
                subs.remove(queue)
            if not subs and call_id in self._subscribers:
                del self._subscribers[call_id]

    async def close_channel(self, call_id: str) -> None:
        """Signal end-of-stream to all subscribers of a call by sending None."""
        async with self._get_lock():
            queues = list(self._subscribers.get(call_id, []))
        for q in queues:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    @asynccontextmanager
    async def listen(self, call_id: str) -> AsyncIterator[asyncio.Queue[bytes | None]]:
        """Context manager that auto-unsubscribes on exit."""
        q = await self.subscribe(call_id)
        try:
            yield q
        finally:
            await self.unsubscribe(call_id, q)
