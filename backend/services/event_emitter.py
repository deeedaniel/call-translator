from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from models.asr_events import BaseEvent

logger = logging.getLogger(__name__)


class EventEmitter:
    """In-process async pub/sub for pipeline events.

    Mirrors the AudioBus pattern: multiple subscribers each receive every
    event independently via their own asyncio.Queue.
    """

    def __init__(self, max_queue_size: int = 200) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[str]]] = defaultdict(list)
        self._max_queue_size = max_queue_size
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def emit(self, event: BaseEvent) -> None:
        """Serialise an event to JSON and fan-out to all subscribers for its call_id."""
        json_str = event.model_dump_json()
        call_id = event.call_id

        logger.info("Event emitted [%s] call=%s: %s", event.__class__.__name__, call_id, json_str)

        async with self._get_lock():
            queues = list(self._subscribers.get(call_id, []))

        for q in queues:
            try:
                q.put_nowait(json_str)
            except asyncio.QueueFull:
                # Drop oldest to prevent unbounded backpressure
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(json_str)

    async def subscribe(self, call_id: str) -> asyncio.Queue[str]:
        """Create and register a new subscriber queue for a call's events."""
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=self._max_queue_size)
        async with self._get_lock():
            self._subscribers[call_id].append(q)
        return q

    async def unsubscribe(self, call_id: str, queue: asyncio.Queue[str]) -> None:
        """Remove a subscriber queue. Safe to call multiple times."""
        async with self._get_lock():
            subs = self._subscribers.get(call_id, [])
            if queue in subs:
                subs.remove(queue)
            if not subs and call_id in self._subscribers:
                del self._subscribers[call_id]
