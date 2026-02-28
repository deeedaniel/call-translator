from __future__ import annotations

import asyncio

import pytest

from services.audio_bus import AudioBus


@pytest.fixture
def bus() -> AudioBus:
    return AudioBus(max_queue_size=50)


@pytest.mark.anyio
async def test_single_subscriber_receives_all_frames(bus: AudioBus):
    q = await bus.subscribe("call-1")
    frames = [f"frame-{i}".encode() for i in range(10)]
    for f in frames:
        await bus.publish("call-1", f)

    received = [q.get_nowait() for _ in range(10)]
    assert received == frames
    await bus.unsubscribe("call-1", q)


@pytest.mark.anyio
async def test_multiple_subscribers_each_receive_all(bus: AudioBus):
    q1 = await bus.subscribe("call-1")
    q2 = await bus.subscribe("call-1")

    frames = [f"frame-{i}".encode() for i in range(5)]
    for f in frames:
        await bus.publish("call-1", f)

    received_1 = [q1.get_nowait() for _ in range(5)]
    received_2 = [q2.get_nowait() for _ in range(5)]
    assert received_1 == frames
    assert received_2 == frames

    await bus.unsubscribe("call-1", q1)
    await bus.unsubscribe("call-1", q2)


@pytest.mark.anyio
async def test_unsubscribe_stops_delivery(bus: AudioBus):
    q = await bus.subscribe("call-1")
    await bus.publish("call-1", b"before")
    await bus.unsubscribe("call-1", q)
    await bus.publish("call-1", b"after")

    assert q.qsize() == 1
    assert q.get_nowait() == b"before"


@pytest.mark.anyio
async def test_no_cross_talk_between_calls(bus: AudioBus):
    q1 = await bus.subscribe("call-A")
    q2 = await bus.subscribe("call-B")

    await bus.publish("call-A", b"for-A")
    await bus.publish("call-B", b"for-B")

    assert q1.get_nowait() == b"for-A"
    assert q2.get_nowait() == b"for-B"
    assert q1.qsize() == 0
    assert q2.qsize() == 0

    await bus.unsubscribe("call-A", q1)
    await bus.unsubscribe("call-B", q2)


@pytest.mark.anyio
async def test_close_channel_sends_sentinel(bus: AudioBus):
    q = await bus.subscribe("call-1")
    await bus.publish("call-1", b"data")
    await bus.close_channel("call-1")

    assert q.get_nowait() == b"data"
    assert q.get_nowait() is None

    await bus.unsubscribe("call-1", q)


@pytest.mark.anyio
async def test_listen_context_manager(bus: AudioBus):
    async with bus.listen("call-1") as q:
        await bus.publish("call-1", b"hello")
        assert q.get_nowait() == b"hello"

    # After exiting, publishing should not reach the old queue
    await bus.publish("call-1", b"gone")
    assert q.qsize() == 0


@pytest.mark.anyio
async def test_drops_oldest_on_full_queue():
    bus = AudioBus(max_queue_size=3)
    q = await bus.subscribe("call-1")

    for i in range(5):
        await bus.publish("call-1", f"f{i}".encode())

    items = []
    while not q.empty():
        items.append(q.get_nowait())

    assert len(items) == 3
    assert items[-1] == b"f4"

    await bus.unsubscribe("call-1", q)
