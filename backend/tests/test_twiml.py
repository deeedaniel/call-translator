from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_voice_webhook_returns_xml():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/twiml/voice")

    assert resp.status_code == 200
    assert "application/xml" in resp.headers["content-type"]


@pytest.mark.anyio
async def test_voice_webhook_contains_stream_element():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/twiml/voice")

    root = ET.fromstring(resp.text)
    stream_el = root.find(".//Stream")
    assert stream_el is not None

    url = stream_el.get("url")
    assert url is not None
    assert url.startswith("wss://")
    assert "/ws/twilio" in url


@pytest.mark.anyio
async def test_voice_webhook_contains_say_element():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/twiml/voice")

    root = ET.fromstring(resp.text)
    say_el = root.find(".//Say")
    assert say_el is not None
    assert "interpreter" in say_el.text.lower()
