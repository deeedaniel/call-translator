from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import twiml, twilio_ws, webrtc_signal
from services.audio_bus import AudioBus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)

_audio_bus = AudioBus()

# Eagerly wire up the bus so TestClient and lifespan-less usage both work
twilio_ws.set_audio_bus(_audio_bus)
webrtc_signal.set_audio_bus(_audio_bus)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.getLogger(__name__).info("AudioBus initialised — telephony ingestion ready")
    yield
    logging.getLogger(__name__).info("Shutting down telephony ingestion")


app = FastAPI(title="Call Translator — Telephony Ingestion", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(twiml.router)
app.include_router(twilio_ws.router)
app.include_router(webrtc_signal.router)


def get_audio_bus() -> AudioBus:
    return _audio_bus


@app.get("/")
async def root():
    return {"message": "Hello World"}
