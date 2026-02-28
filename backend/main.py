from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import twiml, twilio_ws, webrtc_signal
from services.asr_pipeline import ASRPipeline
from services.asr_worker import ASRWorker
from services.audio_bus import AudioBus
from services.event_emitter import EventEmitter
from services.vad_worker import VADWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)

_audio_bus = AudioBus()
_event_emitter = EventEmitter()
_pipelines: dict[str, ASRPipeline] = {}

# Eagerly wire up the bus so TestClient and lifespan-less usage both work
twilio_ws.set_audio_bus(_audio_bus)
webrtc_signal.set_audio_bus(_audio_bus)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.getLogger(__name__).info("AudioBus initialised — telephony ingestion ready")
    yield
    # Stop all running ASR pipelines on shutdown
    for pipeline in list(_pipelines.values()):
        await pipeline.stop()
    _pipelines.clear()
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


def get_event_emitter() -> EventEmitter:
    return _event_emitter


async def start_asr_pipeline(call_id: str) -> ASRPipeline:
    """Launch an ASR pipeline (VAD → ASR → Emitter) for a call."""
    if call_id in _pipelines and _pipelines[call_id].is_running:
        return _pipelines[call_id]

    vad = VADWorker(
        min_speech_ms=settings.vad_min_speech_ms,
        min_silence_ms=settings.vad_min_silence_ms,
    )
    asr = ASRWorker(
        nvidia_api_key=settings.nvidia_api_key,
        function_id=settings.nvidia_asr_function_id,
        grpc_endpoint=settings.nvidia_grpc_endpoint,
    )
    pipeline = ASRPipeline(
        call_id=call_id,
        audio_bus=_audio_bus,
        emitter=_event_emitter,
        asr_worker=asr,
        vad_worker=vad,
    )
    _pipelines[call_id] = pipeline
    await pipeline.start()
    return pipeline


async def stop_asr_pipeline(call_id: str) -> None:
    """Stop and remove the ASR pipeline for a call."""
    pipeline = _pipelines.pop(call_id, None)
    if pipeline is not None:
        await pipeline.stop()


@app.get("/")
async def root():
    return {"message": "Hello World"}
