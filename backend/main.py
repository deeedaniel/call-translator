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
from services.analytics.analytics_pipeline import AnalyticsPipeline
from services.translation_pipeline import TranslationPipeline
from services.translation_worker import TranslationWorker
from services.vad_worker import VADWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)

_audio_bus = AudioBus()
_event_emitter = EventEmitter()
_pipelines: dict[str, ASRPipeline] = {}
_translation_pipelines: dict[str, TranslationPipeline] = {}
_analytics_pipelines: dict[str, AnalyticsPipeline] = {}

# Eagerly wire up the bus so TestClient and lifespan-less usage both work
twilio_ws.set_audio_bus(_audio_bus)
webrtc_signal.set_audio_bus(_audio_bus)
webrtc_signal.set_event_emitter(_event_emitter)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.getLogger(__name__).info("AudioBus initialised — telephony ingestion ready")
    yield
    for pipeline in list(_analytics_pipelines.values()):
        await pipeline.stop()
    _analytics_pipelines.clear()
    for pipeline in list(_translation_pipelines.values()):
        await pipeline.stop()
    _translation_pipelines.clear()
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


async def start_translation_pipeline(
    call_id: str,
    source_language: str,
    target_language: str,
) -> TranslationPipeline:
    """Launch a translation pipeline for a call."""
    if call_id in _translation_pipelines and _translation_pipelines[call_id].is_running:
        return _translation_pipelines[call_id]

    worker = TranslationWorker(
        nvidia_api_key=settings.nvidia_api_key,
        function_id=settings.nvidia_translate_function_id,
        grpc_endpoint=settings.nvidia_translate_grpc_endpoint,
    )
    pipeline = TranslationPipeline(
        call_id=call_id,
        source_language=source_language,
        target_language=target_language,
        emitter=_event_emitter,
        worker=worker,
    )
    _translation_pipelines[call_id] = pipeline
    await pipeline.start()
    return pipeline


async def stop_translation_pipeline(call_id: str) -> None:
    """Stop and remove the translation pipeline for a call."""
    pipeline = _translation_pipelines.pop(call_id, None)
    if pipeline is not None:
        await pipeline.stop()


async def start_analytics_pipeline(call_id: str) -> AnalyticsPipeline:
    """Launch an analytics pipeline (prosody + KWS → Emitter) for a call."""
    if call_id in _analytics_pipelines and _analytics_pipelines[call_id].is_running:
        return _analytics_pipelines[call_id]

    pipeline = AnalyticsPipeline(
        call_id=call_id,
        audio_bus=_audio_bus,
        emitter=_event_emitter,
        keywords_path=settings.analytics_keywords_path or None,
        chunk_ms=settings.analytics_chunk_ms,
        emit_interval_ms=settings.analytics_emit_interval_ms,
        stress_classifier_type=settings.analytics_stress_classifier,
        ml_model_path=settings.analytics_ml_model_path or None,
    )
    _analytics_pipelines[call_id] = pipeline
    await pipeline.start()
    return pipeline


async def stop_analytics_pipeline(call_id: str) -> None:
    """Stop and remove the analytics pipeline for a call."""
    pipeline = _analytics_pipelines.pop(call_id, None)
    if pipeline is not None:
        await pipeline.stop()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/api/active-calls")
async def active_calls():
    """Return the list of call IDs with running ASR pipelines."""
    return {
        "calls": [
            cid for cid, p in _pipelines.items() if p.is_running
        ]
    }
