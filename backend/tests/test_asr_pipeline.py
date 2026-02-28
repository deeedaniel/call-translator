from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from models.asr_events import TranscriptEvent
from services.asr_pipeline import ASRPipeline
from services.asr_worker import ASRWorker
from services.audio_bus import AudioBus
from services.event_emitter import EventEmitter
from services.vad_worker import FRAME_SAMPLES, SAMPLE_RATE, VADWorker


def _generate_sine(duration_s: float, freq: float = 400.0) -> bytes:
    t = np.arange(0, duration_s, 1 / SAMPLE_RATE)
    samples = (16000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return samples.tobytes()


def _generate_silence(duration_s: float) -> bytes:
    n_samples = int(duration_s * SAMPLE_RATE)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_mock_asr_response(transcript: str = "hello world"):
    """Create a mock Riva offline_recognize response."""
    mock_word = MagicMock()
    mock_word.word = "hello"
    mock_word.start_time.seconds = 0
    mock_word.start_time.nanos = 100_000_000
    mock_word.end_time.seconds = 0
    mock_word.end_time.nanos = 400_000_000
    mock_word.confidence = 0.98

    mock_word2 = MagicMock()
    mock_word2.word = "world"
    mock_word2.start_time.seconds = 0
    mock_word2.start_time.nanos = 450_000_000
    mock_word2.end_time.seconds = 0
    mock_word2.end_time.nanos = 800_000_000
    mock_word2.confidence = 0.96

    mock_alt = MagicMock()
    mock_alt.transcript = transcript
    mock_alt.confidence = 0.97
    mock_alt.words = [mock_word, mock_word2]

    mock_result = MagicMock()
    mock_result.alternatives = [mock_alt]

    mock_response = MagicMock()
    mock_response.results = [mock_result]
    return mock_response


@pytest.mark.anyio
async def test_pipeline_end_to_end_with_mocks():
    """Full pipeline: AudioBus → VAD → ASR → EventEmitter, all mocked."""
    bus = AudioBus(max_queue_size=500)
    emitter = EventEmitter()
    call_id = "test-e2e"
    eq = await emitter.subscribe(call_id)

    # Mock TEN VAD
    with patch("services.vad_worker.TenVad") as MockVad:
        mock_vad_instance = MagicMock()

        def vad_side_effect(frame):
            return 1 if np.any(frame != 0) else 0

        mock_vad_instance.process = MagicMock(side_effect=vad_side_effect)
        MockVad.return_value = mock_vad_instance

        # Mock Riva client
        with patch("services.asr_worker.riva") as mock_riva:
            mock_asr_service = MagicMock()
            mock_asr_service.offline_recognize = MagicMock(
                return_value=_make_mock_asr_response("hello world")
            )
            mock_riva.client.ASRService.return_value = mock_asr_service
            mock_riva.client.Auth.return_value = MagicMock()
            mock_riva.client.AudioEncoding.LINEAR_PCM = 1
            mock_riva.client.RecognitionConfig = MagicMock()

            vad = VADWorker(min_speech_ms=50, min_silence_ms=100)
            asr = ASRWorker(nvidia_api_key="test-key")

            pipeline = ASRPipeline(
                call_id=call_id,
                audio_bus=bus,
                emitter=emitter,
                asr_worker=asr,
                vad_worker=vad,
            )

            await pipeline.start()

            # Publish speech + silence + close
            speech = _generate_sine(0.3)
            silence = _generate_silence(0.2)

            for audio in [speech, silence]:
                pcm = np.frombuffer(audio, dtype=np.int16)
                for i in range(0, len(pcm), FRAME_SAMPLES):
                    chunk = pcm[i : i + FRAME_SAMPLES]
                    if len(chunk) == FRAME_SAMPLES:
                        await bus.publish(call_id, chunk.tobytes())
                        await asyncio.sleep(0)

            await bus.close_channel(call_id)

            # Wait for pipeline to finish
            await asyncio.sleep(1.0)

            # Collect events
            events = []
            while not eq.empty():
                events.append(json.loads(eq.get_nowait()))

            # Should have at least one transcript event
            transcript_events = [e for e in events if e["event"] == "transcript"]
            assert len(transcript_events) >= 1
            assert transcript_events[0]["original_transcript"] == "hello world"
            assert len(transcript_events[0]["word_timestamps"]) == 2

            await pipeline.stop()


@pytest.mark.anyio
async def test_pipeline_stop_cancels_tasks():
    """Verify that stopping a pipeline cancels running tasks."""
    bus = AudioBus()
    emitter = EventEmitter()

    with patch("services.vad_worker.TenVad") as MockVad:
        MockVad.return_value = MagicMock()
        MockVad.return_value.process = MagicMock(return_value=0)

        with patch("services.asr_worker.riva") as mock_riva:
            mock_riva.client.ASRService.return_value = MagicMock()
            mock_riva.client.Auth.return_value = MagicMock()
            mock_riva.client.AudioEncoding.LINEAR_PCM = 1

            vad = VADWorker()
            asr = ASRWorker(nvidia_api_key="test-key")

            pipeline = ASRPipeline(
                call_id="test-stop",
                audio_bus=bus,
                emitter=emitter,
                asr_worker=asr,
                vad_worker=vad,
            )

            await pipeline.start()
            assert pipeline.is_running

            await pipeline.stop()
            assert not pipeline.is_running
