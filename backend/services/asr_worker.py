from __future__ import annotations

import asyncio
import logging
from typing import Optional

import grpc
import numpy as np
import riva.client

from models.asr_events import TranscriptEvent, WordTimestamp
from services.event_emitter import EventEmitter
from services.vad_worker import SpeechChunk

logger = logging.getLogger(__name__)

# Audio format expected from the VAD worker
SAMPLE_RATE = 16000
ENCODING = riva.client.AudioEncoding.LINEAR_PCM


class ASRWorker:
    """Sends speech chunks to the NVIDIA NIM Parakeet TDT API via gRPC.

    Uses ``nvidia-riva-client`` to connect to the hosted NIM endpoint
    at ``grpc.nvcf.nvidia.com:443`` over SSL with Bearer-token auth.
    """

    def __init__(
        self,
        nvidia_api_key: str,
        function_id: str = "d3fe9151-442b-4204-a70d-5fcc597fd610",
        grpc_endpoint: str = "grpc.nvcf.nvidia.com:443",
    ) -> None:
        self._api_key = nvidia_api_key
        self._function_id = function_id
        self._grpc_endpoint = grpc_endpoint
        self._auth: Optional[riva.client.Auth] = None
        self._asr_service: Optional[riva.client.ASRService] = None
        self._seq = 0

    # ------------------------------------------------------------------ #
    # Connection                                                          #
    # ------------------------------------------------------------------ #

    def _ensure_connection(self) -> None:
        """Lazily initialise the gRPC channel and ASR service."""
        if self._asr_service is not None:
            return

        metadata = [
            ("function-id", self._function_id),
            ("authorization", f"Bearer {self._api_key}"),
        ]

        self._auth = riva.client.Auth(
            ssl_cert=None,
            use_ssl=True,
            uri=self._grpc_endpoint,
            metadata_args=metadata,
        )
        self._asr_service = riva.client.ASRService(self._auth)
        logger.info(
            "ASR worker connected to NIM endpoint %s", self._grpc_endpoint
        )

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        call_id: str,
        speech_queue: asyncio.Queue[SpeechChunk | None],
        emitter: EventEmitter,
    ) -> None:
        """Consume SpeechChunks from the VAD, transcribe, and emit events."""
        self._ensure_connection()
        logger.info("ASR worker started for call %s", call_id)

        while True:
            chunk = await speech_queue.get()
            if chunk is None:
                logger.info("ASR worker finished for call %s", call_id)
                return

            try:
                transcript_event = await asyncio.get_event_loop().run_in_executor(
                    None, self._transcribe_chunk, call_id, chunk
                )
                if transcript_event is not None:
                    await emitter.emit(transcript_event)
            except Exception:
                logger.exception(
                    "ASR transcription failed for call %s", call_id
                )

    # ------------------------------------------------------------------ #
    # Transcription                                                       #
    # ------------------------------------------------------------------ #

    def _transcribe_chunk(
        self, call_id: str, chunk: SpeechChunk
    ) -> TranscriptEvent | None:
        """Blocking call: send audio to NIM and parse the response."""
        assert self._asr_service is not None

        # Convert int16 PCM to bytes (little-endian)
        audio_bytes = chunk.audio.astype(np.int16).tobytes()

        config = riva.client.RecognitionConfig(
            encoding=ENCODING,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
            max_alternatives=1,
            audio_channel_count=1,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        response = self._asr_service.offline_recognize(
            audio_bytes, config
        )

        if not response.results:
            return None

        best = response.results[0].alternatives[0]
        transcript_text = best.transcript.strip()
        if not transcript_text:
            return None

        # Build word-level timestamps
        word_timestamps: list[WordTimestamp] = []
        for w in best.words:
            word_timestamps.append(
                WordTimestamp(
                    word=w.word,
                    start_s=chunk.start_time_s + w.start_time.seconds + w.start_time.nanos / 1e9,
                    end_s=chunk.start_time_s + w.end_time.seconds + w.end_time.nanos / 1e9,
                    confidence=round(w.confidence, 4) if hasattr(w, "confidence") and w.confidence else best.confidence,
                )
            )

        self._seq += 1
        return TranscriptEvent(
            call_id=call_id,
            seq=self._seq,
            original_transcript=transcript_text,
            detected_language="en",
            asr_confidence=round(best.confidence, 4) if best.confidence else 0.0,
            word_timestamps=word_timestamps,
            is_final=True,
        )
