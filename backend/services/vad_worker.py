from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np
from ten_vad import TenVad

from models.asr_events import BargeInEvent, VADStateEvent
from services.audio_bus import AudioBus
from services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

# TEN VAD operates on 160-sample frames at 16 kHz (10 ms per frame).
FRAME_SAMPLES = 160
SAMPLE_RATE = 16000
FRAME_DURATION_S = FRAME_SAMPLES / SAMPLE_RATE  # 0.01 s

# Default ring buffer holds ~30 s of audio.
RING_BUFFER_SECONDS = 30
RING_BUFFER_SAMPLES = RING_BUFFER_SECONDS * SAMPLE_RATE


@dataclass
class SpeechChunk:
    """A contiguous segment of speech audio extracted by the VAD."""

    audio: np.ndarray  # PCM16 16 kHz, dtype=int16
    start_time_s: float
    end_time_s: float


class VADWorker:
    """Consumes PCM16 16 kHz frames from an AudioBus and emits speech chunks.

    Uses TEN VAD with probabilistic hysteresis to determine speech
    start/end boundaries. Also emits barge-in events when speech is
    detected during TTS playback.

    Parameters
    ----------
    min_speech_ms : int
        Consecutive speech frames required to confirm speech start.
    min_silence_ms : int
        Consecutive silence frames required to confirm speech end.
    """

    def __init__(
        self,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
    ) -> None:
        self._vad = TenVad(SAMPLE_RATE)
        self._min_speech_frames = int(min_speech_ms / (FRAME_DURATION_S * 1000))
        self._min_silence_frames = int(min_silence_ms / (FRAME_DURATION_S * 1000))

        # Ring buffer for raw audio
        self._ring = np.zeros(RING_BUFFER_SAMPLES, dtype=np.int16)
        self._ring_pos = 0  # write cursor (wraps)
        self._total_samples: int = 0  # monotonic sample counter

        # Hysteresis state
        self._speaking = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._speech_start_sample: int = 0

        # External flag: set to True while TTS is playing so barge-in fires
        self.tts_playing = False

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        call_id: str,
        audio_bus: AudioBus,
        speech_queue: asyncio.Queue[SpeechChunk | None],
        emitter: EventEmitter,
    ) -> None:
        """Main loop: subscribe to AudioBus, run VAD, produce SpeechChunks."""
        async with audio_bus.listen(call_id) as bus_queue:
            logger.info("VAD worker started for call %s", call_id)
            leftover = np.array([], dtype=np.int16)

            while True:
                pcm_bytes = await bus_queue.get()
                if pcm_bytes is None:
                    # End of stream — flush any in-progress speech
                    if self._speaking:
                        chunk = self._flush_speech()
                        if chunk is not None:
                            speech_queue.put_nowait(chunk)
                    speech_queue.put_nowait(None)  # sentinel
                    logger.info("VAD worker finished for call %s", call_id)
                    return

                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                # Prepend any leftover samples from previous iteration
                if len(leftover) > 0:
                    pcm = np.concatenate([leftover, pcm])
                    leftover = np.array([], dtype=np.int16)

                offset = 0
                while offset + FRAME_SAMPLES <= len(pcm):
                    frame = pcm[offset : offset + FRAME_SAMPLES]
                    self._write_ring(frame)
                    is_speech = self._process_frame(frame)
                    self._update_hysteresis(
                        is_speech, call_id, speech_queue, emitter
                    )
                    offset += FRAME_SAMPLES

                # Keep remaining samples for next iteration
                if offset < len(pcm):
                    leftover = pcm[offset:]

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _process_frame(self, frame: np.ndarray) -> bool:
        """Run TEN VAD on a single 160-sample frame. Returns True if speech."""
        result = self._vad.process(frame)
        return bool(result)

    def _write_ring(self, frame: np.ndarray) -> None:
        """Append samples to the ring buffer."""
        n = len(frame)
        end = self._ring_pos + n
        if end <= RING_BUFFER_SAMPLES:
            self._ring[self._ring_pos : end] = frame
        else:
            first = RING_BUFFER_SAMPLES - self._ring_pos
            self._ring[self._ring_pos :] = frame[:first]
            self._ring[: n - first] = frame[first:]
        self._ring_pos = end % RING_BUFFER_SAMPLES
        self._total_samples += n

    def _update_hysteresis(
        self,
        is_speech: bool,
        call_id: str,
        speech_queue: asyncio.Queue[SpeechChunk | None],
        emitter: EventEmitter,
    ) -> None:
        """Apply hysteresis logic and emit events on state transitions."""
        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0

        if not self._speaking:
            if self._consecutive_speech >= self._min_speech_frames:
                # ---- Speech START ----
                self._speaking = True
                self._speech_start_sample = self._total_samples - (
                    self._consecutive_speech * FRAME_SAMPLES
                )
                logger.info("VAD: speech start at sample %d", self._speech_start_sample)

                # Fire barge-in if TTS is currently playing
                if self.tts_playing:
                    event = BargeInEvent(call_id=call_id)
                    asyncio.get_event_loop().create_task(emitter.emit(event))

                vad_event = VADStateEvent(
                    call_id=call_id,
                    speaking=True,
                    speech_start_s=self._speech_start_sample / SAMPLE_RATE,
                )
                asyncio.get_event_loop().create_task(emitter.emit(vad_event))
        else:
            if self._consecutive_silence >= self._min_silence_frames:
                # ---- Speech END ----
                chunk = self._flush_speech()
                if chunk is not None:
                    try:
                        speech_queue.put_nowait(chunk)
                    except asyncio.QueueFull:
                        logger.warning("Speech queue full, dropping chunk")

                vad_event = VADStateEvent(
                    call_id=call_id,
                    speaking=False,
                    speech_start_s=self._speech_start_sample / SAMPLE_RATE,
                )
                asyncio.get_event_loop().create_task(emitter.emit(vad_event))

    def _flush_speech(self) -> SpeechChunk | None:
        """Extract buffered speech audio and reset state."""
        speech_end_sample = self._total_samples - (
            self._consecutive_silence * FRAME_SAMPLES
        )
        speech_len = speech_end_sample - self._speech_start_sample
        if speech_len <= 0:
            self._speaking = False
            self._consecutive_speech = 0
            self._consecutive_silence = 0
            return None

        # Cap to ring buffer capacity
        speech_len = min(speech_len, RING_BUFFER_SAMPLES)

        # Read from ring buffer
        end_pos = self._ring_pos - (self._total_samples - speech_end_sample)
        end_pos %= RING_BUFFER_SAMPLES
        start_pos = end_pos - speech_len
        if start_pos < 0:
            start_pos += RING_BUFFER_SAMPLES

        if start_pos < end_pos:
            audio = self._ring[start_pos:end_pos].copy()
        else:
            audio = np.concatenate([
                self._ring[start_pos:],
                self._ring[:end_pos],
            ])

        chunk = SpeechChunk(
            audio=audio,
            start_time_s=self._speech_start_sample / SAMPLE_RATE,
            end_time_s=speech_end_sample / SAMPLE_RATE,
        )

        self._speaking = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        logger.info(
            "VAD: speech chunk %.2fs – %.2fs (%d samples)",
            chunk.start_time_s,
            chunk.end_time_s,
            len(audio),
        )
        return chunk
