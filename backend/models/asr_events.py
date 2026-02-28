from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """A single word with its time boundaries and per-word confidence."""

    word: str
    start_s: float
    end_s: float
    confidence: float = Field(ge=0.0, le=1.0)


class BaseASREvent(BaseModel):
    """Common fields shared by every ASR pipeline event."""

    call_id: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TranscriptEvent(BaseASREvent):
    """Emitted when the ASR worker produces a final or partial transcript."""

    event: Literal["transcript"] = "transcript"
    seq: int = Field(ge=0)
    original_transcript: str
    detected_language: str = "en"
    asr_confidence: float = Field(ge=0.0, le=1.0)
    word_timestamps: list[WordTimestamp] = Field(default_factory=list)
    is_final: bool = True


class BargeInEvent(BaseASREvent):
    """Emitted when the VAD detects caller speech during TTS playback."""

    event: Literal["barge_in"] = "barge_in"
    priority: Literal["high"] = "high"
    reason: str = "caller_speech_detected_during_playback"


class VADStateEvent(BaseASREvent):
    """Optional debugging event tracking VAD state transitions."""

    event: Literal["vad_state"] = "vad_state"
    speaking: bool
    speech_start_s: Optional[float] = None
