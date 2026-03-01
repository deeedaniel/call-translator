from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from models.asr_events import BaseEvent


class ProsodySnapshot(BaseModel):
    """Acoustic features extracted from a single analysis window."""

    pitch_mean_hz: Optional[float] = None
    pitch_std_hz: Optional[float] = None
    rms_energy_db: Optional[float] = None
    mfcc_1_mean: Optional[float] = None


class DetectedKeyword(BaseModel):
    """A keyword detected via transcript matching against the ASR stream."""

    keyword: str
    detected_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    source: Literal["transcript"] = "transcript"
    transcript_seq: int | None = None


class AnalyticsEvent(BaseEvent):
    """Emitted every 500 ms with the latest stress/urgency assessment."""

    event: Literal["analytics"] = "analytics"
    seq: int = Field(ge=0)
    stress_level: Literal["low", "medium", "high"]
    stress_confidence: float = Field(ge=0.0, le=1.0)
    prosody: ProsodySnapshot = Field(default_factory=ProsodySnapshot)
    keywords_detected: list[DetectedKeyword] = Field(default_factory=list)
