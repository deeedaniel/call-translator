from __future__ import annotations

from typing import Literal

from pydantic import Field

from models.asr_events import BaseEvent


class TranslationEvent(BaseEvent):
    """Emitted when a translated text chunk is ready for downstream TTS."""

    event: Literal["translation"] = "translation"
    seq: int = Field(ge=0)
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    translation_model: str = "riva-translate-1.6b"
