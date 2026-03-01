from __future__ import annotations

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from models.asr_events import BaseEvent
from models.translation_events import TranslationEvent


class TestTranslationEvent:
    def test_serialisation_roundtrip(self):
        event = TranslationEvent(
            call_id="CA123",
            seq=1,
            source_text="El paciente tiene dolor en el pecho.",
            translated_text="The patient has chest pain.",
            source_language="es",
            target_language="en",
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["event"] == "translation"
        assert data["call_id"] == "CA123"
        assert data["seq"] == 1
        assert data["source_text"] == "El paciente tiene dolor en el pecho."
        assert data["translated_text"] == "The patient has chest pain."
        assert data["source_language"] == "es"
        assert data["target_language"] == "en"
        assert data["translation_model"] == "riva-translate-1.6b"

        restored = TranslationEvent.model_validate_json(json_str)
        assert restored.seq == 1
        assert restored.translated_text == "The patient has chest pain."

    def test_inherits_base_event(self):
        event = TranslationEvent(
            call_id="CA1",
            seq=0,
            source_text="Hola",
            translated_text="Hello",
            source_language="es",
            target_language="en",
        )
        assert isinstance(event, BaseEvent)
        assert isinstance(event.timestamp_utc, datetime)

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            TranslationEvent(call_id="CA1")

    def test_seq_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            TranslationEvent(
                call_id="CA1",
                seq=-1,
                source_text="a",
                translated_text="b",
                source_language="es",
                target_language="en",
            )

    def test_custom_model_name(self):
        event = TranslationEvent(
            call_id="CA1",
            seq=0,
            source_text="a",
            translated_text="b",
            source_language="es",
            target_language="en",
            translation_model="custom-model",
        )
        assert event.translation_model == "custom-model"
