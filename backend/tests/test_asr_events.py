from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models.asr_events import (
    BargeInEvent,
    TranscriptEvent,
    VADStateEvent,
    WordTimestamp,
)


class TestWordTimestamp:
    def test_valid(self):
        wt = WordTimestamp(word="hello", start_s=0.1, end_s=0.3, confidence=0.95)
        assert wt.word == "hello"
        assert wt.confidence == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            WordTimestamp(word="x", start_s=0.0, end_s=0.1, confidence=1.5)
        with pytest.raises(ValidationError):
            WordTimestamp(word="x", start_s=0.0, end_s=0.1, confidence=-0.1)


class TestTranscriptEvent:
    def test_serialisation_roundtrip(self):
        event = TranscriptEvent(
            call_id="CA123",
            seq=1,
            original_transcript="help",
            detected_language="en",
            asr_confidence=0.99,
            word_timestamps=[
                WordTimestamp(word="help", start_s=0.0, end_s=0.4, confidence=0.99),
            ],
            is_final=True,
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)
        assert data["event"] == "transcript"
        assert data["call_id"] == "CA123"
        assert data["original_transcript"] == "help"
        assert len(data["word_timestamps"]) == 1

        # Round-trip back to model
        restored = TranscriptEvent.model_validate_json(json_str)
        assert restored.seq == 1
        assert restored.word_timestamps[0].word == "help"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            TranscriptEvent(call_id="CA123")  # missing seq, transcript, confidence

    def test_auto_timestamp(self):
        event = TranscriptEvent(
            call_id="CA1",
            seq=0,
            original_transcript="hi",
            asr_confidence=0.5,
        )
        assert isinstance(event.timestamp_utc, datetime)

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            TranscriptEvent(
                call_id="CA1",
                seq=0,
                original_transcript="hi",
                asr_confidence=2.0,
            )


class TestBargeInEvent:
    def test_defaults(self):
        event = BargeInEvent(call_id="CA999")
        assert event.event == "barge_in"
        assert event.priority == "high"
        assert "playback" in event.reason

    def test_json_output(self):
        event = BargeInEvent(call_id="CA1")
        data = json.loads(event.model_dump_json())
        assert data["event"] == "barge_in"
        assert data["priority"] == "high"


class TestVADStateEvent:
    def test_speaking_true(self):
        event = VADStateEvent(call_id="CA1", speaking=True, speech_start_s=1.5)
        assert event.speaking is True
        assert event.speech_start_s == 1.5

    def test_speaking_false(self):
        event = VADStateEvent(call_id="CA1", speaking=False)
        assert event.speaking is False
        assert event.speech_start_s is None
