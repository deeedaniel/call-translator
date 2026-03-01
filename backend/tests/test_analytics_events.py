from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models.analytics_events import (
    AnalyticsEvent,
    DetectedKeyword,
    ProsodySnapshot,
)


class TestProsodySnapshot:
    def test_all_none_defaults(self):
        snap = ProsodySnapshot()
        assert snap.pitch_mean_hz is None
        assert snap.rms_energy_db is None

    def test_with_values(self):
        snap = ProsodySnapshot(
            pitch_mean_hz=250.0,
            pitch_std_hz=30.0,
            rms_energy_db=-14.2,
            mfcc_1_mean=5.5,
        )
        assert snap.pitch_mean_hz == 250.0
        assert snap.mfcc_1_mean == 5.5


class TestDetectedKeyword:
    def test_defaults(self):
        kw = DetectedKeyword(keyword="fire")
        assert kw.keyword == "fire"
        assert kw.source == "transcript"
        assert kw.transcript_seq is None
        assert isinstance(kw.detected_at_utc, datetime)

    def test_with_seq(self):
        kw = DetectedKeyword(keyword="fire", transcript_seq=3)
        assert kw.transcript_seq == 3

    def test_serialisation(self):
        kw = DetectedKeyword(keyword="chest pain", transcript_seq=5)
        data = json.loads(kw.model_dump_json())
        assert data["keyword"] == "chest pain"
        assert data["source"] == "transcript"
        assert data["transcript_seq"] == 5


class TestAnalyticsEvent:
    def test_serialisation_roundtrip(self):
        event = AnalyticsEvent(
            call_id="CA123",
            seq=0,
            stress_level="high",
            stress_confidence=0.92,
            prosody=ProsodySnapshot(pitch_mean_hz=340.0, rms_energy_db=-8.0),
            keywords_detected=[DetectedKeyword(keyword="fire")],
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["event"] == "analytics"
        assert data["call_id"] == "CA123"
        assert data["stress_level"] == "high"
        assert data["prosody"]["pitch_mean_hz"] == 340.0
        assert len(data["keywords_detected"]) == 1
        assert data["keywords_detected"][0]["keyword"] == "fire"

        restored = AnalyticsEvent.model_validate_json(json_str)
        assert restored.seq == 0
        assert restored.stress_level == "high"
        assert restored.keywords_detected[0].keyword == "fire"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AnalyticsEvent(call_id="CA1")

    def test_auto_timestamp(self):
        event = AnalyticsEvent(
            call_id="CA1", seq=0, stress_level="low", stress_confidence=0.5
        )
        assert isinstance(event.timestamp_utc, datetime)

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            AnalyticsEvent(
                call_id="CA1",
                seq=0,
                stress_level="medium",
                stress_confidence=1.5,
            )
        with pytest.raises(ValidationError):
            AnalyticsEvent(
                call_id="CA1",
                seq=0,
                stress_level="medium",
                stress_confidence=-0.1,
            )

    def test_invalid_stress_level(self):
        with pytest.raises(ValidationError):
            AnalyticsEvent(
                call_id="CA1",
                seq=0,
                stress_level="critical",
                stress_confidence=0.5,
            )

    def test_empty_keywords_default(self):
        event = AnalyticsEvent(
            call_id="CA1", seq=5, stress_level="low", stress_confidence=0.6
        )
        assert event.keywords_detected == []
        assert event.prosody.pitch_mean_hz is None

    def test_seq_non_negative(self):
        with pytest.raises(ValidationError):
            AnalyticsEvent(
                call_id="CA1",
                seq=-1,
                stress_level="low",
                stress_confidence=0.5,
            )
