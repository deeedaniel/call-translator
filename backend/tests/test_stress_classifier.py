from __future__ import annotations

import pytest

from models.analytics_events import ProsodySnapshot
from services.analytics.stress_classifier import RuleBasedStressClassifier


@pytest.fixture
def classifier() -> RuleBasedStressClassifier:
    return RuleBasedStressClassifier()


class TestRuleBasedStressClassifier:
    @pytest.mark.parametrize(
        "snapshot, expected_level",
        [
            pytest.param(
                ProsodySnapshot(),
                "low",
                id="silence",
            ),
            pytest.param(
                ProsodySnapshot(pitch_mean_hz=15.0, rms_energy_db=0.1),
                "high",
                id="high-pitch",
            ),
            pytest.param(
                ProsodySnapshot(pitch_mean_hz=3.0, rms_energy_db=0.9),
                "high",
                id="high-loudness",
            ),
            pytest.param(
                ProsodySnapshot(pitch_mean_hz=8.0, rms_energy_db=0.3),
                "medium",
                id="medium-pitch",
            ),
            pytest.param(
                ProsodySnapshot(pitch_mean_hz=3.0, rms_energy_db=0.5),
                "medium",
                id="medium-loudness",
            ),
            pytest.param(
                ProsodySnapshot(pitch_mean_hz=3.0, rms_energy_db=0.1),
                "low",
                id="calm",
            ),
        ],
    )
    def test_classification(
        self,
        classifier: RuleBasedStressClassifier,
        snapshot: ProsodySnapshot,
        expected_level: str,
    ):
        level, confidence = classifier.classify(snapshot)
        assert level == expected_level
        assert 0.0 <= confidence <= 1.0

    def test_silence_confidence(self, classifier: RuleBasedStressClassifier):
        level, confidence = classifier.classify(ProsodySnapshot())
        assert level == "low"
        assert confidence == 0.5

    def test_high_stress_confidence_bounded(self, classifier: RuleBasedStressClassifier):
        snap = ProsodySnapshot(pitch_mean_hz=100.0, rms_energy_db=5.0)
        level, confidence = classifier.classify(snap)
        assert level == "high"
        assert confidence <= 1.0

    def test_custom_thresholds(self):
        strict = RuleBasedStressClassifier(
            pitch_high=5.0, pitch_medium=2.0,
            loudness_high=0.3, loudness_medium=0.1,
        )
        snap = ProsodySnapshot(pitch_mean_hz=6.0, rms_energy_db=0.05)
        level, _ = strict.classify(snap)
        assert level == "high"
