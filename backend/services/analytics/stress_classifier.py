from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from models.analytics_events import ProsodySnapshot

logger = logging.getLogger(__name__)

StressLevel = Literal["low", "medium", "high"]


class StressClassifier(ABC):
    """Interface for prosody-based stress/urgency classification."""

    @abstractmethod
    def classify(self, snapshot: ProsodySnapshot) -> tuple[StressLevel, float]:
        """Return ``(stress_level, confidence)`` for the given prosody snapshot.

        Confidence is in [0.0, 1.0].
        """


class RuleBasedStressClassifier(StressClassifier):
    """Threshold-based heuristic using pitch and loudness.

    Designed as a day-one implementation that can be swapped for an
    ML model (``MLStressClassifier``) once a labelled dataset is available.

    Parameters
    ----------
    pitch_high : float
        Pitch (semitones from 27.5 Hz) above which stress is "high".
    pitch_medium : float
        Pitch above which stress is "medium".
    loudness_high : float
        Loudness (openSMILE ``loudness_sma3_amean``) above which stress is "high".
    loudness_medium : float
        Loudness above which stress is "medium".
    """

    def __init__(
        self,
        pitch_high: float = 12.0,
        pitch_medium: float = 7.0,
        loudness_high: float = 0.8,
        loudness_medium: float = 0.4,
    ) -> None:
        self._pitch_high = pitch_high
        self._pitch_medium = pitch_medium
        self._loudness_high = loudness_high
        self._loudness_medium = loudness_medium

    def classify(self, snapshot: ProsodySnapshot) -> tuple[StressLevel, float]:
        if snapshot.pitch_mean_hz is None and snapshot.rms_energy_db is None:
            return ("low", 0.5)

        pitch = snapshot.pitch_mean_hz or 0.0
        loudness = snapshot.rms_energy_db or 0.0

        if pitch >= self._pitch_high or loudness >= self._loudness_high:
            ref = max(
                pitch / self._pitch_high if self._pitch_high else 0.0,
                loudness / self._loudness_high if self._loudness_high else 0.0,
            )
            confidence = min(0.5 + 0.5 * ref, 1.0)
            return ("high", round(confidence, 3))

        if pitch >= self._pitch_medium or loudness >= self._loudness_medium:
            ref = max(
                pitch / self._pitch_high if self._pitch_high else 0.0,
                loudness / self._loudness_high if self._loudness_high else 0.0,
            )
            confidence = min(0.5 + 0.5 * ref, 1.0)
            return ("medium", round(confidence, 3))

        return ("low", 0.7)


class MLStressClassifier(StressClassifier):
    """Drop-in replacement that loads a pre-trained scikit-learn model.

    Expected workflow:
    1. Train an SVM/RandomForest on labelled openSMILE feature vectors.
    2. Serialise with ``joblib.dump(model, "stress_model.joblib")``.
    3. Pass the path here — inference uses the same ``ProsodySnapshot`` input.
    """

    def __init__(self, model_path: Path) -> None:
        import joblib

        self._model = joblib.load(model_path)
        logger.info("Loaded ML stress model from %s", model_path)

    def classify(self, snapshot: ProsodySnapshot) -> tuple[StressLevel, float]:
        features = [
            snapshot.pitch_mean_hz or 0.0,
            snapshot.pitch_std_hz or 0.0,
            snapshot.rms_energy_db or 0.0,
            snapshot.mfcc_1_mean or 0.0,
        ]
        import numpy as np

        X = np.array(features).reshape(1, -1)
        label: str = self._model.predict(X)[0]

        if label not in ("low", "medium", "high"):
            label = "low"

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)
            confidence = float(proba.max())
        else:
            confidence = 0.8

        return (label, round(confidence, 3))  # type: ignore[return-value]
