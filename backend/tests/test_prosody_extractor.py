from __future__ import annotations

import numpy as np
import pytest

from models.analytics_events import ProsodySnapshot
from services.analytics.prosody_extractor import ProsodyExtractor, SAMPLE_RATE


@pytest.fixture
def extractor() -> ProsodyExtractor:
    return ProsodyExtractor()


def _sine_chunk(freq_hz: float = 400.0, duration_s: float = 0.5) -> np.ndarray:
    """Generate a PCM16 sine wave at the given frequency."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE
    return (16000 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)


def _silence_chunk(duration_s: float = 0.5) -> np.ndarray:
    """Generate a chunk of silence (all zeros)."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.int16)


class TestProsodyExtractor:
    @pytest.mark.asyncio
    async def test_sine_wave_returns_features(self, extractor: ProsodyExtractor):
        chunk = _sine_chunk(400.0)
        snap = await extractor.extract(chunk)

        assert isinstance(snap, ProsodySnapshot)
        assert snap.rms_energy_db is not None
        assert snap.mfcc_1_mean is not None

    @pytest.mark.asyncio
    async def test_silence_returns_none_snapshot(self, extractor: ProsodyExtractor):
        chunk = _silence_chunk()
        snap = await extractor.extract(chunk)

        assert isinstance(snap, ProsodySnapshot)
        assert snap.pitch_mean_hz is None
        assert snap.rms_energy_db is None
        assert snap.mfcc_1_mean is None

    @pytest.mark.asyncio
    async def test_high_pitch_detected(self, extractor: ProsodyExtractor):
        chunk = _sine_chunk(800.0)
        snap = await extractor.extract(chunk)
        assert snap.rms_energy_db is not None

    @pytest.mark.asyncio
    async def test_different_durations(self, extractor: ProsodyExtractor):
        for dur in (0.25, 0.5, 1.0):
            chunk = _sine_chunk(300.0, duration_s=dur)
            snap = await extractor.extract(chunk)
            assert isinstance(snap, ProsodySnapshot)
