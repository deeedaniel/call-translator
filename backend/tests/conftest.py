from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from services.audio_transcoder import AudioTranscoder


@pytest.fixture
def transcoder() -> AudioTranscoder:
    return AudioTranscoder()


@pytest.fixture
def sine_8k() -> np.ndarray:
    """Generate a 400 Hz sine wave at 8 kHz sample rate, 100 ms duration."""
    t = np.arange(0, 0.1, 1 / 8000)
    amplitude = 16000
    return (amplitude * np.sin(2 * np.pi * 400 * t)).astype(np.int16)
