from __future__ import annotations

import asyncio
import logging
from functools import cached_property

import numpy as np
import opensmile

from models.analytics_events import ProsodySnapshot

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# openSMILE feature column names from eGeMAPSv02 Functionals
_COL_PITCH_MEAN = "F0semitoneFrom27.5Hz_sma3nz_amean"
_COL_PITCH_STD = "F0semitoneFrom27.5Hz_sma3nz_stddevNorm"
_COL_LOUDNESS = "loudness_sma3_amean"
_COL_MFCC1 = "mfcc1_sma3_amean"

# Chunks with RMS below this are treated as silence.
_SILENCE_RMS_THRESHOLD = 50.0


class ProsodyExtractor:
    """Extract acoustic prosody features from PCM16 chunks via openSMILE.

    All feature extraction runs in a thread executor so the synchronous
    openSMILE C library does not block the async event loop.
    """

    @cached_property
    def _smile(self) -> opensmile.Smile:
        return opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def _extract_sync(self, chunk: np.ndarray) -> ProsodySnapshot:
        """Synchronous extraction — called inside run_in_executor."""
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < _SILENCE_RMS_THRESHOLD:
            return ProsodySnapshot()

        signal = chunk.astype(np.float32) / 32768.0
        df = self._smile.process_signal(signal, SAMPLE_RATE)

        if df.empty:
            return ProsodySnapshot()

        row = df.iloc[0]
        return ProsodySnapshot(
            pitch_mean_hz=float(row.get(_COL_PITCH_MEAN, 0.0)) or None,
            pitch_std_hz=float(row.get(_COL_PITCH_STD, 0.0)) or None,
            rms_energy_db=float(row.get(_COL_LOUDNESS, 0.0)) or None,
            mfcc_1_mean=float(row.get(_COL_MFCC1, 0.0)) or None,
        )

    async def extract(self, chunk: np.ndarray) -> ProsodySnapshot:
        """Extract prosody features without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._extract_sync, chunk)
