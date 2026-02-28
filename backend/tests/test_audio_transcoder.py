from __future__ import annotations

import numpy as np
import pytest

from services.audio_transcoder import (
    AudioTranscoder,
    linear16_to_ulaw,
    resample_8k_to_16k,
    ulaw_to_linear16,
)


class TestUlawToLinear16:
    def test_silence_byte(self):
        """mu-law 0xFF decodes to 0 (silence)."""
        result = ulaw_to_linear16(bytes([0xFF]))
        assert result[0] == 0

    def test_positive_silence_byte(self):
        """mu-law 0x7F also decodes to 0 (negative silence)."""
        result = ulaw_to_linear16(bytes([0x7F]))
        assert result[0] == 0

    def test_output_dtype(self):
        result = ulaw_to_linear16(bytes([0x00, 0x80, 0xFF]))
        assert result.dtype == np.int16

    def test_output_length_matches_input(self):
        payload = bytes(range(256))
        result = ulaw_to_linear16(payload)
        assert len(result) == 256

    def test_full_table_symmetry(self):
        """First 128 entries are negative, last 128 are positive mirrors."""
        neg = ulaw_to_linear16(bytes(range(0, 128)))
        pos = ulaw_to_linear16(bytes(range(128, 256)))
        np.testing.assert_array_equal(neg, -pos)


class TestResample:
    def test_output_length_doubles(self):
        pcm_8k = np.zeros(160, dtype=np.int16)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        assert len(pcm_16k) == 320

    def test_output_dtype(self):
        pcm_8k = np.zeros(160, dtype=np.int16)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        assert pcm_16k.dtype == np.int16

    def test_silence_stays_silent(self):
        pcm_8k = np.zeros(160, dtype=np.int16)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        np.testing.assert_array_equal(pcm_16k, np.zeros(320, dtype=np.int16))

    def test_preserves_energy(self, sine_8k: np.ndarray):
        pcm_16k = resample_8k_to_16k(sine_8k)
        energy_8k = np.sum(sine_8k.astype(np.float64) ** 2) / len(sine_8k)
        energy_16k = np.sum(pcm_16k.astype(np.float64) ** 2) / len(pcm_16k)
        assert abs(energy_16k - energy_8k) / energy_8k < 0.05


class TestRoundTrip:
    def test_encode_decode_preserves_waveform(self, sine_8k: np.ndarray):
        """Encode to mu-law and decode back; verify waveform correlation."""
        encoded = linear16_to_ulaw(sine_8k)
        decoded = ulaw_to_linear16(encoded)

        sine_f = sine_8k.astype(np.float64)
        decoded_f = decoded.astype(np.float64)
        correlation = np.corrcoef(sine_f, decoded_f)[0, 1]
        assert correlation > 0.99


class TestAudioTranscoder:
    def test_transcode_output_is_bytes(self, transcoder: AudioTranscoder):
        mulaw_payload = bytes([0xFF] * 160)
        result = transcoder.transcode(mulaw_payload)
        assert isinstance(result, bytes)

    def test_transcode_output_length(self, transcoder: AudioTranscoder):
        mulaw_payload = bytes([0xFF] * 160)
        result = transcoder.transcode(mulaw_payload)
        # 160 mu-law samples -> 160 PCM16 -> 320 resampled -> 320 * 2 bytes
        assert len(result) == 320 * 2

    def test_reverse_transcode(self, transcoder: AudioTranscoder):
        pcm16_16k = np.zeros(320, dtype=np.int16).tobytes()
        result = transcoder.reverse_transcode(pcm16_16k)
        assert isinstance(result, bytes)
        assert len(result) == 160
