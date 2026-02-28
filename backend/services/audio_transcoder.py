from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

# ITU-T G.711 mu-law expansion table.
# Index = mu-law compressed byte (0–255), value = 16-bit linear PCM sample.
MULAW_TABLE = np.array([
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364,  -9852,  -9340,  -8828,  -8316,
     -7932,  -7676,  -7420,  -7164,  -6908,  -6652,  -6396,  -6140,
     -5884,  -5628,  -5372,  -5116,  -4860,  -4604,  -4348,  -4092,
     -3900,  -3772,  -3644,  -3516,  -3388,  -3260,  -3132,  -3004,
     -2876,  -2748,  -2620,  -2492,  -2364,  -2236,  -2108,  -1980,
     -1884,  -1820,  -1756,  -1692,  -1628,  -1564,  -1500,  -1436,
     -1372,  -1308,  -1244,  -1180,  -1116,  -1052,   -988,   -924,
      -876,   -844,   -812,   -780,   -748,   -716,   -684,   -652,
      -620,   -588,   -556,   -524,   -492,   -460,   -428,   -396,
      -372,   -356,   -340,   -324,   -308,   -292,   -276,   -260,
      -244,   -228,   -212,   -196,   -180,   -164,   -148,   -132,
      -120,   -112,   -104,    -96,    -88,    -80,    -72,    -64,
       -56,    -48,    -40,    -32,    -24,    -16,     -8,      0,
     32124,  31100,  30076,  29052,  28028,  27004,  25980,  24956,
     23932,  22908,  21884,  20860,  19836,  18812,  17788,  16764,
     15996,  15484,  14972,  14460,  13948,  13436,  12924,  12412,
     11900,  11388,  10876,  10364,   9852,   9340,   8828,   8316,
      7932,   7676,   7420,   7164,   6908,   6652,   6396,   6140,
      5884,   5628,   5372,   5116,   4860,   4604,   4348,   4092,
      3900,   3772,   3644,   3516,   3388,   3260,   3132,   3004,
      2876,   2748,   2620,   2492,   2364,   2236,   2108,   1980,
      1884,   1820,   1756,   1692,   1628,   1564,   1500,   1436,
      1372,   1308,   1244,   1180,   1116,   1052,    988,    924,
       876,    844,    812,    780,    748,    716,    684,    652,
       620,    588,    556,    524,    492,    460,    428,    396,
       372,    356,    340,    324,    308,    292,    276,    260,
       244,    228,    212,    196,    180,    164,    148,    132,
       120,    112,    104,     96,     88,     80,     72,     64,
        56,     48,     40,     32,     24,     16,      8,      0,
], dtype=np.int16)


def ulaw_to_linear16(payload: bytes) -> np.ndarray:
    """Expand mu-law encoded bytes to 16-bit signed linear PCM samples."""
    indices = np.frombuffer(payload, dtype=np.uint8)
    return MULAW_TABLE[indices]


def linear16_to_ulaw(pcm: np.ndarray) -> bytes:
    """Compress 16-bit signed linear PCM samples to mu-law bytes.

    Uses the algorithmic approach specified in ITU-T G.711.
    """
    BIAS = 0x84
    CLIP = 32635

    pcm = pcm.astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0).astype(np.int32)
    sample = np.clip(np.abs(pcm), 0, CLIP) + BIAS

    # Segment number from highest set bit position: bit_pos - 7, clamped to [0, 7].
    # Minimum biased sample is 132 (bit 7), maximum is 32767 (bit 14).
    highest_bit = np.floor(np.log2(np.maximum(sample, 1))).astype(np.int32)
    exponent = np.clip(highest_bit - 7, 0, 7)

    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw_byte.astype(np.uint8).tobytes()


def resample_8k_to_16k(pcm_8k: np.ndarray) -> np.ndarray:
    """Upsample PCM audio from 8000 Hz to 16000 Hz."""
    resampled = resample_poly(pcm_8k.astype(np.float64), up=2, down=1)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


class AudioTranscoder:
    """Transcodes mu-law 8 kHz audio to PCM16 linear 16 kHz.

    Designed for the Twilio Media Stream -> NeMo Parakeet STT pipeline.
    """

    def transcode(self, mulaw_payload: bytes) -> bytes:
        """Convert raw mu-law bytes to PCM16 little-endian bytes at 16 kHz."""
        pcm_8k = ulaw_to_linear16(mulaw_payload)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        return pcm_16k.tobytes()

    def reverse_transcode(self, pcm16_16k: bytes) -> bytes:
        """Convert PCM16 LE 16 kHz bytes back to mu-law 8 kHz bytes.

        Used for sending operator audio back through the Twilio stream.
        """
        pcm_16k = np.frombuffer(pcm16_16k, dtype=np.int16)
        pcm_8k = resample_poly(pcm_16k.astype(np.float64), up=1, down=2)
        pcm_8k = np.clip(pcm_8k, -32768, 32767).astype(np.int16)
        return linear16_to_ulaw(pcm_8k)
