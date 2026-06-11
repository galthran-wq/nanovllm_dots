"""WAV (PCM16) encoding helpers for the dots.tts HTTP responses."""
from __future__ import annotations

import struct

import numpy as np
import torch


def pcm16(wav: torch.Tensor) -> bytes:
    """Float waveform in [-1, 1] -> little-endian 16-bit PCM bytes."""
    a = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


def wav_header(sample_rate: int, data_bytes: int, *, channels: int = 1,
               bits: int = 16) -> bytes:
    """44-byte WAV header.

    For streaming, pass ``data_bytes < 0`` to emit a max placeholder length
    (0xFFFFFFFF) so the header can go out before the total length is known --
    players read until the stream ends.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    riff = 0xFFFFFFFF if data_bytes < 0 else 36 + data_bytes
    data = 0xFFFFFFFF if data_bytes < 0 else data_bytes
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                    byte_rate, block_align, bits)
            + b"data" + struct.pack("<I", data))
