"""Point of impact: the streaming BigVGAN vocoder.

The engine vocodes incrementally (step_stream: one stream_step per emitted patch +
a flush). The concatenated streamed audio must reconstruct the SAME waveform as
one-shot decoding the full latent sequence (vocoder.inference_from_latents).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import MF, set_seed

pytestmark = pytest.mark.advanced
MODEL = MF

EN1 = "Hello, this is a streaming synthesis test for the dots text to speech engine."


def test_streaming_vocoder_matches_oneshot(make_engine):
    set_seed(1234)
    eng = make_engine(fm_accel="cudagraph", flash_pe=True)
    eng.add_request("r0", EN1, num_steps=4, guidance_scale=1.2)

    chunks: list[torch.Tensor] = []
    n_done = 0
    for sid, kind, data in eng.generate_stream():
        if kind == "audio":
            chunks.append(data)
        elif kind == "done":
            n_done += 1

    streamed = torch.cat(chunks) if chunks else torch.zeros(0)
    latents = torch.cat(eng._results["r0"], dim=1)   # [1, frames, latent_dim]
    oneshot = eng.vocode(latents)

    n = min(streamed.numel(), oneshot.numel())
    cos = F.cosine_similarity(streamed[:n], oneshot[:n], dim=0).item() if n else 0.0
    assert streamed.numel() > 0, "no streamed audio"
    assert n_done == 1, f"expected one done event, got {n_done}"
    assert cos > 0.999, f"stream-vs-oneshot cos={cos:.5f}"
