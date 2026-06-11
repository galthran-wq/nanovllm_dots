"""Port validation: the native BigVGAN AudioVAE vocoder == the reference one,
one-shot and streaming, on golden latents. The vocoder runs in float32."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from conftest import GOLDEN_MF_DIR, MF, assert_cos

pytestmark = pytest.mark.advanced
MODEL = MF


def _golden_latents(model_dir, ld) -> torch.Tensor:
    """[1, frames, latent_dim] reference latents (golden), trimmed to a multiple
    of latent_patch_size for the streaming chunking."""
    path = os.path.join(GOLDEN_MF_DIR, "en1.latents.npy")
    if not os.path.exists(path):
        pytest.skip(f"golden latents not found: {path}")
    lat = torch.from_numpy(np.load(path)).float().cuda()  # [1, F, ld]
    assert lat.size(-1) == ld
    return lat


def test_native_vocoder_oneshot_matches_reference(reference, model_dir):
    from nanovllm_dots.models.dots.native.loader import build_native_vocoder

    ref_voc = reference.model.vocoder
    nat_voc = build_native_vocoder(model_dir)
    ld = reference.model.core.latent_dim
    lat = _golden_latents(model_dir, ld)

    x = lat.transpose(1, 2).float()  # [1, ld, F], mirrors _decode_latents
    with torch.no_grad():
        w_ref = ref_voc.inference_from_latents(x, do_sample=False).reshape(-1).float()
        w_nat = nat_voc.inference_from_latents(x, do_sample=False).reshape(-1).float()
    assert w_nat.numel() > 0
    assert_cos(w_ref, w_nat, 0.9999, label="native-vocoder one-shot")


def test_native_vocoder_streaming_matches_reference(reference, model_dir):
    from nanovllm_dots.models.dots.native.loader import build_native_vocoder

    ref_voc = reference.model.vocoder
    nat_voc = build_native_vocoder(model_dir)
    core = reference.model.core
    ld, ps = core.latent_dim, core.latent_patch_size
    lat = _golden_latents(model_dir, ld)
    n_patches = lat.size(1) // ps

    ref_state = ref_voc.init_stream_state(batch_size=1, chunk_size=ps)
    nat_state = nat_voc.init_stream_state(batch_size=1, chunk_size=ps)
    ref_chunks, nat_chunks = [], []
    with torch.no_grad():
        for i in range(n_patches):
            patch = lat[:, i * ps:(i + 1) * ps, :].transpose(1, 2).float()  # [1, ld, ps]
            ref_chunks.append(ref_voc.stream_step(patch, ref_state).reshape(-1))
            nat_chunks.append(nat_voc.stream_step(patch, nat_state).reshape(-1))
        ref_chunks.append(ref_voc.stream_flush(ref_state).reshape(-1))
        nat_chunks.append(nat_voc.stream_flush(nat_state).reshape(-1))

    w_ref = torch.cat(ref_chunks).float()
    w_nat = torch.cat(nat_chunks).float()
    assert w_nat.numel() > 0
    assert_cos(w_ref, w_nat, 0.9999, label="native-vocoder streaming")
