"""Port validation: the native DiT == the reference velocity_field_predictor.

Same code, same weights, identical seeded inputs -> outputs must match at the
bf16 floor (they are the same computation). This is the first native module that
lets the engine stop importing dots_tts for the FM head.
"""
from __future__ import annotations

import pytest
import torch

from conftest import MF, assert_cos, set_seed

pytestmark = pytest.mark.advanced
MODEL = MF  # mf = meanflow DiT (exercises the duration_embedder branch)


def test_native_dit_matches_reference(reference, model_dir):
    from nanovllm_dots.models.dots.native.loader import build_native_dit

    core = reference.model.core
    ref_dit = core.velocity_field_predictor
    native = build_native_dit(model_dir)

    H, ld = core.fm_hidden_size, core.latent_dim
    dev, dt = torch.device("cuda"), torch.bfloat16
    is_meanflow = getattr(ref_dit, "duration_embedder", None) is not None

    for B in (1, 2):
        for L in (8, 30, 90):
            set_seed(1000 * B + L)
            x = torch.randn(B, L, H, device=dev, dtype=dt)
            t = torch.rand(B, device=dev, dtype=dt)
            dur = torch.rand(B, device=dev, dtype=dt) if is_meanflow else None
            g = torch.randn(B, H, device=dev, dtype=dt)
            pos = torch.arange(L, device=dev).float().unsqueeze(0).expand(B, -1).contiguous()
            mask = torch.ones(B, L, L, dtype=torch.bool, device=dev).tril()
            with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                o_ref = ref_dit(x=x, timesteps=t, duration=dur, attn_mask=mask,
                                pos_ids=pos, g_cond=g).float()
                o_nat = native(x=x, timesteps=t, duration=dur, attn_mask=mask,
                               pos_ids=pos, g_cond=g).float()
            assert_cos(o_ref, o_nat, 0.9999, latent_dim=ld, label=f"native-dit B={B} L={L}")


def test_native_dit_drop_in_accelerator(reference, model_dir):
    """The native DiT is a drop-in for the FM accelerator: batched_meanflow with
    vfp=native == vfp=reference (proves the accelerator path uses it unchanged)."""
    import types
    from nanovllm_dots.models.dots.native.loader import build_native_dit
    from nanovllm_dots.models.dots.batched_fm import batched_meanflow

    dots = reference.model
    core = dots.core
    native = build_native_dit(model_dir)
    H, patch, ld = core.fm_hidden_size, core.latent_patch_size, core.latent_dim
    dev, dt = torch.device("cuda"), torch.bfloat16

    for L in (12, 28):
        set_seed(L)
        hist = torch.randn(1, L, H, device=dev, dtype=dt)
        noise = torch.randn(1, patch, ld, device=dev, dtype=dt)
        total = L + patch
        inp = torch.zeros(1, total, H, device=dev, dtype=dt)
        inp[0, :L] = hist[0, :L]
        mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
        pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
        st = types.SimpleNamespace(fm_seq_len=L)
        dots._build_fm_attn_mask(state=st, attn_mask=mask)
        dots._build_fm_pos_ids(state=st, pos_ids=pos)
        g = torch.zeros(1, H, device=dev, dtype=dt)
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            ref = batched_meanflow(core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                                   g_cond=g, num_steps=4, noise=noise.clone())[0].float()
            nat = batched_meanflow(core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                                   g_cond=g, num_steps=4, noise=noise.clone(), vfp=native)[0].float()
        assert_cos(ref, nat, 0.9999, label=f"native-dit-in-meanflow L={L}")
