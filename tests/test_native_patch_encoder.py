"""Port validation: the native VAE patch_encoder == the reference one, and is a
drop-in for FlashPatchEncoder (which binds to its leaves)."""
from __future__ import annotations

import pytest
import torch

from conftest import MF, assert_cos, set_seed

pytestmark = pytest.mark.advanced
MODEL = MF


def test_native_patch_encoder_decode_matches_reference(reference, model_dir):
    from nanovllm_dots.models.dots.native.loader import build_native_patch_encoder

    core = reference.model.core
    ref_pe = core.patch_encoder
    nat_pe = build_native_patch_encoder(model_dir)
    dev, dt = torch.device("cuda"), torch.bfloat16
    ps, ld = core.latent_patch_size, core.latent_dim

    set_seed(1234)
    ref_st = ref_pe.init_decode_state(max_audio_patch_count=256, batch_size=1, device=dev, dtype=dt)
    nat_st = nat_pe.init_decode_state(max_audio_patch_count=256, batch_size=1, device=dev, dtype=dt)
    for _ in range(30):
        patch = torch.randn(1, ps, ld, device=dev, dtype=dt)
        p_ref = torch.arange(ref_pe.out_ds_rate, device=dev, dtype=torch.long) + ref_st.seq_len
        r_emb, r_ct = ref_pe.decode_patch(patch, ref_st.conv_tail, ref_st.layer_caches, p_ref)
        ref_st.conv_tail.copy_(r_ct); ref_st.seq_len += ref_pe.out_ds_rate
        p_nat = torch.arange(nat_pe.out_ds_rate, device=dev, dtype=torch.long) + nat_st.seq_len
        n_emb, n_ct = nat_pe.decode_patch(patch, nat_st.conv_tail, nat_st.layer_caches, p_nat)
        nat_st.conv_tail.copy_(n_ct); nat_st.seq_len += nat_pe.out_ds_rate
        assert_cos(r_emb.reshape(-1, r_emb.size(-1)), n_emb.reshape(-1, n_emb.size(-1)),
                   0.9999, label="native-pe decode")


def test_native_patch_encoder_prefill_matches_reference(reference, model_dir):
    from nanovllm_dots.models.dots.native.loader import build_native_patch_encoder

    core = reference.model.core
    ref_pe = core.patch_encoder
    nat_pe = build_native_patch_encoder(model_dir)
    dev, dt = torch.device("cuda"), torch.bfloat16
    ps, ld = core.latent_patch_size, core.latent_dim
    P = 22

    set_seed(1234)
    prompt = torch.randn(1, P * ps, ld, device=dev, dtype=dt)
    ref_st = ref_pe.init_decode_state(max_audio_patch_count=256, batch_size=1, device=dev, dtype=dt)
    nat_st = nat_pe.init_decode_state(max_audio_patch_count=256, batch_size=1, device=dev, dtype=dt)
    with torch.autocast("cuda", dtype=dt):
        r_emb, ref_st = ref_pe.prefill(prompt, ref_st)
        n_emb, nat_st = nat_pe.prefill(prompt, nat_st)
    assert_cos(r_emb.reshape(-1, r_emb.size(-1)), n_emb.reshape(-1, n_emb.size(-1)),
               0.9999, label="native-pe prefill")

    # continuity: decode after prefill stays matched
    for _ in range(10):
        patch = torch.randn(1, ps, ld, device=dev, dtype=dt)
        p_ref = torch.arange(ref_pe.out_ds_rate, device=dev, dtype=torch.long) + ref_st.seq_len
        with torch.autocast("cuda", dtype=dt):
            r_e, r_ct = ref_pe.decode_patch(patch, ref_st.conv_tail, ref_st.layer_caches, p_ref)
        ref_st.conv_tail.copy_(r_ct); ref_st.seq_len += ref_pe.out_ds_rate
        p_nat = torch.arange(nat_pe.out_ds_rate, device=dev, dtype=torch.long) + nat_st.seq_len
        with torch.autocast("cuda", dtype=dt):
            n_e, n_ct = nat_pe.decode_patch(patch, nat_st.conv_tail, nat_st.layer_caches, p_nat)
        nat_st.conv_tail.copy_(n_ct); nat_st.seq_len += nat_pe.out_ds_rate
        assert_cos(r_e.reshape(-1), n_e.reshape(-1), 0.9999, label="native-pe post-prefill decode")


def test_native_pe_drop_in_flash(reference, model_dir):
    """FlashPatchEncoder built on the native pe == built on the reference pe
    (proves the accelerator's leaf bindings resolve identically)."""
    from nanovllm_dots.models.dots.native.loader import build_native_patch_encoder
    from nanovllm_dots.models.dots.flash_patch_encoder import FlashPatchEncoder

    core = reference.model.core
    ref_pe = core.patch_encoder
    nat_pe = build_native_patch_encoder(model_dir)
    dev, dt = torch.device("cuda"), torch.bfloat16
    ps, ld = core.latent_patch_size, core.latent_dim
    cap = 256 * ref_pe.out_ds_rate

    flash_ref = FlashPatchEncoder(ref_pe, max_batch=1, max_seq_len=cap, device=dev, dtype=dt)
    flash_nat = FlashPatchEncoder(nat_pe, max_batch=1, max_seq_len=cap, device=dev, dtype=dt)
    rows = torch.tensor([0], device=dev, dtype=torch.int32)
    set_seed(1234)
    for _ in range(20):
        patch = torch.randn(1, ps, ld, device=dev, dtype=dt)
        e_ref = flash_ref.decode_patch(patch, rows)
        e_nat = flash_nat.decode_patch(patch, rows)
        assert_cos(e_ref.reshape(-1), e_nat.reshape(-1), 0.9999, label="native-pe flash drop-in")
