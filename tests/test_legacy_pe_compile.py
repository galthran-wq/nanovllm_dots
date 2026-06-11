"""Legacy: torch.compile(patch_encoder.decode_patch) == eager, per patch.

A compile variant of the patch_encoder (the shipping path uses FlashPatchEncoder,
not a compiled decode_patch). Kept for regression coverage. Runs on mf.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import EN1, MF, set_seed

pytestmark = [pytest.mark.advanced, pytest.mark.legacy]
MODEL = MF


def test_pe_compile_matches_eager(make_engine):
    eng = make_engine(fm_accel="none")
    pe = eng.core.patch_encoder
    eager_decode = pe.decode_patch
    compiled_decode = torch.compile(pe.decode_patch, dynamic=False)
    coss: list[float] = []

    def patched(state, patch):
        dots, core = eng.dots, eng.core
        dots._append_history_chunk(state, patch)
        cur = 0 if state.patch_encoder_state is None else state.patch_encoder_state.seq_len
        dots._ensure_patch_encoder_state_capacity(
            state, required_seq_len=cur + core.patch_encoder.out_ds_rate,
            device=eng.device, dtype=eng.dtype)
        patch_for_llm = core.io_helper.denormalize(patch)
        positions = torch.arange(core.patch_encoder.out_ds_rate, device=eng.device, dtype=torch.long) \
            + state.patch_encoder_state.seq_len
        e_eag, tail_e = eager_decode(patch_for_llm, state.patch_encoder_state.conv_tail,
                                     state.patch_encoder_state.layer_caches, positions)
        e_cmp, _ = compiled_decode(patch_for_llm, state.patch_encoder_state.conv_tail,
                                   state.patch_encoder_state.layer_caches, positions)
        coss.append(F.cosine_similarity(e_eag[0].float(), e_cmp[0].float(), dim=-1).min().item())
        state.patch_encoder_state.conv_tail.copy_(tail_e)
        state.patch_encoder_state.seq_len += core.patch_encoder.out_ds_rate
        return e_eag[0]

    set_seed(1234)
    eng._patch_to_embed = patched
    eng.add_request("r0", EN1, num_steps=4, guidance_scale=1.2)
    eng.run_all()

    assert coss, "no patches produced"
    assert min(coss) > 0.999, f"compiled-vs-eager embed cos min={min(coss):.6f}"
