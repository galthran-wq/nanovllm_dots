"""Port validation: the native CAM++ speaker x-vector encoder == the reference.

The x-vector is what voice cloning conditions on (xvector -> core.xvec_proj ->
g_cond). The encoder runs in float32; we compare under bf16 autocast to mirror
how _prepare_prompt_conditioning calls it.
"""
from __future__ import annotations

import os

import pytest
import torch

from conftest import GOLDEN_MF_DIR, MF, assert_cos

pytestmark = pytest.mark.advanced
MODEL = MF

REF_A = os.path.join(GOLDEN_MF_DIR, "en1.wav")
REF_B = os.path.join(GOLDEN_MF_DIR, "zh1.wav")


def test_native_speaker_xvector_matches_reference(reference, model_dir):
    from nanovllm_dots.models.dots.native.loader import build_native_speaker

    for ref_wav in (REF_A, REF_B):
        if not os.path.exists(ref_wav):
            pytest.skip(f"reference audio not found: {ref_wav}")

    ref_se = reference.model.xvector_extractor
    nat_se = build_native_speaker(model_dir)

    for ref_wav in (REF_A, REF_B):
        pa = reference._load_prompt_audio(ref_wav).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x_ref = ref_se(pa[None, :]).reshape(-1).float()
            x_nat = nat_se(pa[None, :]).reshape(-1).float()
        assert x_nat.numel() > 0
        assert_cos(x_ref, x_nat, 0.9999, label=f"native-speaker xvector {os.path.basename(ref_wav)}")
