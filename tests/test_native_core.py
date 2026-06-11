"""Port validation: native DotsCore projections / eos head / embeddings load
correctly and match the reference core."""
from __future__ import annotations

import pytest
import torch

from conftest import MF, assert_cos, set_seed

pytestmark = pytest.mark.advanced
MODEL = MF


def test_native_core_components_match_reference(reference, model_dir):
    from transformers import AutoTokenizer
    from nanovllm_dots.models.dots.native.core import DotsCore

    tok = AutoTokenizer.from_pretrained(model_dir)
    nat = DotsCore.from_pretrained(model_dir, tok)
    ref = reference.model.core
    dev, dt = torch.device("cuda"), torch.bfloat16

    # config scalars
    assert nat.latent_dim == ref.latent_dim
    assert nat.latent_patch_size == ref.latent_patch_size
    assert nat.fm_hidden_size == ref.fm_hidden_size
    assert list(nat.audio_span_token_ids) == list(ref.audio_span_token_ids)

    set_seed(1234)
    with torch.no_grad():
        x = torch.randn(1, 4, nat.latent_dim, device=dev, dtype=dt)
        assert_cos(ref.coordinate_proj(x), nat.coordinate_proj(x), 0.9999, label="coordinate_proj")
        h = torch.randn(1, nat.fm_hidden_size, device=dev, dtype=dt)  # reuse fm_hidden==llm_hidden? no
        hh = torch.randn(1, nat.llm_hidden_size, device=dev, dtype=dt)
        assert_cos(ref.hidden_proj(hh), nat.hidden_proj(hh), 0.9999, label="hidden_proj")
        assert_cos(ref.latent_proj(x), nat.latent_proj(x), 0.9999, label="latent_proj")
        xv = torch.randn(1, nat.xvec_dim, device=dev, dtype=dt)
        assert_cos(ref.xvec_proj(xv), nat.xvec_proj(xv), 0.9999, label="xvec_proj")
        assert_cos(ref.eos_proj(hh), nat.eos_proj(hh), 0.999, label="eos_proj")
        ids = torch.tensor([[1, 5, 99, 1234]], device=dev)
        e_ref = ref.llm.get_input_embeddings()(ids)
        e_nat = nat.embed_tokens(ids)
        assert_cos(e_ref.reshape(-1, e_ref.size(-1)), e_nat.reshape(-1, e_nat.size(-1)),
                   0.9999, label="embed_tokens")
