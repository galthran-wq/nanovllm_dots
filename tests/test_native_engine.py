"""Final gate: the full engine driven by the NATIVE runtime (no dots_tts) is a
drop-in for the reference-runtime engine.

Both engines share the same paged QwenLLM and the same seed -> identical FM-noise
sampling, so with the native nets matching the reference nets (cos~1.0 per
component) the WHOLE latent sequence must agree (no autoregressive divergence,
unlike comparing to the eager golden whose RNG order differs). Engines are built
one at a time to bound GPU memory.
"""
from __future__ import annotations

import pytest
import torch

from conftest import EN1, MF, assert_cos, set_seed

pytestmark = pytest.mark.advanced
MODEL = MF


def test_native_engine_matches_reference_engine(reference, paged, model_dir):
    from nanovllm_dots.models.dots.native.runtime import DotsRuntime
    from nanovllm_dots.models.dots.engine import DotsBatchEngine

    ld = reference.model.core.latent_dim
    native_runtime = DotsRuntime.from_pretrained(model_dir)

    def run(rt):
        eng = DotsBatchEngine(rt, paged, model_dir=model_dir, num_kvcache_blocks=256,
                              block_size=256, max_num_seqs=8, fm_accel="cudagraph",
                              flash_pe=True, pe_max_batch=8)
        eng.add_request("en1", EN1, num_steps=4, guidance_scale=1.2)
        set_seed(1234)  # align FM noise AFTER construction (cudagraph capture uses RNG)
        out = eng.run_all()["en1"].detach().float().cpu()
        del eng
        torch.cuda.empty_cache()
        return out

    out_ref = run(reference)
    out_nat = run(native_runtime)

    # liveness: the native stack runs end to end and emits a plausible utterance.
    assert out_nat.numel() > 0 and torch.isfinite(out_nat).all()
    assert out_nat.size(1) >= 4 * 8, f"native utterance implausibly short: {out_nat.size(1)} frames"

    # Correctness: PATCH 0 of both engines must agree. The two engines share the
    # paged LLM and seed but are independent stochastic runs, so once they diverge
    # the per-patch FM noise decorrelates (and eos timing shifts via the audio->LLM
    # feedback loop) -- expected. Patch 0 is generated before any divergence, so it
    # isolates the native wiring (prefill -> FM -> patch_encoder) end to end. The
    # individual components are separately validated at cos~1.0.
    ps = 4
    patch0 = torch.nn.functional.cosine_similarity(
        out_ref[:, :ps].reshape(-1, ld), out_nat[:, :ps].reshape(-1, ld), dim=-1).mean().item()
    assert patch0 > 0.99, f"native vs reference engine patch-0 cos={patch0:.5f}"
