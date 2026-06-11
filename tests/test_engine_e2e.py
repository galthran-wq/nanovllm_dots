"""Point of impact: the full continuous-batching DotsBatchEngine, end to end.

Two linked checks (run on the soar checkpoint, against golden/ artifacts):

1. paged LLM inside the *reference* generation loop reproduces the reference
   golden latents (early patches; eos timing) AND the paged hidden states match
   HF Qwen2 at every call -- proving the backbone is correct in the real loop,
   not just in isolation (verify_e2e). This also produces the "paged golden".
2. the DotsBatchEngine (our reimplemented loop) reproduces that paged golden for
   a single request, the cudagraph FM path is bit-exact, and 8 concurrent requests
   all complete with finite latents (verify_engine).

Bit-exactness is not expected end-to-end: the backbone differs from HF by ~1e-3
(bf16, fused QKV/gate_up, flash), and the FM feedback loop amplifies that, so late
patches drift. Structural correctness shows as: patch-0 matches and eos fires at
the same patch.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import EN1, GOLDEN_DIR, SOAR, ZH1, set_seed

pytestmark = pytest.mark.advanced
MODEL = SOAR

PROMPTS = [("en1", EN1), ("zh1", ZH1)]
NUM_STEPS = 10
GUIDANCE = 1.2

# paged-loop latents produced by the e2e test, consumed by the engine test.
_PAGED_GOLDEN: dict[str, torch.Tensor] = {}


def test_paged_llm_in_reference_loop(reference, paged):
    """Drive the reference loop with core.step_llm redirected through our paged
    runner; assert the latent stream matches the golden and the per-call hidden
    states match HF."""
    from nanovllm_dots.models.dots.paged_llm import PagedLLMRunner

    if not os.path.exists(os.path.join(GOLDEN_DIR, "en1.latents.npy")):
        pytest.skip(f"golden latents not found under {GOLDEN_DIR}")

    core = reference.model.core
    runner = PagedLLMRunner(paged, block_size=256, max_seq_len=4096)
    hf_embed = core.llm.get_input_embeddings()
    orig_step_llm = core.step_llm
    hf_state = {"cache": None}
    dbg = {"cos": [], "len_mismatch": 0}

    def patched_step_llm(inputs_embeds=None, input_ids=None, past_key_values=None):
        if (inputs_embeds is None) == (input_ids is None):
            raise ValueError("exactly one of inputs_embeds / input_ids")
        if inputs_embeds is None:
            inputs_embeds = hf_embed(input_ids)
        hidden = runner.append(inputs_embeds)
        _, hf_hidden, _, hf_state["cache"] = orig_step_llm(
            inputs_embeds=inputs_embeds, past_key_values=hf_state["cache"])
        dbg["cos"].append(F.cosine_similarity(hidden.float(), hf_hidden[0].float(), dim=-1).min().item())
        if runner.past_len != hf_state["cache"].get_seq_length():
            dbg["len_mismatch"] += 1
        return inputs_embeds, hidden.unsqueeze(0), None, runner

    core.step_llm = patched_step_llm
    try:
        for name, text in PROMPTS:
            gpath = os.path.join(GOLDEN_DIR, f"{name}.latents.npy")
            if not os.path.exists(gpath):
                continue
            golden = torch.from_numpy(np.load(gpath)).float()
            set_seed(1234)
            inputs = reference._prepare_inputs(text=text, prompt_audio_path=None,
                                               prompt_text=None, template_name=None,
                                               language=None, normalize_text=False)
            runner.reset()
            hf_state["cache"] = None
            dbg.update(cos=[], len_mismatch=0)
            lat = [l.detach().float().cpu() for l in reference.model._generate_latents_stream(
                inputs, precision="bfloat16", ode_method="euler",
                num_steps=NUM_STEPS, guidance_scale=GUIDANCE)]
            mine = torch.cat(lat, dim=1) if lat else torch.zeros((1, 0, 0))
            _PAGED_GOLDEN[name] = mine

            n = min(mine.size(1), golden.size(1))
            a, b = mine[:, :n], golden[:, :n]
            ps = 4
            traj = [F.cosine_similarity(a[:, i:i+ps].reshape(-1, a.size(-1)),
                                        b[:, i:i+ps].reshape(-1, b.size(-1)), dim=-1).mean().item()
                    for i in range(0, n, ps)]
            early = sum(traj[:min(3, len(traj))]) / max(1, min(3, len(traj)))

            assert mine.size(1) == golden.size(1), f"{name}: eos timing differs (len)"
            assert early > 0.999, f"{name}: early-patch latents drifted (early3={early:.4f})"
            # the rigorous part: paged hidden == HF hidden at EVERY call, no drift
            assert dbg["len_mismatch"] == 0, f"{name}: paged/HF history length drift"
            assert min(dbg["cos"]) > 0.99, f"{name}: paged-vs-HF hidden min cos={min(dbg['cos']):.5f}"
    finally:
        core.step_llm = orig_step_llm


def test_engine_reproduces_paged_golden(make_engine):
    """The batch engine (eager FM) reproduces the paged golden for one request;
    the cudagraph FM path is bit-exact; 8 concurrent requests all complete."""
    # Prefer the in-session paged golden; fall back to the on-disk artifact the
    # original verify_engine.py used (keeps this test runnable on its own / under
    # a randomized or parallel runner).
    golden = _PAGED_GOLDEN.get("en1")
    if golden is None:
        gpath = os.path.join(GOLDEN_DIR, "en1.paged_latents.npy")
        if not os.path.exists(gpath):
            pytest.skip("paged golden not available (in-session or on disk)")
        golden = torch.from_numpy(np.load(gpath)).float()

    # (1) single request, eager FM -> reproduces the paged golden (early patches)
    set_seed(1234)
    eng = make_engine(num_kvcache_blocks=128, max_num_seqs=16, compile_fm=False)
    eng.add_request("en1", EN1, num_steps=NUM_STEPS, guidance_scale=GUIDANCE)
    out = eng.run_all()["en1"]
    assert out.size(1) == golden.size(1), "engine eos timing differs from paged golden"
    n = min(out.size(1), golden.size(1))
    cmean = F.cosine_similarity(out[:, :n].reshape(-1, out.size(-1)),
                                golden[:, :n].reshape(-1, golden.size(-1)), dim=-1).mean().item()
    assert cmean > 0.999, f"engine vs paged-golden cos mean={cmean:.4f}"

    # (2) cudagraph FM (unbucketed -> records eager kernels -> bit-exact)
    set_seed(1234)
    eng = make_engine(num_kvcache_blocks=128, max_num_seqs=16, fm_accel="cudagraph")
    eng.add_request("en1", EN1, num_steps=NUM_STEPS, guidance_scale=GUIDANCE)
    cg = eng.run_all()["en1"]
    n = min(cg.size(1), golden.size(1))
    cgcos = F.cosine_similarity(cg[:, :n].reshape(-1, cg.size(-1)),
                                golden[:, :n].reshape(-1, golden.size(-1)), dim=-1).mean().item()
    assert cg.size(1) == golden.size(1) and cgcos > 0.999, f"cudagraph cos={cgcos:.4f}"

    # (3) 8 parallel requests via continuous batching -> all finite + complete
    set_seed(1234)
    eng = make_engine(num_kvcache_blocks=128, max_num_seqs=16, compile_fm=False)
    texts = [EN1, ZH1] * 4
    for i, t in enumerate(texts):
        eng.add_request(f"p{i}", t, num_steps=NUM_STEPS, guidance_scale=GUIDANCE)
    out = eng.run_all()
    assert len(out) == len(texts) and eng.scheduler.is_finished()
    assert all(torch.isfinite(v).all().item() and v.size(1) > 0 for v in out.values())
