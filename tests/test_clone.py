"""Point of impact: voice cloning (speaker g_cond + in-context prompt prefill).

g_cond: the engine's speaker conditioning must be bit-faithful to the reference
`_prepare_prompt_conditioning`, distinguish speakers, and change the audio.
prefill: the full clone-prefill path (flash_pe.prefill + prompt spans in the LLM
prefill + prompt (hidden,latent) FM seeding) must run end-to-end.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from conftest import GOLDEN_MF_DIR, MF, set_seed

pytestmark = pytest.mark.advanced
MODEL = MF

TEXT = "Voice cloning conditions the synthesizer on a reference speaker."
REF_A = os.path.join(GOLDEN_MF_DIR, "en1.wav")
REF_B = os.path.join(GOLDEN_MF_DIR, "zh1.wav")


def _require_refs(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"reference audio not found: {p}")


def test_gcond_faithful_distinct_and_audible(make_engine, reference):
    """(1) engine g_cond == reference g_cond, (2) distinct speakers -> distinct
    g_cond, (3) cloned audio differs from un-cloned."""
    _require_refs(REF_A, REF_B)
    prep = reference.model._prepare_prompt_conditioning

    # (1) faithfulness
    pa = reference._load_prompt_audio(REF_A).to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gA = prep(pa, use_prompt_prefill=False, speaker_scale=1.5).g_cond
    eng0 = make_engine(fm_accel="cudagraph", flash_pe=True)
    eng0.add_request("a", TEXT, num_steps=4, prompt_audio_path=REF_A, speaker_scale=1.5)
    eng_g = eng0.scheduler._id_to_seq["a"].custom_payload.g_cond
    faith = F.cosine_similarity(gA.reshape(-1).float(), eng_g.reshape(-1).float(), dim=0).item()
    assert faith > 0.999, f"engine g_cond vs reference cos={faith:.5f}"

    # (2) distinct speakers
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gB = prep(reference._load_prompt_audio(REF_B).to("cuda"),
                  use_prompt_prefill=False, speaker_scale=1.5).g_cond
    spk_cos = F.cosine_similarity(gA.reshape(-1).float(), gB.reshape(-1).float(), dim=0).item()
    assert spk_cos < 0.99, f"speakers A and B not distinct (cos={spk_cos:.4f})"

    # (3) cloned audio differs from null (same FM noise -> only conditioning differs)
    wavs = {}
    for tag, ref in [("null", None), ("cloneA", REF_A)]:
        set_seed(1234)
        eng = make_engine(fm_accel="cudagraph", flash_pe=True)
        eng.add_request("r", TEXT, num_steps=4, prompt_audio_path=ref, speaker_scale=1.5)
        wavs[tag] = eng.vocode(eng.run_all()["r"])
    n = min(wavs["null"].numel(), wavs["cloneA"].numel())
    null_vs_A = F.cosine_similarity(wavs["null"][:n], wavs["cloneA"][:n], dim=0).item()
    assert null_vs_A < 0.999, f"cloning did not change the audio (cos={null_vs_A:.4f})"


def test_clone_prefill_runs(make_engine):
    """Full in-context clone-prefill path runs end-to-end and seeds prompt patches."""
    _require_refs(REF_A)
    prompt_text = "Hello, this is a reference sample generated for regression testing."

    set_seed(1234)
    eng = make_engine(fm_accel="cudagraph", flash_pe=True, num_kvcache_blocks=512)
    eng.add_request("c", "The quick brown fox jumps over the lazy dog near the river bank.",
                    num_steps=4, prompt_audio_path=REF_A, prompt_text=prompt_text, clone_prefill=True)
    payload = eng.scheduler._id_to_seq["c"].custom_payload
    prompt_P = int(payload.prompt_patches.size(1))
    wav = eng.vocode(eng.run_all()["c"])

    assert prompt_P > 0, "prompt patches were not seeded"
    assert wav.numel() > 0, "clone-prefill produced no audio"
