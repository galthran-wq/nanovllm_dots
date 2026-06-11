"""Point of impact: the Qwen2 LLM backbone (paged-KV, flash varlen, batched decode).

Asserts our ported backbone + paged cache stay consistent with (a) HuggingFace
Qwen2 and (b) a one-shot forward of our own model.
"""
from __future__ import annotations

import os

import pytest
import torch

from conftest import EN1, MF, assert_cos

pytestmark = pytest.mark.advanced
MODEL = MF  # the backbone is identical across models; use the one that's present

TEXTS = [
    "Hello there.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "In a distant land beyond the mountains, a small village prepared for the "
    "coming winter, gathering firewood and salting fish for the long cold months ahead.",
]


def _ids(model_dir: str, text: str = EN1) -> torch.Tensor:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    return tok(text, return_tensors="pt").input_ids[0].cuda()


def test_qwen_backbone_matches_huggingface(paged, model_dir):
    """Single full-sequence prefill (no KV write) == transformers Qwen2."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from safetensors import safe_open
    from nanovllm_dots.utils.context import reset_context, set_context

    ids = _ids(model_dir)
    seq_len = int(ids.shape[0])
    positions = torch.arange(seq_len, device="cuda")
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    slot = torch.full((seq_len,), -1, dtype=torch.int32, device="cuda")  # no KV write

    set_context(True, cu, cu, seq_len, seq_len, slot, None, None)
    with torch.no_grad():
        embeds = paged.embed_tokens(ids)
        mine = paged(embeds, positions).float()  # [L, H]
    reset_context()

    ckpt = os.path.join(model_dir, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
    ref = Qwen2ForCausalLM(cfg)
    sd = {}
    with safe_open(ckpt, "pt", "cpu") as f:
        for k in f.keys():
            if k.startswith("llm."):
                sd[k[len("llm."):]] = f.get_tensor(k)
    ref.load_state_dict(sd, strict=False)
    ref = ref.to(torch.bfloat16).cuda().eval()
    with torch.no_grad():
        ref_h = ref.model(input_ids=ids.unsqueeze(0), use_cache=False).last_hidden_state[0].float()
    del ref
    torch.cuda.empty_cache()

    assert_cos(mine, ref_h, 0.99, label="qwen-vs-hf")


def test_paged_incremental_append_matches_oneshot(paged, model_dir, chunk=4):
    """Chunked appends through a growing paged KV cache == one-shot forward."""
    from nanovllm_dots.models.dots.paged_llm import PagedLLMRunner
    from nanovllm_dots.utils.context import reset_context, set_context

    ids = _ids(model_dir)
    seq_len = int(ids.shape[0])
    with torch.no_grad():
        embeds = paged.embed_tokens(ids)  # [L, H]

    # reference: single full-sequence forward, no KV cache
    positions = torch.arange(seq_len, device="cuda")
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    slot = torch.full((seq_len,), -1, dtype=torch.int32, device="cuda")
    set_context(True, cu, cu, seq_len, seq_len, slot, None, None)
    with torch.no_grad():
        full = paged(embeds, positions).float()
    reset_context()

    # incremental: prefill a prefix, then append `chunk`-sized blocks
    runner = PagedLLMRunner(paged, block_size=256, max_seq_len=4096)
    prefill_len = max(1, seq_len // 2)
    pieces = [embeds[:prefill_len]]
    p = prefill_len
    while p < seq_len:
        pieces.append(embeds[p : p + chunk])
        p += chunk

    runner.reset()
    outs = []
    with torch.no_grad():
        for piece in pieces:
            outs.append(runner.append(piece).float())
    inc = torch.cat(outs, dim=0)

    assert_cos(inc, full, 0.999, label="paged-incremental-vs-oneshot")


def test_batched_driver_matches_single_seq(paged, model_dir):
    """3 ragged sequences advanced together (one varlen call) == per-seq runner.

    Catches cross-sequence contamination in packing / slot-mapping / block-table
    isolation.
    """
    from transformers import AutoTokenizer
    from nanovllm_dots.models.dots.paged_llm import PagedLLMRunner
    from nanovllm_dots.models.dots.batched_llm import BatchedPagedLLM

    tok = AutoTokenizer.from_pretrained(model_dir)
    with torch.no_grad():
        embeds = [paged.embed_tokens(tok(t, return_tensors="pt").input_ids[0].cuda()) for t in TEXTS]
    lens = [e.size(0) for e in embeds]

    # ground truth: single-seq runner per sequence (prefill half, then 1-tok decodes)
    single = PagedLLMRunner(paged, block_size=256, max_seq_len=2048)
    gt = []
    with torch.no_grad():
        for e in embeds:
            single.reset()
            half = max(1, e.size(0) // 2)
            outs = [single.append(e[:half])]
            for p in range(half, e.size(0)):
                outs.append(single.append(e[p : p + 1]))
            gt.append(torch.cat(outs, dim=0).float())

    # batched: 3 sequences together in disjoint block ranges, ragged rounds
    bps = 4
    batched = BatchedPagedLLM(paged, num_blocks=bps * len(TEXTS), block_size=256)
    block_tables = [list(range(i * bps, (i + 1) * bps)) for i in range(len(TEXTS))]
    cursor = [0] * len(TEXTS)
    collected: list[list[torch.Tensor]] = [[] for _ in TEXTS]

    with torch.no_grad():
        idx = list(range(len(TEXTS)))
        chunks = [embeds[i][: max(1, lens[i] // 2)] for i in idx]
        hids = batched.append_batch(chunks, [block_tables[i] for i in idx], [0 for _ in idx])
        for i, h in zip(idx, hids):
            collected[i].append(h.float())
            cursor[i] = max(1, lens[i] // 2)
        while any(cursor[i] < lens[i] for i in range(len(TEXTS))):
            idx = [i for i in range(len(TEXTS)) if cursor[i] < lens[i]]
            chunks = [embeds[i][cursor[i] : cursor[i] + 1] for i in idx]
            hids = batched.append_batch(
                chunks, [block_tables[i] for i in idx], [cursor[i] for i in idx]
            )
            for i, h in zip(idx, hids):
                collected[i].append(h.float())
                cursor[i] += 1

    for i in range(len(TEXTS)):
        b = torch.cat(collected[i], dim=0)
        assert b.size(0) == lens[i]
        assert_cos(b, gt[i], 0.999, label=f"batched-seq{i}")


@pytest.mark.parametrize("graph_decode", [False, True], ids=["eager", "graphed"])
def test_decode_path_matches_varlen(make_engine, graph_decode):
    """flash_attn_with_kvcache decode path == the validated varlen path, per step.

    Drives a real generation through the varlen path and, at each 1-token decode
    step, also runs the decode kernel on the same cache and compares.
    """
    import torch.nn.functional as F

    eng = make_engine(fm_accel="none", graph_decode=graph_decode)
    bl = eng.batched_llm
    orig = bl.append_batch
    coss: list[float] = []

    def patched(chunks, block_tables, cached_lens):
        sizes = [(c[0] if c.dim() == 3 else c).size(0) for c in chunks]
        if all(s == 1 for s in sizes):
            hd = bl._decode_batch(chunks, block_tables, cached_lens)
            hv = bl._varlen_batch(chunks, block_tables, cached_lens)
            for a, b in zip(hd, hv):
                coss.append(F.cosine_similarity(a.float(), b.float(), dim=-1).min().item())
            return hv
        return bl._varlen_batch(chunks, block_tables, cached_lens)

    bl.append_batch = patched
    try:
        # two requests of different lengths -> the decode rounds form a ragged
        # n>1 decode batch (exercises per-row context_lens + the decode graph)
        eng.add_request("r0", EN1, num_steps=4, guidance_scale=1.2)
        eng.add_request("r1", EN1 + " " + EN1, num_steps=4, guidance_scale=1.2)
        eng.run_all()
    finally:
        bl.append_batch = orig

    assert coss, "no decode steps were exercised"
    cmin = min(coss)
    assert cmin > 0.999, f"decode-vs-varlen cos min={cmin:.6f}"
