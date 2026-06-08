#!/usr/bin/env python
"""Phase 1, step 4a: validate the batched (continuous-batching) LLM driver.

`BatchedPagedLLM.append_batch` advances many sequences in one varlen flash-attn
call. Batching must be transparent: each sequence's per-token hidden states must
equal what the validated single-sequence `PagedLLMRunner` produces for the same
tokens. We run 3 sequences of *different* lengths together in ragged lockstep
rounds (prefill half, then 1-token decodes, sequences finishing at different
rounds) and compare to the single-seq ground truth. High cosine => no cross-
sequence contamination in the packing / slot-mapping / block-table isolation.

Run:
    PYTHONPATH=. .venv/bin/python scripts/verify_batched.py --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


TEXTS = [
    "Hello there.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "In a distant land beyond the mountains, a small village prepared for the "
    "coming winter, gathering firewood and salting fish for the long cold months ahead.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29515")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import AutoTokenizer, Qwen2Config

    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.paged_llm import PagedLLMRunner
    from nanovllm_dots.models.dots.batched_llm import BatchedPagedLLM

    model_dir = args.model
    ckpt = os.path.join(model_dir, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
    tok = AutoTokenizer.from_pretrained(model_dir)

    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        llm = QwenLLM(cfg).eval()
    load_llm_weights(llm, ckpt)
    torch.set_default_dtype(torch.float32)

    with torch.no_grad():
        embeds = [llm.embed_tokens(tok(t, return_tensors="pt").input_ids[0].cuda()) for t in TEXTS]
    lens = [e.size(0) for e in embeds]
    print(f"[verify_batched] seq lens={lens}")

    # --- ground truth: single-seq PagedLLMRunner per sequence ---
    single = PagedLLMRunner(llm, block_size=256, max_seq_len=2048)
    gt = []
    with torch.no_grad():
        for e in embeds:
            single.reset()
            half = max(1, e.size(0) // 2)
            outs = [single.append(e[:half])]
            for p in range(half, e.size(0)):
                outs.append(single.append(e[p : p + 1]))
            gt.append(torch.cat(outs, dim=0).float())  # [Li, H]

    # --- batched: 3 sequences together, disjoint block ranges, ragged rounds ---
    blocks_per_seq = 4  # 256*4 = 1024 tokens capacity per seq, ample
    batched = BatchedPagedLLM(
        llm, num_blocks=blocks_per_seq * len(TEXTS), block_size=256
    )
    block_tables = [
        list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq)) for i in range(len(TEXTS))
    ]
    cursor = [0] * len(TEXTS)
    collected: list[list[torch.Tensor]] = [[] for _ in TEXTS]

    with torch.no_grad():
        # round 0: prefill the first half of every sequence
        idx = list(range(len(TEXTS)))
        chunks = [embeds[i][: max(1, lens[i] // 2)] for i in idx]
        hids = batched.append_batch(chunks, [block_tables[i] for i in idx], [0 for _ in idx])
        for i, h in zip(idx, hids):
            collected[i].append(h.float())
            cursor[i] = max(1, lens[i] // 2)

        # decode rounds: append one token to each still-unfinished sequence
        while any(cursor[i] < lens[i] for i in range(len(TEXTS))):
            idx = [i for i in range(len(TEXTS)) if cursor[i] < lens[i]]
            chunks = [embeds[i][cursor[i] : cursor[i] + 1] for i in idx]
            hids = batched.append_batch(
                chunks, [block_tables[i] for i in idx], [cursor[i] for i in idx]
            )
            for i, h in zip(idx, hids):
                collected[i].append(h.float())
                cursor[i] += 1

    ok = True
    for i in range(len(TEXTS)):
        b = torch.cat(collected[i], dim=0)  # [Li, H]
        cos = torch.nn.functional.cosine_similarity(b, gt[i], dim=-1)
        seq_ok = cos.min().item() > 0.999
        ok = ok and seq_ok and b.size(0) == lens[i]
        print(
            f"[verify_batched] seq{i} len={b.size(0)} "
            f"cos(min/mean)={cos.min().item():.6f}/{cos.mean().item():.6f} "
            f"{'PASS' if seq_ok else 'FAIL'}"
        )

    print(f"[verify_batched] {'PASS' if ok else 'FAIL'} (batched == single-seq)")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
