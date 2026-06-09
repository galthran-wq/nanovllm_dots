#!/usr/bin/env python
"""Validate the new flash_attn_with_kvcache decode path == the varlen path.

The engine previously ran EVERY per-patch LLM step through the varlen prefill
kernel (rebuilding a block table each call). We added a decode path
(`_decode_batch`, flash_attn_with_kvcache) for one-token-per-seq batches. Both
read the same paged K/V cache; only the flash kernel differs, so per step they
must agree at the bf16 floor (cos ~1.0). We drive a real generation through the
validated varlen path and, at each decode step, also run the decode path on the
SAME inputs/cache and compare the returned hidden states.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_llm_decode.py \
        --model models/dots.tts-mf
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

EN1 = "Hello, this is a reference sample generated for regression testing."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--graph-decode", action="store_true",
                    help="route the decode path through its CUDA graph (tests graphed==varlen)")
    ap.add_argument("--num-reqs", type=int, default=1,
                    help=">1 drives ragged concurrent decodes (different lengths) through one batch")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29522")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.engine import DotsBatchEngine

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    ckpt = os.path.join(args.model, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(args.model, "llm_config.json"))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        paged = QwenLLM(cfg).eval()
    load_llm_weights(paged, ckpt)
    torch.set_default_dtype(torch.float32)

    eng = DotsBatchEngine(
        runtime, paged, model_dir=args.model, num_kvcache_blocks=256,
        block_size=256, max_num_seqs=8, fm_accel="none",
        graph_decode=args.graph_decode,
    )
    bl = eng.batched_llm
    orig_append = bl.append_batch
    coss: list[float] = []

    def patched(chunks, block_tables, cached_lens):
        sizes = [(c[0] if c.dim() == 3 else c).size(0) for c in chunks]
        if all(s == 1 for s in sizes):
            # decode path first (stores K/V), then varlen on the same cache.
            hd = bl._decode_batch(chunks, block_tables, cached_lens)
            hv = bl._varlen_batch(chunks, block_tables, cached_lens)
            for a, b in zip(hd, hv):
                c = F.cosine_similarity(a.float(), b.float(), dim=-1)
                coss.append(c.min().item())
            return hv  # drive with the validated path
        return bl._varlen_batch(chunks, block_tables, cached_lens)

    bl.append_batch = patched
    # Different text lengths -> ragged KV lengths -> exercises per-row context_lens
    # and the n>1 decode batch (and its graph) when --num-reqs > 1.
    for i in range(args.num_reqs):
        text = " ".join([EN1] * (1 + i))
        eng.add_request(f"r{i}", text, num_steps=args.num_steps, guidance_scale=1.2)
    eng.run_all()
    bl.append_batch = orig_append

    cmin = min(coss) if coss else 0.0
    cmean = sum(coss) / len(coss) if coss else 0.0
    ok = cmin > 0.999
    print(f"[verify_llm_decode] decode steps={len(coss)} decode-vs-varlen hidden "
          f"cos(min/mean)={cmin:.6f}/{cmean:.6f} {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
