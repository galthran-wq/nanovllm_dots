#!/usr/bin/env python
"""Phase 1, step 4b: validate the continuous-batching DotsBatchEngine.

Two checks:
  1. SINGLE request through the engine reproduces the paged-golden latents
     (same QwenLLM + same dots FM under the same seed; batched-of-1 == single).
  2. EIGHT parallel requests all complete via continuous batching and produce
     finite, non-empty latents. (Per-seq FM noise interleaves across the batch,
     so these are independent valid samples -- not byte-equal to the golden;
     we check liveness + sanity, the per-seq LLM correctness is covered by 4a.)

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_engine.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


EN1 = "Hello, this is a reference sample generated for regression testing."
ZH1 = "这是一个用于回归测试的参考样本。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--golden", default="golden")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=1.2)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29516")
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

    def make_engine():
        return DotsBatchEngine(
            runtime, paged, model_dir=args.model,
            num_kvcache_blocks=128, block_size=256, max_num_seqs=16,
        )

    all_pass = True

    # --- Test 1: single request vs paged-golden ---
    gpath = os.path.join(args.golden, "en1.paged_latents.npy")
    if os.path.exists(gpath):
        golden = torch.from_numpy(np.load(gpath)).float()
        set_seed(args.seed)
        eng = make_engine()
        eng.add_request("en1", EN1, num_steps=args.num_steps, guidance_scale=args.guidance_scale)
        out = eng.run_all()["en1"]
        n = min(out.size(1), golden.size(1))
        same_len = out.size(1) == golden.size(1)
        if n > 0:
            cos = torch.nn.functional.cosine_similarity(
                out[:, :n].reshape(-1, out.size(-1)),
                golden[:, :n].reshape(-1, golden.size(-1)), dim=-1,
            )
            cmin, cmean = cos.min().item(), cos.mean().item()
        else:
            cmin = cmean = 0.0
        ok = same_len and cmean > 0.999
        all_pass = all_pass and ok
        print(
            f"[verify_engine] single: out={list(out.shape)} golden={list(golden.shape)} "
            f"len_match={same_len} cos(min/mean)={cmin:.4f}/{cmean:.4f} "
            f"{'PASS' if ok else 'FAIL'}"
        )
    else:
        print(f"[verify_engine] single: no paged-golden at {gpath} (run verify_e2e --save-paged-golden); skipping")

    # --- Test 2: 8 parallel requests via continuous batching ---
    set_seed(args.seed)
    eng = make_engine()
    texts = [EN1, ZH1] * 4
    for i, t in enumerate(texts):
        eng.add_request(f"p{i}", t, num_steps=args.num_steps, guidance_scale=args.guidance_scale)
    out = eng.run_all()
    n_steps = 0  # informational
    lengths = {sid: v.size(1) for sid, v in out.items()}
    finite = all(torch.isfinite(v).all().item() and v.size(1) > 0 for v in out.values())
    complete = len(out) == len(texts) and eng.scheduler.is_finished()
    ok = finite and complete
    all_pass = all_pass and ok
    print(
        f"[verify_engine] parallel-{len(texts)}: complete={complete} finite={finite} "
        f"lengths={lengths} {'PASS' if ok else 'FAIL'}"
    )

    print(f"[verify_engine] {'ALL PASS' if all_pass else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
