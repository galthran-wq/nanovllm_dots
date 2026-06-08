#!/usr/bin/env python
"""Phase 2.6 step (e3): GraphedFlashCachedFMHead == eager FlashCachedFMHead.

The CUDA-graphed head replays the SAME flash kernels as the eager flash head, so
with matched noise the full-sequence latents must agree to ~bit (cos ~1.0). Runs
the engine twice (graphed off/on) under the same seed and compares per-request
latents.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_graphed_cache.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

EN1 = "Hello, this is a reference sample generated for regression testing."


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
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

    def run(graphed: bool):
        set_seed(args.seed)
        eng = DotsBatchEngine(
            runtime, paged, model_dir=args.model, num_kvcache_blocks=128,
            block_size=256, max_num_seqs=8, fm_accel="kvcache", kvcache_graphed=graphed,
        )
        eng.add_request("en1", EN1, num_steps=10, guidance_scale=1.2)
        return eng.run_all()["en1"]

    eager = run(False)
    graphed = run(True)
    n = min(eager.size(1), graphed.size(1))
    same_len = eager.size(1) == graphed.size(1)
    cos = F.cosine_similarity(
        eager[:, :n].reshape(-1, eager.size(-1)),
        graphed[:, :n].reshape(-1, graphed.size(-1)), dim=-1,
    )
    ok = same_len and cos.mean().item() > 0.999
    print(f"[verify_graphed_cache] eager={list(eager.shape)} graphed={list(graphed.shape)} "
          f"len_match={same_len} cos(min/mean)={cos.min().item():.5f}/{cos.mean().item():.5f} "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"[verify_graphed_cache] {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
