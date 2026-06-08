#!/usr/bin/env python
"""Phase 2 benchmark: engine throughput vs concurrency.

Measures the DotsBatchEngine's single-stream RTF and aggregate throughput
(audio-seconds produced per wall-second) at several concurrency levels. This is
the baseline the FM optimizations must beat: with serial per-request FM, adding
concurrency batches only the LLM (16%), so aggregate throughput should barely
rise; once FM is batched it should scale with concurrency.

Latent->audio: 1 latent frame = 1920 samples @ 48 kHz = 0.04 s (from golden).

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/bench_engine.py \
        --model models/dots.tts-soar --concurrency 1,4,8
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

SAMPLES_PER_FRAME = 1920
SR = 48000
SEC_PER_FRAME = SAMPLES_PER_FRAME / SR  # 0.04

EN1 = "Hello, this is a reference sample generated for regression testing."
ZH1 = "这是一个用于回归测试的参考样本。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=1.2)
    ap.add_argument("--no-compile", action="store_true", help="disable FM torch.compile")
    args = ap.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
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

    # Share ONE compiled FM wrapper across engine instances so the CUDA-graph /
    # compile cache persists (each run_batch builds a fresh engine).
    compile_fm = not args.no_compile
    shared_vfp = (
        torch.compile(runtime.model.core.velocity_field_predictor,
                      mode="default", dynamic=False)
        if compile_fm else None
    )

    def run_batch(n: int):
        eng = DotsBatchEngine(
            runtime, paged, model_dir=args.model,
            num_kvcache_blocks=256, block_size=256, max_num_seqs=max(n, 8),
            compile_fm=compile_fm, fm_vfp=shared_vfp,
        )
        texts = ([EN1, ZH1] * ((n + 1) // 2))[:n]
        for i, t in enumerate(texts):
            eng.add_request(f"r{i}", t, num_steps=args.num_steps, guidance_scale=args.guidance_scale)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = eng.run_all()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        frames = sum(v.size(1) for v in out.values())
        return wall, frames

    print(f"compile_fm={compile_fm}")
    print(f"{'conc':>5} {'wall_s':>8} {'frames':>7} {'audio_s':>8} "
          f"{'thrpt(x_rt)':>11} {'RTF/req':>8}")
    for n in levels:
        run_batch(n)          # warmup: compile/capture this level's shapes
        wall, frames = run_batch(n)
        audio_s = frames * SEC_PER_FRAME
        throughput = audio_s / wall          # audio-seconds per wall-second (xRT)
        rtf_per_req = wall / (audio_s / n)    # mean per-request RTF
        print(f"{n:>5} {wall:>8.3f} {frames:>7} {audio_s:>8.2f} "
              f"{throughput:>11.3f} {rtf_per_req:>8.3f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
