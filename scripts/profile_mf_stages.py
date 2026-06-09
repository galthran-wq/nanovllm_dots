#!/usr/bin/env python
"""Per-stage CUDA-time breakdown of the engine's single-stream decode, to see
where RTF goes AFTER the FM is cudagraphed. Monkeypatches the three stage entry
points (LLM append_batch, FM _batched_fm, patch_encoder _patch_to_embed) with
cuda.Event timers. Vocoder is NOT in the engine (latent-only; Phase 3 streaming),
so it is intentionally absent.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/profile_mf_stages.py \
        --model models/dots.tts-mf --num-steps 4 --fm-accel cudagraph
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

EN1 = "Hello, this is a reference sample generated for regression testing."
SEC_PER_FRAME = 1920 / 48000


class Acc:
    def __init__(self):
        self.pairs = []          # (e0, e1) recorded inline, summed once at the end

    @property
    def n(self):
        return len(self.pairs)

    @property
    def ms(self):
        # call ONLY after a torch.cuda.synchronize()
        return sum(e0.elapsed_time(e1) for e0, e1 in self.pairs)


def timed(acc, fn):
    def wrap(*a, **k):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = fn(*a, **k)
        e1.record()
        acc.pairs.append((e0, e1))   # NO sync here -- preserves cross-stage pipelining
        return out
    return wrap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--fm-accel", default="cudagraph")
    ap.add_argument("--text-repeat", type=int, default=1)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.engine import DotsBatchEngine
    from nanovllm_dots.models.dots.cudagraph_dit import (
        CudaGraphRunner, make_dit_capture_safe,
    )

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

    shared_vfp = None
    if args.fm_accel == "cudagraph":
        make_dit_capture_safe(runtime.model.core.velocity_field_predictor, torch.device("cuda"))
        shared_vfp = CudaGraphRunner(runtime.model.core.velocity_field_predictor)

    def build():
        eng = DotsBatchEngine(
            runtime, paged, model_dir=args.model, num_kvcache_blocks=256,
            block_size=256, max_num_seqs=8, fm_accel=args.fm_accel,
            fm_vfp=shared_vfp, fm_len_bucket=0,
        )
        text = " ".join([EN1] * args.text_repeat)
        eng.add_request("r0", text, num_steps=args.num_steps, guidance_scale=1.2)
        return eng

    # warmup (capture graphs)
    build().run_all()

    eng = build()
    llm, fm, pe = Acc(), Acc(), Acc()
    eng.batched_llm.append_batch = timed(llm, eng.batched_llm.append_batch)
    eng._batched_fm = timed(fm, eng._batched_fm)
    eng._patch_to_embed = timed(pe, eng._patch_to_embed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng.run_all()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    frames = sum(v.size(1) for v in out.values())
    audio_s = frames * SEC_PER_FRAME
    patches = fm.n
    total_ms = wall * 1000
    acc_ms = llm.ms + fm.ms + pe.ms
    print(f"fm_accel={args.fm_accel} num_steps={args.num_steps} "
          f"patches={patches} audio_s={audio_s:.2f} wall={wall:.3f}s RTF={wall/audio_s:.3f}")
    print(f"{'stage':<16}{'total_ms':>10}{'ms/patch':>10}{'% wall':>8}")
    for name, a in (("LLM", llm), ("FM", fm), ("patch_encoder", pe)):
        pp = a.ms / max(patches, 1)
        print(f"{name:<16}{a.ms:>10.1f}{pp:>10.2f}{100*a.ms/total_ms:>8.1f}")
    print(f"{'measured sum':<16}{acc_ms:>10.1f}{'':>10}{100*acc_ms/total_ms:>8.1f}")
    print(f"{'wall (other)':<16}{total_ms-acc_ms:>10.1f}{'':>10}{100*(total_ms-acc_ms)/total_ms:>8.1f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
