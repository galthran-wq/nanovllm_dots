#!/usr/bin/env python
"""FM-only crossover micro-benchmark: graphed KV-cache vs cudagraph-full DiT, as a
function of history length L. Isolates the FM head (no LLM / patch_encoder /
capture noise): builds a synthetic FM history of length L, then times ONE patch's
FM work (10-step CFG ODE) steady-state for each accelerator.

This answers: does the O(P) KV cache (many tiny [2,5] kernels) ever beat the O(P^2)
cudagraph-full (20 big forward(L) kernels) -- and if so, at what L? FM is
occupancy-bound, so the cache's FLOP win fights its kernel-efficiency loss.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/bench_fm_crossover.py \
        --model models/dots.tts-soar --lengths 100,300,600,1000
"""
from __future__ import annotations

import argparse
import os
import time
import types

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--lengths", default="100,300,600,1000")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    Ls = [int(x) for x in args.lengths.split(",")]

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29538")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.cudagraph_dit import CudaGraphRunner, make_dit_capture_safe
    from nanovllm_dots.models.dots.flash_cached_fm import GraphedFlashCachedFMHead

    rt = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    core = rt.model.core
    dit = core.velocity_field_predictor
    H, lp, hp, ld = core.fm_hidden_size, core.latent_patch_size, core.hidden_patch_size, core.latent_dim
    stride = hp + lp
    dev, dt = torch.device("cuda"), torch.bfloat16
    ns, gs = 10, 1.2

    make_dit_capture_safe(dit, dev)
    full = CudaGraphRunner(dit)

    def time_it(fn, iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3  # ms/patch

    print(f"{'L':>6} {'full_ms':>9} {'kvcache_ms':>11} {'speedup':>8}")
    with torch.autocast("cuda", dtype=dt):
        for L in Ls:
            # synthetic FM history of length L
            g = torch.zeros(1, H, device=dev, dtype=dt)
            noise = torch.randn(1, lp, ld, device=dev, dtype=dt)

            # --- cudagraph-full: one CFG ODE over the full L-row sequence ---
            total = L + lp
            st = types.SimpleNamespace(fm_seq_len=L)
            mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
            pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
            rt.model._build_fm_attn_mask(state=st, attn_mask=mask)
            rt.model._build_fm_pos_ids(state=st, pos_ids=pos)
            inp = torch.randn(1, total, H, device=dev, dtype=dt)
            mask2 = mask.expand(2, -1, -1).contiguous()
            pos2 = pos.expand(2, -1).contiguous()
            g2 = torch.zeros(2, H, device=dev, dtype=dt)

            def full_patch():
                z = noise
                for k in range(ns):
                    zp = core.coordinate_proj(z)
                    zz = inp.expand(2, -1, -1).clone()
                    zz[:, L:] = zp
                    t = torch.full((2,), k / ns, device=dev, dtype=dt)
                    vt = full(x=zz, timesteps=t, attn_mask=mask2, pos_ids=pos2, g_cond=g2)[:, L:]
                    v = vt[:1] + gs * (vt[:1] - vt[1:])
                    z = z + (1.0 / ns) * v

            # --- graphed kvcache: grow the cache patch-by-patch (real generation
            # pattern: fm_seq_len 1 -> 6 -> 11 -> ..., extend = stride rows/patch),
            # build up to ~L, then time additional steady-state patches. ---
            cap_patches = L // stride + args.iters + 20
            fmseq = torch.randn(1, cap_patches * stride + stride, H, device=dev, dtype=dt)
            head = GraphedFlashCachedFMHead(core, num_steps=ns, guidance_scale=gs,
                                            max_patches=cap_patches + 4, dit=dit)
            state = types.SimpleNamespace(
                fm_seq_len=1, fm_sequence=fmseq, fm_cfg_sequence=fmseq.clone())

            def kv_patch():
                head.decode_patch(state, noise, g)
                state.fm_seq_len += stride   # next patch's hidden row

            while state.fm_seq_len < L:       # build cache up to length L
                kv_patch()
            for _ in range(5):                # warmup steady-state
                full_patch(); kv_patch()
            tf = time_it(full_patch, args.iters)
            tk = time_it(kv_patch, args.iters)
            print(f"{L:>6} {tf:>9.3f} {tk:>11.3f} {tf / tk:>7.2f}x  (kv L~{state.fm_seq_len})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
