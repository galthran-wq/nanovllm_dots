#!/usr/bin/env python
"""The FM KV-cache primitives reproduce the full DiT forward.

Builds the same single-patch FM input as `verify_fm_split.py` (prefix rows
[0:L-1] + 5 active rows = last-hidden + latent patch, no padding), then computes
the latent velocity two ways:

  full:   dit(x, ...)                         -- the reference forward
  cached: extend_cache(empty, prefix rows)    -- prefill the prefix K/V
          active_forward(active rows)         -- read cache, no recompute of prefix

They must match (cos ~0.9999, the bf16 attention-reorder noise from the split
foundation). This validates BOTH the prefill path (extend_cache from empty ==
causal prefix) and the cache-read path (active_forward). Tested at B=1 and B=2
(CFG-style batch) so the primitive is batch-correct from the start.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_fm_cache.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.cached_fm import (
        FMCache, extend_cache, active_forward,
    )

    rt = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    core = rt.model.core
    dit = core.velocity_field_predictor
    H, lp, hp, ld = (
        core.fm_hidden_size, core.latent_patch_size,
        core.hidden_patch_size, core.latent_dim,
    )
    dev, dt = torch.device("cuda"), torch.bfloat16
    n_layers = dit.num_layers

    ok = True
    for B in (1, 2):
        for L in (8, 30, 90):
            total = L + lp                      # no padding: latent_start == L
            torch.manual_seed(1000 * B + L)
            x = torch.randn(B, total, H, device=dev, dtype=dt)
            t = torch.rand(B, device=dev, dtype=dt)
            g = torch.zeros(B, H, device=dev, dtype=dt)

            # reference mask/pos via the engine builders (single-stream shapes)
            st = types.SimpleNamespace(fm_seq_len=L)
            mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
            pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
            rt.model._build_fm_attn_mask(state=st, attn_mask=mask)
            rt.model._build_fm_pos_ids(state=st, pos_ids=pos)
            mask_b = mask.expand(B, -1, -1)
            pos_b = pos.expand(B, -1)

            with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                vf = dit(x=x, timesteps=t, attn_mask=mask_b, pos_ids=pos_b, g_cond=g).float()

                # cached path: prefill prefix [0:L-1], then active [L-1:]
                cache = FMCache(num_steps=1, num_layers=n_layers)
                extend_cache(
                    dit, cache,
                    x_new=x[:, : L - 1], pos_new=pos_b[:, : L - 1],
                    t_scalar=t, g_cond=g, k_index=0,
                )
                vs = active_forward(
                    dit, cache,
                    x_active=x[:, L - 1 :], pos_active=pos_b[:, L - 1 :],
                    t_scalar=t, g_cond=g, k_index=0,
                ).float()

            # compare the latent rows (final lp rows of active == final lp of full)
            a = vf[:, L:].reshape(-1, ld)
            b = vs[:, hp:].reshape(-1, ld)
            cos = F.cosine_similarity(a, b, dim=-1)
            seq_ok = cos.mean().item() > 0.999 and cache.length() == L - 1
            ok = ok and seq_ok
            print(f"[verify_fm_cache] B={B} L={L}: latent cos(min/mean)="
                  f"{cos.min().item():.5f}/{cos.mean().item():.5f} "
                  f"prefixP={cache.length()} {'PASS' if seq_ok else 'FAIL'}")

    print(f"[verify_fm_cache] {'PASS' if ok else 'FAIL'} (cache primitives == full DiT)")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
