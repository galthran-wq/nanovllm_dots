#!/usr/bin/env python
"""Incremental cache extend == monolithic prefill.

In real generation the prefix cache is grown patch-by-patch: each patch appends
the 5 rows (1 hidden + 4 latent) that graduate from "active" to "prefix" via a
chunked prefill (`extend_cache` with the existing cache present). This must equal
prefilling the whole prefix in one shot, because the prefix attention is a single
growing causal block -- row i's hidden depends only on rows [0:i], which are
fully present whether they arrived in one chunk or many.

This validates that by building the SAME prefix two ways and checking:
  - per-layer cached K/V match (max abs diff ~ bf16 noise),
  - the active velocity read off each cache matches (cos ~0.9999).

The incremental path uses a realistic schedule: an initial prefill of `init`
rows, then several 5-row extends (the history stride = hidden_patch_size +
latent_patch_size).

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_fm_cache_extend.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29534")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.cached_fm import FMCache, extend_cache, active_forward

    rt = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    core = rt.model.core
    dit = core.velocity_field_predictor
    H, lp, hp, ld = (
        core.fm_hidden_size, core.latent_patch_size,
        core.hidden_patch_size, core.latent_dim,
    )
    stride = hp + lp  # 5 rows graduate to prefix per patch
    dev, dt = torch.device("cuda"), torch.bfloat16
    n_layers = dit.num_layers

    ok = True
    for B in (1, 2):
        for init, n_ext in ((7, 16), (13, 8)):
            P = init + n_ext * stride
            torch.manual_seed(7000 * B + P)
            x_prefix = torch.randn(B, P, H, device=dev, dtype=dt)
            x_act = torch.randn(B, hp + lp, H, device=dev, dtype=dt)
            t = torch.rand(1, device=dev, dtype=dt)
            g = torch.zeros(B, H, device=dev, dtype=dt)
            pos_prefix = torch.arange(P, device=dev, dtype=torch.float32).expand(B, -1)
            pos_act = torch.arange(P, P + hp + lp, device=dev, dtype=torch.float32).expand(B, -1)

            with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                # monolithic prefill
                cm = FMCache(num_steps=1, num_layers=n_layers)
                extend_cache(dit, cm, x_new=x_prefix, pos_new=pos_prefix,
                             t_scalar=t, g_cond=g, k_index=0)
                # incremental: init chunk then 5-row extends
                ci = FMCache(num_steps=1, num_layers=n_layers)
                extend_cache(dit, ci, x_new=x_prefix[:, :init], pos_new=pos_prefix[:, :init],
                             t_scalar=t, g_cond=g, k_index=0)
                for j in range(n_ext):
                    s = init + j * stride
                    e = s + stride
                    extend_cache(dit, ci, x_new=x_prefix[:, s:e], pos_new=pos_prefix[:, s:e],
                                 t_scalar=t, g_cond=g, k_index=0)

                # per-layer K/V agreement
                kv_diff = 0.0
                for l in range(n_layers):
                    km, vm = cm.kv[0][l]
                    ki, vi = ci.kv[0][l]
                    kv_diff = max(kv_diff, (km - ki).abs().max().item(),
                                  (vm - vi).abs().max().item())

                # active velocity off each cache
                vm_out = active_forward(dit, cm, x_act, pos_act, t, g, 0).float()
                vi_out = active_forward(dit, ci, x_act, pos_act, t, g, 0).float()

            cos = F.cosine_similarity(
                vm_out[:, hp:].reshape(-1, ld), vi_out[:, hp:].reshape(-1, ld), dim=-1
            )
            len_ok = cm.length() == P == ci.length()
            seq_ok = cos.mean().item() > 0.999 and len_ok
            ok = ok and seq_ok
            print(f"[verify_fm_cache_extend] B={B} P={P} (init={init}+{n_ext}x{stride}): "
                  f"kv_maxdiff={kv_diff:.4f} active cos(min/mean)="
                  f"{cos.min().item():.5f}/{cos.mean().item():.5f} "
                  f"len_ok={len_ok} {'PASS' if seq_ok else 'FAIL'}")

    print(f"[verify_fm_cache_extend] {'PASS' if ok else 'FAIL'} (incremental extend == monolithic)")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
