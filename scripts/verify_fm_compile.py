#!/usr/bin/env python
"""Phase 2.3: validate that torch.compile of the FM DiT is numerically faithful.

The compiled FM produces a valid-but-not-bit-identical sample vs the eager
golden: inductor reorders bf16 ops (~3% per-call reldiff), and the autoregressive
FM amplifies that to ~cos 0.93 over a full sequence (eos timing unchanged) --
exactly the Phase-1 paged-LLM pattern. So we don't compare the *sequence* to the
eager golden; we check the thing that actually proves faithfulness: the compiled
DiT forward matches the eager DiT forward on identical inputs, per call.

cos ~0.999 per call (bf16 fusion noise, same order as batch-kernel noise) =>
faithful speedup. A miscompile would show low per-call cos.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_fm_compile.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29522")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    core = runtime.model.core
    H, ld = core.fm_hidden_size, core.latent_dim
    vfp_c = torch.compile(core.velocity_field_predictor, mode="default", dynamic=True)

    torch.manual_seed(7)
    ok = True
    for L in (12, 40, 90):
        x = torch.randn(2, L, H, device="cuda", dtype=torch.bfloat16)
        t = torch.rand(2, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(2, L, L, dtype=torch.bool, device="cuda").tril()
        pos = torch.arange(L, device="cuda").float().unsqueeze(0).expand(2, -1).contiguous()
        g = torch.zeros(2, H, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            oe = core.velocity_field_predictor(
                x=x, timesteps=t, attn_mask=mask, pos_ids=pos, g_cond=g
            ).float()
            oc = vfp_c(x=x, timesteps=t, attn_mask=mask, pos_ids=pos, g_cond=g).float()
        cos = torch.nn.functional.cosine_similarity(oe.reshape(-1, ld), oc.reshape(-1, ld), dim=-1)
        reld = ((oe - oc).norm() / oe.norm()).item()
        seq_ok = cos.mean().item() > 0.99
        ok = ok and seq_ok
        print(f"[verify_fm_compile] L={L}: cos(min/mean)="
              f"{cos.min().item():.5f}/{cos.mean().item():.5f} maxreldiff={reld:.4e} "
              f"{'PASS' if seq_ok else 'FAIL'}")

    print(f"[verify_fm_compile] {'PASS' if ok else 'FAIL'} (compiled DiT faithful per-call)")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
