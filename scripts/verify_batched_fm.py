#!/usr/bin/env python
"""Validate batched FM == per-request FM on ragged histories.

The engine's single-request path already matches the golden at cosine 1.0
(proving the unpadded math). The new risk is PADDING: when sequences of
different history lengths are batched, each must get the same latent it would
get alone. We build synthetic FM states of different `fm_seq_len`, fix the ODE
noise per sequence, run them (a) alone and (b) together in one ragged batch,
and compare. Matching => the padding / mask / rotary-position handling is
transparent (analogous to verify_batched.py for the LLM).

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_batched_fm.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os
import types

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--lens", default="12,28,7")
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29518")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.batched_fm import batched_flow_matching

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    dots = runtime.model
    core = dots.core
    H, patch, latent_dim = core.fm_hidden_size, core.latent_patch_size, core.latent_dim
    dev, dt = torch.device("cuda"), torch.bfloat16

    torch.manual_seed(0)
    histories = [torch.randn(1, L, H, device=dev, dtype=dt) for L in lens]
    cfg_hist = [torch.randn(1, L, H, device=dev, dtype=dt) for L in lens]
    noises = [torch.randn(1, patch, latent_dim, device=dev, dtype=dt) for _ in lens]

    def build(state_lens, hists, cfgs):
        n = len(state_lens)
        max_len = max(state_lens)
        total = max_len + patch
        inp = torch.zeros(n, total, H, device=dev, dtype=dt)
        cfg = torch.zeros(n, total, H, device=dev, dtype=dt)
        mask = torch.zeros(n, total, total, dtype=torch.bool, device=dev)
        pos = torch.zeros(n, total, dtype=torch.float32, device=dev)
        for i, L in enumerate(state_lens):
            inp[i, :L] = hists[i][0, :L]
            cfg[i, :L] = cfgs[i][0, :L]
            st = types.SimpleNamespace(fm_seq_len=L)
            dots._build_fm_attn_mask(state=st, attn_mask=mask[i : i + 1])
            dots._build_fm_pos_ids(state=st, pos_ids=pos[i : i + 1])
        g = torch.zeros(n, H, device=dev, dtype=dt)
        return inp, cfg, mask, pos, g

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dt):
        # alone
        alone = []
        for i, L in enumerate(lens):
            inp, cfg, mask, pos, g = build([L], [histories[i]], [cfg_hist[i]])
            out = batched_flow_matching(
                core, input_sequence=inp, cfg_sequence=cfg, attn_mask=mask, pos_ids=pos,
                g_cond=g, num_steps=10, guidance_scale=1.2, noise=noises[i],
            )
            alone.append(out[0].float())  # [patch, latent]
        # ragged batch
        inp, cfg, mask, pos, g = build(lens, histories, cfg_hist)
        batched = batched_flow_matching(
            core, input_sequence=inp, cfg_sequence=cfg, attn_mask=mask, pos_ids=pos,
            g_cond=g, num_steps=10, guidance_scale=1.2,
            noise=torch.cat(noises, dim=0),
        ).float()

    # Threshold catches a padding/mask LEAK (which yields cos ~0.5 or NaN for the
    # padded seq), not bf16 batch-kernel noise: batching changes SDPA tiling /
    # reduction order, so even an EQUAL-length batch (zero padding) agrees only to
    # ~5e-4 over this deep 10-step x 18-layer x CFG ODE. The golden-compared
    # single-request path has no batching and is exact (cos 1.0).
    ok = True
    for i, L in enumerate(lens):
        cos = torch.nn.functional.cosine_similarity(batched[i], alone[i], dim=-1)
        seq_ok = cos.min().item() > 0.995
        ok = ok and seq_ok
        print(f"[verify_batched_fm] seq{i} L={L} "
              f"cos(min/mean)={cos.min().item():.6f}/{cos.mean().item():.6f} "
              f"{'PASS' if seq_ok else 'FAIL'}")
    print(f"[verify_batched_fm] {'PASS' if ok else 'FAIL'} (ragged batch == alone)")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
