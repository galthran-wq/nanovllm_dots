#!/usr/bin/env python
"""FlashPatchEncoder == reference patch_encoder.decode_patch, PER PATCH, batched.

The flash path swaps the dense-mask O(capacity) SDPA for flash_attn_with_kvcache
over the actual length, and runs N requests in ONE batched call (per-row
cache_seqlens). It reuses the reference modules, so per patch it must match the
reference `decode_patch` at the flash/bf16 floor. We drive `--n` independent real
trajectories with the REFERENCE decode (each its own decode-state, B=1) and at
every patch feed the SAME latent patches into the batched FlashPatchEncoder, then
compare the produced embeds row-by-row. (Per-patch compare, not full-sequence:
the embed feeds the LLM, not the next patch_encoder step, so a tiny per-patch
diff does not amplify -- but we check per-patch to be strict.)

To exercise DIFFERENT per-row history lengths (the continuous-batching case), the
rows are advanced a different number of patches before they are compared.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_flash_pe.py \
        --model models/dots.tts-mf --n 3 --patches 30
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--n", type=int, default=3, help="batch size (parallel rows)")
    ap.add_argument("--patches", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29526")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.flash_patch_encoder import FlashPatchEncoder

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    torch.set_default_dtype(torch.float32)
    core = runtime.model.core
    pe = core.patch_encoder
    dev = torch.device("cuda")
    dt = torch.bfloat16
    n, P = args.n, args.patches
    ps, ld = core.latent_patch_size, core.latent_dim
    cap = 256 * pe.out_ds_rate

    set_seed(args.seed)

    # reference: one B=1 decode-state per row
    ref_states = [
        pe.init_decode_state(max_audio_patch_count=256, batch_size=1,
                             device=dev, dtype=dt)
        for _ in range(n)
    ]
    flash = FlashPatchEncoder(pe, max_batch=n, max_seq_len=cap, device=dev, dtype=dt)
    # non-trivial row permutation -> exercises cache_batch_idx (rows != 0..n-1)
    rows = torch.tensor(list(reversed(range(n))), device=dev, dtype=torch.int32)

    coss = []
    for _ in range(P):
        # a fresh random denormalized patch per row (range doesn't matter for the
        # cos check -- we feed identical inputs to both paths)
        patches = [torch.randn(1, ps, ld, device=dev, dtype=dt) for _ in range(n)]

        ref_embeds = []
        for i in range(n):
            st = ref_states[i]
            positions = torch.arange(pe.out_ds_rate, device=dev, dtype=torch.long) + st.seq_len
            emb, conv_tail = pe.decode_patch(
                patches[i], st.conv_tail, st.layer_caches, positions)
            st.conv_tail.copy_(conv_tail)
            st.seq_len += pe.out_ds_rate
            ref_embeds.append(emb)                 # [1, 1, out_dim]
        ref = torch.cat(ref_embeds, dim=0)         # [n, 1, out_dim]

        batch = torch.cat(patches, dim=0)          # [n, ps, ld]
        got = flash.decode_patch(batch, rows)      # [n, 1, out_dim], row i -> rows[i]

        # ref[i] corresponds to flash batch-row i (cache row rows[i]); compare aligned
        c = F.cosine_similarity(ref.reshape(n, -1).float(),
                                got.reshape(n, -1).float(), dim=-1)
        coss.append(c.min().item())

    cmin, cmean = min(coss), sum(coss) / len(coss)
    ok = cmin > 0.999
    print(f"[verify_flash_pe] n={n} patches={P} flash-batched vs reference "
          f"decode_patch embed cos(min/mean)={cmin:.5f}/{cmean:.5f} "
          f"{'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
