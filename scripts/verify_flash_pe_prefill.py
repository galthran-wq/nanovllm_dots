#!/usr/bin/env python
"""FlashPatchEncoder.prefill == reference patch_encoder.prefill, + cache continuity.

The voice-clone prompt prefill seeds the encoder cache in one pass. This validates
the NEW flash prefill numerically: (1) prompt patch embeddings match the reference
patch_encoder.prefill at the flash floor, and (2) decode_patch AFTER the prefill
matches the reference decode_patch on a prefilled state -- i.e. the seeded cache is
correct and continues correctly (the bit that the engine relies on for cloning).

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_flash_pe_prefill.py \
        --model models/dots.tts-mf
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--prompt-patches", type=int, default=22)
    ap.add_argument("--decode-patches", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29532")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.flash_patch_encoder import FlashPatchEncoder

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=64)
    torch.set_default_dtype(torch.float32)
    core = runtime.model.core
    pe = core.patch_encoder
    dev, dt = torch.device("cuda"), torch.bfloat16
    ps, ld = core.latent_patch_size, core.latent_dim
    cap = 256 * pe.out_ds_rate

    set_seed(args.seed)
    P = args.prompt_patches
    prompt_latents = torch.randn(1, P * ps, ld, device=dev, dtype=dt)

    # reference prefill (dense)
    ref_state = pe.init_decode_state(max_audio_patch_count=256, batch_size=1,
                                     device=dev, dtype=dt)
    with torch.autocast("cuda", dtype=dt):
        emb_ref, ref_state = pe.prefill(prompt_latents, ref_state)

    # flash prefill
    flash = FlashPatchEncoder(pe, max_batch=1, max_seq_len=cap, device=dev, dtype=dt)
    with torch.autocast("cuda", dtype=dt):
        emb_ours = flash.prefill(prompt_latents, 0)

    n = min(emb_ref.size(1), emb_ours.size(1))
    pre_cos = F.cosine_similarity(emb_ref[:, :n].reshape(n, -1).float(),
                                  emb_ours[:, :n].reshape(n, -1).float(), dim=-1).min().item()

    # continuity: decode_patch after the prefill, both paths, same inputs
    rows = torch.tensor([0], device=dev, dtype=torch.int32)
    coss = []
    for _ in range(args.decode_patches):
        patch = torch.randn(1, ps, ld, device=dev, dtype=dt)
        positions = torch.arange(pe.out_ds_rate, device=dev, dtype=torch.long) + ref_state.seq_len
        with torch.autocast("cuda", dtype=dt):
            r_emb, ct = pe.decode_patch(patch, ref_state.conv_tail,
                                        ref_state.layer_caches, positions)
            ref_state.conv_tail.copy_(ct); ref_state.seq_len += pe.out_ds_rate
            o_emb = flash.decode_patch(patch, rows)
        coss.append(F.cosine_similarity(r_emb.reshape(-1).float(),
                                        o_emb.reshape(-1).float(), dim=0).item())

    dmin = min(coss)
    ok = pre_cos > 0.999 and dmin > 0.999
    print(f"[verify_flash_pe_prefill] prompt_P={P} prefill-emb cos={pre_cos:.5f} "
          f"post-prefill decode cos(min over {len(coss)})={dmin:.5f} "
          f"{'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
