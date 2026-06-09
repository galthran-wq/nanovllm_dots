#!/usr/bin/env python
"""Validate torch.compile(patch_encoder.decode_patch) == eager, per patch.

decode_patch has static shapes (it attends over the fixed cache_capacity), so a
single compile specialization fuses its kernels. Compilation should be
numerically faithful; we confirm per patch by running BOTH the eager and the
compiled decode_patch on the SAME inputs and comparing the returned embedding.
Both write the same K/V into the cache at the same positions (idempotent), so
running them back to back is safe.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_pe_compile.py \
        --model models/dots.tts-mf
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

EN1 = "Hello, this is a reference sample generated for regression testing."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--num-steps", type=int, default=4)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29523")
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

    eng = DotsBatchEngine(
        runtime, paged, model_dir=args.model, num_kvcache_blocks=256,
        block_size=256, max_num_seqs=8, fm_accel="none",
    )
    pe = eng.core.patch_encoder
    eager_decode = pe.decode_patch
    compiled_decode = torch.compile(pe.decode_patch, dynamic=False)

    coss: list[float] = []
    orig_pte = eng._patch_to_embed

    def patched(state, patch):
        # mirror _patch_to_embed but run BOTH decode variants on identical inputs
        dots, core = eng.dots, eng.core
        dots._append_history_chunk(state, patch)
        cur = 0 if state.patch_encoder_state is None else state.patch_encoder_state.seq_len
        dots._ensure_patch_encoder_state_capacity(
            state, required_seq_len=cur + core.patch_encoder.out_ds_rate,
            device=eng.device, dtype=eng.dtype,
        )
        patch_for_llm = core.io_helper.denormalize(patch)
        positions = torch.arange(
            core.patch_encoder.out_ds_rate, device=eng.device, dtype=torch.long
        ) + state.patch_encoder_state.seq_len
        e_eag, tail_e = eager_decode(
            patch_for_llm, state.patch_encoder_state.conv_tail,
            state.patch_encoder_state.layer_caches, positions)
        e_cmp, tail_c = compiled_decode(
            patch_for_llm, state.patch_encoder_state.conv_tail,
            state.patch_encoder_state.layer_caches, positions)
        c = F.cosine_similarity(e_eag[0].float(), e_cmp[0].float(), dim=-1)
        coss.append(c.min().item())
        state.patch_encoder_state.conv_tail.copy_(tail_e)
        state.patch_encoder_state.seq_len += core.patch_encoder.out_ds_rate
        return e_eag[0]

    eng._patch_to_embed = patched
    eng.add_request("r0", EN1, num_steps=args.num_steps, guidance_scale=1.2)
    eng.run_all()
    eng._patch_to_embed = orig_pte

    cmin = min(coss) if coss else 0.0
    cmean = sum(coss) / len(coss) if coss else 0.0
    ok = cmin > 0.999
    print(f"[verify_pe_compile] patches={len(coss)} compiled-vs-eager embed "
          f"cos(min/mean)={cmin:.6f}/{cmean:.6f} {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
