#!/usr/bin/env python
"""Phase 2.6 step (e3): GraphedFlashCachedFMHead == eager FlashCachedFMHead, PER
PATCH.

The graphed head batches the extend over all ODE timesteps, so its flash kernels
(batch 2*nfe) differ from the eager head's (batch 2) -- an equally-valid but not
bit-identical bf16 result. Per patch the latents match at the bf16 floor (cos
~0.9999); over a full autoregressive sequence that tiny diff amplifies (and can
shift eos), so a full-sequence comparison is the WRONG metric (same lesson as the
compile path). We drive ONE real trajectory (eager flash) and, at each patch, run
the graphed head on the SAME state + SAME noise and compare per-patch latents.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_graphed_cache.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

EN1 = "Hello, this is a reference sample generated for regression testing."


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.engine import DotsBatchEngine
    from nanovllm_dots.models.dots.flash_cached_fm import (
        FlashCachedFMHead, GraphedFlashCachedFMHead,
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

    class DualEngine(DotsBatchEngine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.eager_head = None
            self.graphed_head = None
            self.coss: list[float] = []

        def _batched_fm(self, payloads):
            assert len(payloads) == 1
            p = payloads[0]; st = p.gen_state; core = self.core
            lp, ld = core.latent_patch_size, core.latent_dim
            stride = core.hidden_patch_size + lp
            dev, dt = self.device, self.dtype
            if self.eager_head is None:
                mk = lambda C: C(core, num_steps=p.num_steps, guidance_scale=p.guidance_scale,
                                 max_patches=st.fm_capacity // stride + 1, dit=self._vfp)
                self.eager_head = mk(FlashCachedFMHead)
                self.graphed_head = mk(GraphedFlashCachedFMHead)
            gcond = (p.g_cond.to(dev, dt).reshape(1, -1) if p.g_cond is not None
                     else st.fm_null_g_cond.to(dev, dt))
            noise = torch.randn((1, lp, ld), device=dev, dtype=dt)
            le = self.eager_head.decode_patch(st, noise.clone(), gcond)
            lg = self.graphed_head.decode_patch(st, noise.clone(), gcond)
            self.coss.append(F.cosine_similarity(
                le.reshape(-1, ld).float(), lg.reshape(-1, ld).float(), dim=-1).mean().item())
            return le  # eager drives the trajectory

    set_seed(args.seed)
    eng = DualEngine(
        runtime, paged, model_dir=args.model, num_kvcache_blocks=128,
        block_size=256, max_num_seqs=8, fm_accel="kvcache",
    )
    eng.add_request("en1", EN1, num_steps=10, guidance_scale=1.2)
    eng.run_all()

    coss = eng.coss
    cmin, cmean = min(coss), sum(coss) / len(coss)
    ok = cmin > 0.999
    print(f"[verify_graphed_cache] patches={len(coss)} per-patch graphed-vs-eager latent cos "
          f"min/mean={cmin:.5f}/{cmean:.5f} {'PASS' if ok else 'FAIL'}")
    print(f"[verify_graphed_cache] {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
