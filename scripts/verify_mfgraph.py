#!/usr/bin/env python
"""dots.tts-mf whole-patch MeanFlow graph == eager batched_meanflow, PER PATCH.

The whole-patch graph captures the entire nfe-step solver (coordinate_proj +
clones + DiT + z-update) into one CUDA graph. It records the eager kernels, so
per patch it must match `batched_meanflow` at the bf16 floor. We drive ONE real
trajectory (eager) and at each patch run the graphed solver on the SAME state +
SAME noise and compare the produced latents. (A full-sequence compare is wrong:
the tiny per-patch bf16 diff amplifies autoregressively -- same lesson as the FM
cudagraph / kvcache paths.)

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_mfgraph.py \
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

EN1 = "Hello, this is a reference sample generated for regression testing."


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29524")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.engine import DotsBatchEngine
    from nanovllm_dots.models.dots.batched_fm import batched_meanflow
    from nanovllm_dots.models.dots.graphed_meanflow import GraphedMeanflow
    from nanovllm_dots.models.dots.cudagraph_dit import make_dit_capture_safe

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
            make_dit_capture_safe(self.core.velocity_field_predictor, self.device)
            self._g = GraphedMeanflow(self.core, num_steps=args.num_steps,
                                      dit=self.core.velocity_field_predictor)
            self.coss: list[float] = []

        def _batched_fm(self, payloads):
            assert len(payloads) == 1
            inp, _, mask, pos, gcond, p0, _ = self._pack_fm_inputs(payloads)
            ld = self.core.latent_dim
            noise = torch.randn(1, self.core.latent_patch_size, ld,
                                device=self.device, dtype=self.dtype)
            eager = batched_meanflow(
                self.core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                g_cond=gcond, num_steps=p0.num_steps, noise=noise.clone())
            graphed = self._g(input_sequence=inp, attn_mask=mask, pos_ids=pos,
                              g_cond=gcond, num_steps=p0.num_steps, noise=noise.clone())
            c = F.cosine_similarity(eager.reshape(-1, ld).float(),
                                    graphed.reshape(-1, ld).float(), dim=-1)
            self.coss.append(c.min().item())
            return eager  # eager drives the trajectory

    set_seed(args.seed)
    eng = DualEngine(
        runtime, paged, model_dir=args.model, num_kvcache_blocks=256,
        block_size=256, max_num_seqs=8, fm_accel="none",
    )
    eng.add_request("r0", EN1, num_steps=args.num_steps, guidance_scale=1.2)
    eng.run_all()

    coss = eng.coss
    cmin, cmean = min(coss), sum(coss) / len(coss)
    ok = cmin > 0.999
    print(f"[verify_mfgraph] patches={len(coss)} whole-patch-graph vs eager latent "
          f"cos(min/mean)={cmin:.5f}/{cmean:.5f} {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
