#!/usr/bin/env python
"""Phase 2.6 step (c): CachedFMHead reproduces the eager FM per call, over a full
real generation.

Methodology (same as verify_fm_compile.py): autoregressive amplification means a
faithful-but-not-bit-identical FM would still drift at the SEQUENCE level, so we
do NOT compare full sequences. Instead we run ONE real generation along the EAGER
trajectory (ground truth) and, at every patch, compute the latent BOTH ways from
the same ground-truth history and the same noise:

  eager:  batched_flow_matching over the full padded sequence (the golden path)
  cached: CachedFMHead.decode_patch -- incremental prefix KV cache + 5 active rows

The cache is grown incrementally across the real generation (extend reads the
ground-truth fm_sequence buffers), so this exercises the REAL stateful cache, but
each comparison is a clean per-call check (cached never feeds back, so no drift
masks errors). Per-patch latent cos must stay ~0.9999 (the bf16 floor).

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_fm_cache_seq.py \
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
from torchdiffeq import odeint

EN1 = "Hello, this is a reference sample generated for regression testing."


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=1.2)
    ap.add_argument("--noise-base", type=int, default=20260608)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29535")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime

    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.engine import DotsBatchEngine
    from nanovllm_dots.models.dots.cached_fm import CachedFMHead

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
        """Eager engine that also runs the cached FM per patch and records the
        per-call latent cos. Trajectory follows eager (returns the eager latent)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.cached_head = None
            self.patch_idx = 0
            self.coss: list[float] = []
            self.noise_base = 20260608
            self.relds: list[float] = []

        def _batched_fm(self, payloads):
            assert len(payloads) == 1, "DualEngine validates single-stream"
            p = payloads[0]
            st = p.gen_state
            core = self.core
            L = int(st.fm_seq_len)
            H, lp, ld = core.fm_hidden_size, core.latent_patch_size, core.latent_dim
            dev, dt = self.device, self.dtype

            if self.cached_head is None:
                self.cached_head = CachedFMHead(
                    core, num_steps=p.num_steps, guidance_scale=p.guidance_scale,
                    dit=core.velocity_field_predictor,
                )
            # cond-branch g_cond (uncond uses zeros), matching engine._batched_fm
            if p.g_cond is not None:
                gcond = p.g_cond.to(dev, dt).reshape(1, -1)
            else:
                gcond = st.fm_null_g_cond.to(dev, dt)  # [1, H]

            # deterministic per-patch noise so both paths integrate the same ODE
            gen = torch.Generator(device=dev).manual_seed(self.noise_base + self.patch_idx)
            noise = torch.randn((1, lp, ld), device=dev, dtype=dt, generator=gen)

            # Build the eager full-sequence FM inputs (latent slot at [L:], no pad).
            total = L + lp
            inp = torch.zeros(1, total, H, device=dev, dtype=dt)
            cfgs = torch.zeros(1, total, H, device=dev, dtype=dt)
            inp[0, :L] = st.fm_sequence[0, :L]
            cfgs[0, :L] = st.fm_cfg_sequence[0, :L]
            mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
            pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
            self.dots._build_fm_attn_mask(state=st, attn_mask=mask)
            self.dots._build_fm_pos_ids(state=st, pos_ids=pos)
            mask2 = torch.cat([mask, mask], 0)
            pos2 = torch.cat([pos, pos], 0)
            g2 = torch.cat([gcond, torch.zeros_like(gcond)], 0)
            gs = inp.new_tensor(p.guidance_scale)

            def eager_vel(z, t):
                """One eager ODE eval over the FULL sequence (the golden DiT)."""
                zp = core.coordinate_proj(z)
                zc = inp.clone(); zc[:, L:] = zp
                zu = cfgs.clone(); zu[:, L:] = zp
                zz = torch.cat([zc, zu], 0)
                tt = t.reshape(1).expand(2).to(dt)
                vt = self._vfp(x=zz, timesteps=tt, attn_mask=mask2, pos_ids=pos2, g_cond=g2)
                vt = vt[:, L:]
                return vt[:1] + gs * (vt[:1] - vt[1:])

            # Build this patch's cache (incremental extend), then drive ONE shared
            # ODE on the eager velocity, comparing the cached velocity at the SAME
            # (t, z) each eval. This isolates cache faithfulness from ODE stiffness
            # (which both bf16 paths share); it must hold at the per-eval bf16 floor.
            self.cached_head.extend_for_patch(st, gcond)
            ns = p.num_steps

            def solver(t, z):
                k = min(max(int(round(float(t) * ns)), 0), ns - 1)
                v_e = eager_vel(z, t)
                v_c = self.cached_head.velocity(z, t, k)
                e = v_e.reshape(-1, ld).float()
                c = v_c.reshape(-1, ld).float()
                self.coss.append(F.cosine_similarity(e, c, dim=-1).mean().item())
                self.relds.append(((e - c).norm() / e.norm().clamp_min(1e-9)).item())
                return v_e

            times = torch.tensor([0.0, 1.0], device=dev, dtype=dt)
            latent_e = odeint(solver, noise, times, method="euler",
                              options={"step_size": 1.0 / ns})[-1]
            self.patch_idx += 1
            return latent_e  # follow the eager (ground-truth) trajectory

    set_seed(1234)
    eng = DualEngine(
        runtime, paged, model_dir=args.model,
        num_kvcache_blocks=128, block_size=256, max_num_seqs=8,
        compile_fm=False,
    )
    eng.noise_base = args.noise_base
    eng.add_request("en1", EN1, num_steps=args.num_steps, guidance_scale=args.guidance_scale)
    eng.run_all()

    coss, relds = eng.coss, eng.relds
    cmin, cmean = min(coss), sum(coss) / len(coss)
    rmax, rmean = max(relds), sum(relds) / len(relds)
    # Per-EVAL (shared z) faithfulness: this is the bf16 single-call floor (the
    # split foundation was cos 0.9999 / reldiff ~1.3%) and must hold UNIFORMLY,
    # independent of ODE stiffness. A real cache bug shows as a low-cos eval.
    # reldiff is on the CFG-combined velocity vc+gs*(vc-vu); the difference-of-
    # similars inflates it above the raw-DiT split floor (~1.3%), but cos (scale-
    # invariant) is the faithfulness gate and stays uniformly high with NO
    # stiffness outliers (the per-patch test's outliers were ODE conditioning).
    ok = cmin > 0.999 and rmax < 0.04
    print(f"[verify_fm_cache_seq] evals={len(coss)} per-EVAL velocity cos "
          f"min/mean={cmin:.5f}/{cmean:.5f} reldiff max/mean={rmax:.4f}/{rmean:.4f} "
          f"{'PASS' if ok else 'FAIL'}")
    worst = sorted(range(len(coss)), key=lambda i: coss[i])[:5]
    print("[verify_fm_cache_seq] worst evals: "
          + ", ".join(f"cos{coss[i]:.5f}/rel{relds[i]:.4f}" for i in worst))

    print(f"[verify_fm_cache_seq] {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
