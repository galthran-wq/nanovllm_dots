#!/usr/bin/env python
"""dots.tts-mf: validate our `batched_meanflow` == reference meanflow solver, and
that ragged batching is padding-transparent.

MeanFlow differs from flow-matching: NO CFG (single branch), the DiT additionally
consumes `duration=dt`, and the integrator is an explicit few-step `z += v*dt`
over a uniform [0,1] grid. We check two things on the mf checkpoint:

  (A) faithfulness -- our solver vs a hand-rolled reference loop built directly
      from `core.meanflow_solver_step` (the reference per-step primitive), under
      MATCHED noise on the same synthetic state. Same kernels/order => cos ~1.0.
      Catches a wrong grid, wrong duration, wrong sign/order, or g_cond mishandle.
  (B) padding -- each ragged sequence batched together gets the same latent it
      gets alone (mask / rotary-position transparency), like verify_batched_fm.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_meanflow_fm.py \
        --model models/dots.tts-mf
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
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--lens", default="12,28,7")
    ap.add_argument("--num-steps", type=int, default=4)
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    nfe = args.num_steps

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.batched_fm import batched_meanflow

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    dots = runtime.model
    core = dots.core
    assert getattr(core, "mode", None) == "meanflow", \
        f"expected meanflow checkpoint, got mode={getattr(core, 'mode', None)}"
    H, patch, latent_dim = core.fm_hidden_size, core.latent_patch_size, core.latent_dim
    dev, dt = torch.device("cuda"), torch.bfloat16

    torch.manual_seed(0)
    histories = [torch.randn(1, L, H, device=dev, dtype=dt) for L in lens]
    noises = [torch.randn(1, patch, latent_dim, device=dev, dtype=dt) for _ in lens]

    def build(state_lens, hists):
        n = len(state_lens)
        max_len = max(state_lens)
        total = max_len + patch
        inp = torch.zeros(n, total, H, device=dev, dtype=dt)
        mask = torch.zeros(n, total, total, dtype=torch.bool, device=dev)
        pos = torch.zeros(n, total, dtype=torch.float32, device=dev)
        for i, L in enumerate(state_lens):
            inp[i, :L] = hists[i][0, :L]
            st = types.SimpleNamespace(fm_seq_len=L)
            dots._build_fm_attn_mask(state=st, attn_mask=mask[i : i + 1])
            dots._build_fm_pos_ids(state=st, pos_ids=pos[i : i + 1])
        g = torch.zeros(n, H, device=dev, dtype=dt)
        return inp, mask, pos, g

    @torch.no_grad()
    def reference_meanflow(inp, mask, pos, g, noise):
        """Hand-rolled reference loop from core.meanflow_solver_step (n=1)."""
        n = inp.size(0)
        z = noise.clone()
        times = torch.linspace(0.0, 1.0, nfe + 1, device=dev, dtype=dt)
        for step in range(nfe):
            t = times[step].expand(n)
            ddt = (times[step + 1] - times[step]).expand(n)
            z = core.meanflow_solver_step(
                z, t=t, dt=ddt, input_sequence=inp, attn_mask=mask,
                pos_ids=pos, patch_size=patch, g_cond=g,
            )
        return z

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dt):
        # (A) faithfulness vs reference solver, per sequence, matched noise.
        ok_a = True
        for i, L in enumerate(lens):
            inp, mask, pos, g = build([L], [histories[i]])
            ours = batched_meanflow(
                core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                g_cond=g, num_steps=nfe, noise=noises[i],
            )[0].float()
            ref = reference_meanflow(inp, mask, pos, g, noises[i])[0].float()
            cos = F.cosine_similarity(ours, ref, dim=-1)
            seq_ok = cos.min().item() > 0.999
            ok_a = ok_a and seq_ok
            print(f"[verify_meanflow_fm] (A) seq{i} L={L} ours-vs-ref "
                  f"cos(min/mean)={cos.min().item():.6f}/{cos.mean().item():.6f} "
                  f"{'PASS' if seq_ok else 'FAIL'}")

        # (B) padding transparency: ragged batch == alone.
        alone = []
        for i, L in enumerate(lens):
            inp, mask, pos, g = build([L], [histories[i]])
            alone.append(batched_meanflow(
                core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                g_cond=g, num_steps=nfe, noise=noises[i],
            )[0].float())
        inp, mask, pos, g = build(lens, histories)
        batched = batched_meanflow(
            core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
            g_cond=g, num_steps=nfe, noise=torch.cat(noises, dim=0),
        ).float()
        ok_b = True
        for i, L in enumerate(lens):
            cos = F.cosine_similarity(batched[i], alone[i], dim=-1)
            seq_ok = cos.min().item() > 0.995
            ok_b = ok_b and seq_ok
            print(f"[verify_meanflow_fm] (B) seq{i} L={L} batch-vs-alone "
                  f"cos(min/mean)={cos.min().item():.6f}/{cos.mean().item():.6f} "
                  f"{'PASS' if seq_ok else 'FAIL'}")

    # (C) cudagraph faithfulness: the CudaGraphRunner records the eager DiT
    # kernels, so graphed meanflow must equal eager meanflow (with the duration
    # input now wired through the runner). Per-patch, matched noise.
    from nanovllm_dots.models.dots.cudagraph_dit import (
        CudaGraphRunner, make_dit_capture_safe,
    )
    make_dit_capture_safe(core.velocity_field_predictor, dev)
    runner = CudaGraphRunner(core.velocity_field_predictor)
    ok_c = True
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dt):
        for i, L in enumerate(lens):
            inp, mask, pos, g = build([L], [histories[i]])
            eager = batched_meanflow(
                core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                g_cond=g, num_steps=nfe, noise=noises[i],
            )[0].float()
            graphed = batched_meanflow(
                core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                g_cond=g, num_steps=nfe, noise=noises[i], vfp=runner,
            )[0].float()
            cos = F.cosine_similarity(eager, graphed, dim=-1)
            seq_ok = cos.min().item() > 0.999
            ok_c = ok_c and seq_ok
            print(f"[verify_meanflow_fm] (C) seq{i} L={L} graphed-vs-eager "
                  f"cos(min/mean)={cos.min().item():.6f}/{cos.mean().item():.6f} "
                  f"{'PASS' if seq_ok else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"[verify_meanflow_fm] {'PASS' if ok else 'FAIL'} "
          f"(A faithful={ok_a}, B padding={ok_b}, C cudagraph={ok_c})")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
