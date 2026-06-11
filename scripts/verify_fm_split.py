#!/usr/bin/env python
"""Prefix/active split of the FM DiT forward.

The big single-stream lever is to stop reprocessing the whole FM history on every
ODE step of every patch (O(P^2)). The history rows attend only causally (not to
the latent), so the latent's velocity depends on them ONLY through their per-layer
attention K/V -- and those K/V are identical across patches at a fixed ODE
timestep. So we can cache prefix K/V and recompute only the ~5 active rows
(last hidden + latent patch) per step.

This script de-risks that by validating the SPLIT reimplementation of the DiT:
run the prefix rows [0:L-1] through the layers (causal) to get their per-layer
K/V, then run the active rows [L-1:L+patch] attending to [prefix K/V + active
K/V]. The latent velocity must match the full DiT forward.

Result: latent cos ~0.9999 (bf16 attention-reorder noise) at L=8/30/90 => the
split is faithful; the KV cache can be built on it.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_fm_split.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from dots_tts.runtime import DotsTtsRuntime
    from dots_tts.modules.backbone.layers import apply_rotary_pos_emb

    rt = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    core = rt.model.core
    dit = core.velocity_field_predictor
    H, lp, hp, ld = core.fm_hidden_size, core.latent_patch_size, core.hidden_patch_size, core.latent_dim
    dev, dt = torch.device("cuda"), torch.bfloat16

    def modulate(x, sh, sc):
        return x * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)

    def qkv(a, x, pos):
        q = rearrange(a.q_proj(x), "b n (h d) -> b h n d", h=a.num_heads)
        k = rearrange(a.k_proj(x), "b n (h d) -> b h n d", h=a.num_heads)
        v = rearrange(a.v_proj(x), "b n (h d) -> b h n d", h=a.num_heads)
        q, k = a.q_norm(q), a.k_norm(k)
        if a.rotary_bias:
            r = a.rotary(pos)
            q, k = apply_rotary_pos_emb(r, q), apply_rotary_pos_emb(r, k)
        return q, k, v

    def oproj(a, o):
        return a.o_proj(rearrange(o, "b h n d -> b n (h d)"))

    def split_block(blk, prefix, active, c, pp, pa):
        s_a, sc_a, g_a, s_f, sc_f, g_f = blk.adaLN_modulation(c).chunk(6, dim=1)
        g_a, g_f = g_a.unsqueeze(1), g_f.unsqueeze(1)
        a = blk.attn
        qp, kp, vp = qkv(a, modulate(blk.norm1(prefix), s_a, sc_a), pp)
        qa, ka, va = qkv(a, modulate(blk.norm1(active), s_a, sc_a), pa)
        op = F.scaled_dot_product_attention(qp, kp, vp, is_causal=True)
        prefix = prefix + g_a * oproj(a, op)
        oa = F.scaled_dot_product_attention(qa, torch.cat([kp, ka], 2), torch.cat([vp, va], 2))
        active = active + g_a * oproj(a, oa)
        prefix = prefix + g_f * blk.ffn(modulate(blk.norm2(prefix), s_f, sc_f))
        active = active + g_f * blk.ffn(modulate(blk.norm2(active), s_f, sc_f))
        return prefix, active

    def final(ol, x, c):
        sh, sc = ol.adaLN_modulation(c).chunk(2, dim=1)
        return ol.linear(modulate(ol.norm(x), sh, sc))

    def split_fwd(x, t, pos, g, L):
        c = dit.time_embedder(t)
        c = c + g if g is not None else c
        h = dit.input_layer(x)
        prefix, active = h[:, : L - 1], h[:, L - 1 :]
        pp, pa = pos[:, : L - 1], pos[:, L - 1 :]
        for blk in dit.blocks:
            prefix, active = split_block(blk, prefix, active, c, pp, pa)
        return final(dit.output_layer, active, c)

    ok = True
    for L in (8, 30, 90):
        total = L + lp
        torch.manual_seed(L)
        x = torch.randn(1, total, H, device=dev, dtype=dt)
        t = torch.rand(1, device=dev, dtype=dt)
        g = torch.zeros(1, H, device=dev, dtype=dt)
        st = types.SimpleNamespace(fm_seq_len=L)
        mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
        pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
        rt.model._build_fm_attn_mask(state=st, attn_mask=mask)
        rt.model._build_fm_pos_ids(state=st, pos_ids=pos)
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            vf = dit(x=x, timesteps=t, attn_mask=mask, pos_ids=pos, g_cond=g).float()
            vs = split_fwd(x, t, pos, g, L).float()
        a = vf[:, L:].reshape(-1, ld)
        b = vs[:, hp:].reshape(-1, ld)
        cos = F.cosine_similarity(a, b, dim=-1)
        seq_ok = cos.mean().item() > 0.999
        ok = ok and seq_ok
        print(f"[verify_fm_split] L={L}: latent cos(min/mean)="
              f"{cos.min().item():.5f}/{cos.mean().item():.5f} "
              f"{'PASS' if seq_ok else 'FAIL'}")

    print(f"[verify_fm_split] {'PASS' if ok else 'FAIL'} (split == full DiT forward)")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
