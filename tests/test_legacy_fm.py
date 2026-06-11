"""Legacy FM (flow-matching) path -- the soar checkpoint's DiT cache machinery.

These validate the predecessors of the current shipping path: the prefix/active
DiT split, the SDPA `cached_fm` head and its incremental cache, the
`flash_cached_fm` head (+ its whole-patch graph), batched flow-matching, and the
torch.compile FM speedup. Superseded by the MeanFlow + flash-pe + cudagraph path
(see test_meanflow.py / test_fm via the engine), kept for regression coverage of
the FM model. All run on the soar checkpoint.
"""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from conftest import EN1, SOAR, set_seed

pytestmark = [pytest.mark.advanced, pytest.mark.legacy]
MODEL = SOAR


# --------------------------------------------------------------------------- #
# DiT split + SDPA cache primitives (no engine)                               #
# --------------------------------------------------------------------------- #
def test_fm_dit_prefix_active_split(reference):
    """Hand-rolled prefix/active split of the DiT forward == full DiT (verify_fm_split)."""
    from einops import rearrange
    from dots_tts.modules.backbone.layers import apply_rotary_pos_emb

    core = reference.model.core
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

    for L in (8, 30, 90):
        total = L + lp
        torch.manual_seed(L)
        x = torch.randn(1, total, H, device=dev, dtype=dt)
        t = torch.rand(1, device=dev, dtype=dt)
        g = torch.zeros(1, H, device=dev, dtype=dt)
        st = types.SimpleNamespace(fm_seq_len=L)
        mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
        pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
        reference.model._build_fm_attn_mask(state=st, attn_mask=mask)
        reference.model._build_fm_pos_ids(state=st, pos_ids=pos)
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            vf = dit(x=x, timesteps=t, attn_mask=mask, pos_ids=pos, g_cond=g).float()
            vs = split_fwd(x, t, pos, g, L).float()
        cmean = F.cosine_similarity(vf[:, L:].reshape(-1, ld), vs[:, hp:].reshape(-1, ld), dim=-1).mean().item()
        assert cmean > 0.999, f"L={L} split-vs-full cos mean={cmean:.5f}"


def test_fm_cache_primitives_match_full_dit(reference):
    """extend_cache (prefill prefix) + active_forward == full DiT (verify_fm_cache)."""
    from nanovllm_dots.models.dots.cached_fm import FMCache, extend_cache, active_forward

    core = reference.model.core
    dit = core.velocity_field_predictor
    H, lp, hp, ld = core.fm_hidden_size, core.latent_patch_size, core.hidden_patch_size, core.latent_dim
    dev, dt = torch.device("cuda"), torch.bfloat16
    n_layers = dit.num_layers

    for B in (1, 2):
        for L in (8, 30, 90):
            total = L + lp
            torch.manual_seed(1000 * B + L)
            x = torch.randn(B, total, H, device=dev, dtype=dt)
            t = torch.rand(B, device=dev, dtype=dt)
            g = torch.zeros(B, H, device=dev, dtype=dt)
            st = types.SimpleNamespace(fm_seq_len=L)
            mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
            pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
            reference.model._build_fm_attn_mask(state=st, attn_mask=mask)
            reference.model._build_fm_pos_ids(state=st, pos_ids=pos)
            mask_b, pos_b = mask.expand(B, -1, -1), pos.expand(B, -1)
            with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                vf = dit(x=x, timesteps=t, attn_mask=mask_b, pos_ids=pos_b, g_cond=g).float()
                cache = FMCache(num_steps=1, num_layers=n_layers)
                extend_cache(dit, cache, x_new=x[:, : L - 1], pos_new=pos_b[:, : L - 1],
                             t_scalar=t, g_cond=g, k_index=0)
                vs = active_forward(dit, cache, x_active=x[:, L - 1 :], pos_active=pos_b[:, L - 1 :],
                                    t_scalar=t, g_cond=g, k_index=0).float()
            cmean = F.cosine_similarity(vf[:, L:].reshape(-1, ld), vs[:, hp:].reshape(-1, ld), dim=-1).mean().item()
            assert cmean > 0.999 and cache.length() == L - 1, f"B={B} L={L} cos={cmean:.5f}"


def test_fm_cache_incremental_extend_equals_monolithic(reference):
    """Incremental 5-row extends build the same prefix cache as one-shot (verify_fm_cache_extend)."""
    from nanovllm_dots.models.dots.cached_fm import FMCache, extend_cache, active_forward

    core = reference.model.core
    dit = core.velocity_field_predictor
    H, lp, hp, ld = core.fm_hidden_size, core.latent_patch_size, core.hidden_patch_size, core.latent_dim
    stride = hp + lp
    dev, dt = torch.device("cuda"), torch.bfloat16
    n_layers = dit.num_layers

    for B in (1, 2):
        for init, n_ext in ((7, 16), (13, 8)):
            P = init + n_ext * stride
            torch.manual_seed(7000 * B + P)
            x_prefix = torch.randn(B, P, H, device=dev, dtype=dt)
            x_act = torch.randn(B, hp + lp, H, device=dev, dtype=dt)
            t = torch.rand(1, device=dev, dtype=dt)
            g = torch.zeros(B, H, device=dev, dtype=dt)
            pos_prefix = torch.arange(P, device=dev, dtype=torch.float32).expand(B, -1)
            pos_act = torch.arange(P, P + hp + lp, device=dev, dtype=torch.float32).expand(B, -1)
            with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                cm = FMCache(num_steps=1, num_layers=n_layers)
                extend_cache(dit, cm, x_new=x_prefix, pos_new=pos_prefix, t_scalar=t, g_cond=g, k_index=0)
                ci = FMCache(num_steps=1, num_layers=n_layers)
                extend_cache(dit, ci, x_new=x_prefix[:, :init], pos_new=pos_prefix[:, :init],
                             t_scalar=t, g_cond=g, k_index=0)
                for j in range(n_ext):
                    s = init + j * stride
                    extend_cache(dit, ci, x_new=x_prefix[:, s:s + stride], pos_new=pos_prefix[:, s:s + stride],
                                 t_scalar=t, g_cond=g, k_index=0)
                vm_out = active_forward(dit, cm, x_act, pos_act, t, g, 0).float()
                vi_out = active_forward(dit, ci, x_act, pos_act, t, g, 0).float()
            cmean = F.cosine_similarity(vm_out[:, hp:].reshape(-1, ld), vi_out[:, hp:].reshape(-1, ld), dim=-1).mean().item()
            assert cm.length() == P == ci.length()
            assert cmean > 0.999, f"B={B} P={P} incremental-vs-monolithic cos={cmean:.5f}"


def test_batched_flow_matching_padding_transparent(reference):
    """Ragged batched flow-matching == each sequence run alone (verify_batched_fm)."""
    from nanovllm_dots.models.dots.batched_fm import batched_flow_matching

    dots = reference.model
    core = dots.core
    H, patch, ld = core.fm_hidden_size, core.latent_patch_size, core.latent_dim
    dev, dt = torch.device("cuda"), torch.bfloat16
    lens = [12, 28, 7]

    torch.manual_seed(0)
    histories = [torch.randn(1, L, H, device=dev, dtype=dt) for L in lens]
    cfg_hist = [torch.randn(1, L, H, device=dev, dtype=dt) for L in lens]
    noises = [torch.randn(1, patch, ld, device=dev, dtype=dt) for _ in lens]

    def build(state_lens, hists, cfgs):
        n, max_len = len(state_lens), max(state_lens)
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
        return inp, cfg, mask, pos, torch.zeros(n, H, device=dev, dtype=dt)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dt):
        alone = []
        for i, L in enumerate(lens):
            inp, cfg, mask, pos, g = build([L], [histories[i]], [cfg_hist[i]])
            alone.append(batched_flow_matching(core, input_sequence=inp, cfg_sequence=cfg, attn_mask=mask,
                                               pos_ids=pos, g_cond=g, num_steps=10, guidance_scale=1.2,
                                               noise=noises[i])[0].float())
        inp, cfg, mask, pos, g = build(lens, histories, cfg_hist)
        batched = batched_flow_matching(core, input_sequence=inp, cfg_sequence=cfg, attn_mask=mask,
                                        pos_ids=pos, g_cond=g, num_steps=10, guidance_scale=1.2,
                                        noise=torch.cat(noises, dim=0)).float()
    for i, L in enumerate(lens):
        cmin = F.cosine_similarity(batched[i], alone[i], dim=-1).min().item()
        assert cmin > 0.995, f"seq{i} L={L} batch-vs-alone cos min={cmin:.6f}"


# --------------------------------------------------------------------------- #
# stateful cache heads inside the engine (DualEngine shadow path)             #
# --------------------------------------------------------------------------- #
def _shared_traj_cache_test(make_engine, head_factory, cos_thresh, reld_thresh):
    """Drive the eager trajectory and compare a cache head's per-eval velocity at
    the same (k, z). Shared by the SDPA and flash cache validators."""
    from torchdiffeq import odeint
    from nanovllm_dots.models.dots.engine import DotsBatchEngine

    class DualEngine(DotsBatchEngine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.head = None
            self.coss: list[float] = []
            self.relds: list[float] = []

        def _batched_fm(self, payloads):
            assert len(payloads) == 1
            p = payloads[0]; st = p.gen_state; core = self.core
            L = int(st.fm_seq_len)
            H, lp, ld = core.fm_hidden_size, core.latent_patch_size, core.latent_dim
            dev, dt = self.device, self.dtype
            if self.head is None:
                self.head = head_factory(self, core, p, st)
            gcond = (p.g_cond.to(dev, dt).reshape(1, -1) if p.g_cond is not None
                     else st.fm_null_g_cond.to(dev, dt))
            noise = torch.randn((1, lp, ld), device=dev, dtype=dt)
            total = L + lp
            inp = torch.zeros(1, total, H, device=dev, dtype=dt)
            cfgs = torch.zeros(1, total, H, device=dev, dtype=dt)
            inp[0, :L] = st.fm_sequence[0, :L]
            cfgs[0, :L] = st.fm_cfg_sequence[0, :L]
            mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
            pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
            self.dots._build_fm_attn_mask(state=st, attn_mask=mask)
            self.dots._build_fm_pos_ids(state=st, pos_ids=pos)
            mask2, pos2 = torch.cat([mask, mask], 0), torch.cat([pos, pos], 0)
            g2 = torch.cat([gcond, torch.zeros_like(gcond)], 0)
            gs = inp.new_tensor(p.guidance_scale)

            def eager_vel(z, t):
                zp = core.coordinate_proj(z)
                zc = inp.clone(); zc[:, L:] = zp
                zu = cfgs.clone(); zu[:, L:] = zp
                tt = t.reshape(1).expand(2).to(dt)
                vt = self._vfp(x=torch.cat([zc, zu], 0), timesteps=tt, attn_mask=mask2, pos_ids=pos2, g_cond=g2)[:, L:]
                return vt[:1] + gs * (vt[:1] - vt[1:])

            self.head.extend_for_patch(st, gcond)
            ns = p.num_steps

            def solver(t, z):
                k = min(max(int(round(float(t) * ns)), 0), ns - 1)
                v_e = eager_vel(z, t)
                v_c = self.head.velocity(z, t, k) if head_factory.takes_t else self.head.velocity(z, k)
                e, c = v_e.reshape(-1, ld).float(), v_c.reshape(-1, ld).float()
                self.coss.append(F.cosine_similarity(e, c, dim=-1).mean().item())
                self.relds.append(((e - c).norm() / e.norm().clamp_min(1e-9)).item())
                return v_e

            times = torch.tensor([0.0, 1.0], device=dev, dtype=dt)
            return odeint(solver, noise, times, method="euler", options={"step_size": 1.0 / ns})[-1]

    set_seed(1234)
    eng = make_engine(cls=DualEngine, num_kvcache_blocks=128, max_num_seqs=8, compile_fm=False)
    eng.add_request("en1", EN1, num_steps=10, guidance_scale=1.2)
    eng.run_all()
    assert eng.coss
    assert min(eng.coss) > cos_thresh, f"velocity cos min={min(eng.coss):.5f}"
    assert max(eng.relds) < reld_thresh, f"velocity reldiff max={max(eng.relds):.4f}"


def test_cached_fm_head_faithful_per_eval(make_engine):
    """SDPA CachedFMHead == eager FM, per ODE eval over a real generation (verify_fm_cache_seq)."""
    from nanovllm_dots.models.dots.cached_fm import CachedFMHead

    def factory(eng, core, p, st):
        return CachedFMHead(core, num_steps=p.num_steps, guidance_scale=p.guidance_scale,
                            dit=core.velocity_field_predictor)
    factory.takes_t = True
    _shared_traj_cache_test(make_engine, factory, cos_thresh=0.999, reld_thresh=0.04)


def test_flash_cached_fm_head_faithful_per_eval(make_engine):
    """FlashCachedFMHead == eager FM, per ODE eval over a real generation (verify_flash_cache)."""
    from nanovllm_dots.models.dots.flash_cached_fm import FlashCachedFMHead

    def factory(eng, core, p, st):
        max_patches = st.fm_capacity // (core.hidden_patch_size + core.latent_patch_size) + 1
        return FlashCachedFMHead(core, num_steps=p.num_steps, guidance_scale=p.guidance_scale,
                                 max_patches=max_patches, dit=core.velocity_field_predictor)
    factory.takes_t = False
    _shared_traj_cache_test(make_engine, factory, cos_thresh=0.998, reld_thresh=0.05)


def test_graphed_flash_cache_matches_eager_per_patch(make_engine):
    """GraphedFlashCachedFMHead == eager FlashCachedFMHead, per patch (verify_graphed_cache)."""
    from nanovllm_dots.models.dots.engine import DotsBatchEngine
    from nanovllm_dots.models.dots.flash_cached_fm import FlashCachedFMHead, GraphedFlashCachedFMHead

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
            return le

    set_seed(1234)
    eng = make_engine(cls=DualEngine, num_kvcache_blocks=128, max_num_seqs=8, fm_accel="kvcache")
    eng.add_request("en1", EN1, num_steps=10, guidance_scale=1.2)
    eng.run_all()
    assert eng.coss
    assert min(eng.coss) > 0.999, f"graphed-vs-eager per-patch cos min={min(eng.coss):.5f}"


def test_fm_compile_faithful_per_call(reference):
    """torch.compile(velocity_field_predictor) == eager, per call (verify_fm_compile)."""
    core = reference.model.core
    H, ld = core.fm_hidden_size, core.latent_dim
    vfp_c = torch.compile(core.velocity_field_predictor, mode="default", dynamic=True)

    torch.manual_seed(7)
    for L in (12, 40, 90):
        x = torch.randn(2, L, H, device="cuda", dtype=torch.bfloat16)
        t = torch.rand(2, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(2, L, L, dtype=torch.bool, device="cuda").tril()
        pos = torch.arange(L, device="cuda").float().unsqueeze(0).expand(2, -1).contiguous()
        g = torch.zeros(2, H, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            oe = core.velocity_field_predictor(x=x, timesteps=t, attn_mask=mask, pos_ids=pos, g_cond=g).float()
            oc = vfp_c(x=x, timesteps=t, attn_mask=mask, pos_ids=pos, g_cond=g).float()
        cmean = F.cosine_similarity(oe.reshape(-1, ld), oc.reshape(-1, ld), dim=-1).mean().item()
        assert cmean > 0.99, f"L={L} compiled-vs-eager cos mean={cmean:.5f}"
