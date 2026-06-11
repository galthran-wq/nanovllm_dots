"""Flash-attention KV-cache FM head -- the cudagraph-able, fast realization of the
per-timestep prefix cache validated in `cached_fm.py`.

The SDPA prototype (`CachedFMHead`) proved the math but is launch-bound: it grows
the cache with `torch.cat` and runs 40 tiny eager forwards per patch. This rewrite
makes it fast:

  - PREALLOCATED cache buffers (no per-step allocation) indexed by (timestep,
    layer), shape [2, cap, heads, dim] where the 2 rows are the CFG cond/uncond
    branches folded into the attention batch (one flash call does both).
  - `flash_attn_with_kvcache` for the attention: fixed query shape (5 active rows,
    or 5 extend rows) with a `cache_seqlens` that varies as the prefix grows -> ONE
    kernel/graph handles every history length (the LLM-decode pattern). It appends
    the new K/V into the cache buffer and attends in one call.
      * extend: causal=True, PERSIST (advance seqlens) -> the prefix grows.
      * active: causal=False, NO persist (write to scratch slots [P:P+5], leave
        seqlens) -> active rows see [all prefix + all active], transient.

Faithful to the reference DiT (validated against eager FM per-eval by
`scripts/verify_flash_cache.py`); numerics differ from the SDPA path only by the
flash-vs-SDPA kernel (still ~cos 0.999).
"""
from __future__ import annotations

import torch
from einops import rearrange
from flash_attn import flash_attn_with_kvcache

from dots_tts.modules.backbone.layers import apply_rotary_pos_emb
from nanovllm_dots.models.dots.cached_fm import (
    modulate, block_modulation, final_velocity,
)


def _project_qkv_flash(attn, x, pos):
    """q/k/v projection + qk_norm + rotary, returned in flash layout [B, n, h, d]."""
    q = rearrange(attn.q_proj(x), "b n (h d) -> b h n d", h=attn.num_heads)
    k = rearrange(attn.k_proj(x), "b n (h d) -> b h n d", h=attn.num_heads)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rotary_bias:
        r = attn.rotary(pos)
        q, k = apply_rotary_pos_emb(r, q), apply_rotary_pos_emb(r, k)
    v = rearrange(attn.v_proj(x), "b n (h d) -> b n h d", h=attn.num_heads)
    # rotary runs in float32 (RotaryEmbedding forces it); flash needs fp16/bf16, so
    # cast q/k back to v's dtype (autocast would have done this for the SDPA path).
    q = q.transpose(1, 2).to(v.dtype).contiguous()     # [B, n, h, d]
    k = k.transpose(1, 2).to(v.dtype).contiguous()
    return q, k, v.contiguous()


def _o_proj_flash(attn, o):
    return attn.o_proj(rearrange(o, "b n h d -> b n (h d)"))


class FlashFMCache:
    """Preallocated per-(timestep, layer) prefix K/V for the 2 CFG branches.

    kbuf/vbuf: [num_layers, num_steps, 2, cap, heads, dim]; row 0 = cond, 1 = uncond.
    Layout is layer-major so a per-layer slice kbuf[l] = [steps, 2, cap, ...] is
    contiguous and reshapes to [steps*2, cap, ...] for the batched-over-timesteps
    extend (one flash call for all ODE timesteps).
    seqlens:   [2] int32, the shared prefix length (all timesteps/layers grow in
    lockstep, one patch = +stride). `cap` must hold max prefix + the 5 active
    scratch slots.
    """

    def __init__(self, num_steps, num_layers, cap, heads, dim, device, dtype):
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.cap = cap
        z = lambda: torch.zeros(num_layers, num_steps, 2, cap, heads, dim,
                                device=device, dtype=dtype)
        self.kbuf = z()
        self.vbuf = z()
        self.seqlens = torch.zeros(2, device=device, dtype=torch.int32)

    def prefix_len(self) -> int:
        return int(self.seqlens[0].item())


class FlashCachedFMHead:
    """Per-stream flash KV-cache FM head (cudagraph-able). Same interface as
    `CachedFMHead`: extend_for_patch / velocity / decode_patch."""

    def __init__(self, core, *, num_steps: int, guidance_scale: float,
                 max_patches: int, dit=None):
        self.core = core
        self.dit = dit if dit is not None else core.velocity_field_predictor
        self.num_steps = num_steps
        self.guidance_scale = float(guidance_scale)
        self.hp = core.hidden_patch_size
        self.lp = core.latent_patch_size
        self.stride = self.hp + self.lp
        self.latent_dim = core.latent_dim
        self.n_layers = self.dit.num_layers
        self.heads = self.dit.blocks[0].attn.num_heads
        self.head_dim = self.dit.blocks[0].attn.head_dim
        self.device = next(self.dit.parameters()).device
        self.dtype = torch.bfloat16
        cap = max_patches * self.stride + self.stride  # + active scratch
        self.cache = FlashFMCache(num_steps, self.n_layers, cap, self.heads,
                                  self.head_dim, self.device, self.dtype)
        self.cached_P = 0
        # euler grid matching torchdiffeq (bit-identical integrator; see CachedFMHead)
        g = (torch.arange(num_steps + 1, device=self.device, dtype=torch.float32)
             / num_steps).to(self.dtype)
        self._grid, self._dts = g, g[1:] - g[:-1]
        self._c = None  # per-timestep condition embeddings [num_steps, 2, model_dim]

    def _build_conditions(self, g_cond):
        """Precompute c_k = time_embedder(t_k) (+ g) for both branches, all steps.
        Returns [num_steps, 2, model_dim]; row0 cond (g_cond), row1 uncond (0)."""
        dit = self.dit
        cs = []
        for k in range(self.num_steps):
            t = self._grid[k].reshape(1).expand(2)
            c = dit.time_embedder(t)                    # [2, model_dim]
            g2 = torch.stack([g_cond.reshape(-1).to(c.dtype),
                              torch.zeros_like(g_cond.reshape(-1).to(c.dtype))], 0)
            cs.append(c + g2)
        return torch.stack(cs, 0)                       # [num_steps, 2, model_dim]

    @torch.no_grad()
    def extend_for_patch(self, state, g_cond) -> None:
        dit, cache = self.dit, self.cache
        L = int(state.fm_seq_len)
        seq, cfg = state.fm_sequence, state.fm_cfg_sequence
        dev = self.device
        self._c = self._build_conditions(g_cond)

        if L - 1 > self.cached_P:
            s, e = self.cached_P, L - 1
            m = e - s
            # [2, m, H]: row0 cond rows, row1 uncond rows
            rows = torch.cat([seq[:, s:e], cfg[:, s:e]], dim=0).to(self.dtype)
            pos = torch.arange(s, e, device=dev, dtype=torch.float32).reshape(1, -1).expand(2, -1)
            for k in range(self.num_steps):
                self._layer_pass(k, rows, pos, self._c[k], causal=True)
            cache.seqlens += m
            self.cached_P = e

        a = self.hp + self.lp
        self._L = L
        self._hid = torch.cat([seq[:, L - 1 : L], cfg[:, L - 1 : L]], dim=0).to(self.dtype)  # [2, hp, H]
        self._pos_act = torch.arange(L - 1, L - 1 + a, device=dev, dtype=torch.float32).reshape(1, -1).expand(2, -1)

    def _layer_pass(self, k_index, h, pos, c, *, causal: bool) -> torch.Tensor:
        """Run rows `h` [2, n, H] through all DiT layers against cache timestep
        `k_index`, with the given causal flag. For extend (causal=True) the K/V
        persist; for active (causal=False) they land in scratch. Returns the final
        hidden [2, n, H] (used by active; ignored by extend)."""
        dit, cache = self.dit, self.cache
        h = dit.input_layer(h)
        for l, block in enumerate(dit.blocks):
            s_a, sc_a, g_a, s_f, sc_f, g_f = block_modulation(block, c)
            attn = block.attn
            q, kk, vv = _project_qkv_flash(attn, modulate(block.norm1(h), s_a, sc_a), pos)
            o = flash_attn_with_kvcache(
                q, cache.kbuf[l, k_index], cache.vbuf[l, k_index],
                k=kk, v=vv, cache_seqlens=cache.seqlens, causal=causal,
            )
            h = h + g_a * _o_proj_flash(attn, o)
            h = h + g_f * block.ffn(modulate(block.norm2(h), s_f, sc_f))
        return h

    @torch.no_grad()
    def velocity(self, z, k_index: int) -> torch.Tensor:
        """One CFG ODE eval at timestep k -> [1, lp, latent_dim]."""
        core = self.core
        zp = core.coordinate_proj(z)                    # [1, lp, H] (single stream)
        # active rows for both branches: [2, a, H]
        h = torch.cat([
            torch.cat([self._hid[0:1], zp], dim=1),
            torch.cat([self._hid[1:2], zp], dim=1),
        ], dim=0)
        out = self._layer_pass(k_index, h, self._pos_act, self._c[k_index], causal=False)
        vel = final_velocity(self.dit, out, self._c[k_index])[:, self.hp :]  # [2, lp, ld]
        vc, vu = vel[0:1], vel[1:2]
        gs = z.new_tensor(self.guidance_scale)
        return vc + gs * (vc - vu)

    @torch.no_grad()
    def decode_patch(self, state, noise, g_cond) -> torch.Tensor:
        self.extend_for_patch(state, g_cond)
        z = noise.clone()
        for k in range(self.num_steps):
            z = z + self._dts[k] * self.velocity(z, k)
        return z


class GraphedFlashCachedFMHead(FlashCachedFMHead):
    """CUDA-graphed FlashCachedFMHead. Captures the fixed-shape [2, 5] active and
    extend layer-passes once (per ODE timestep) and replays them; only the device
    `cache_seqlens` (the prefix length) varies across patches, so a single graph
    per timestep serves every history length (the LLM-decode pattern). Collapses
    the ~360 eager layer-iters/patch into ~20 graph replays.

    All per-patch-varying inputs live in preallocated STATIC buffers (z, hid, pos,
    conditions, seqlens) that the captured graphs read from; each call copies the
    new values in place then replays. Falls back to eager for the (currently never)
    case where a patch's extend isn't exactly `stride` rows.
    """

    def __init__(self, core, *, num_steps, guidance_scale, max_patches, dit=None,
                 warmup: int = 3):
        super().__init__(core, num_steps=num_steps, guidance_scale=guidance_scale,
                         max_patches=max_patches, dit=dit)
        H = core.fm_hidden_size
        a = self.hp + self.lp
        dev, dt = self.device, self.dtype
        z = lambda *s, d=dt: torch.zeros(*s, device=dev, dtype=d)
        ns = num_steps
        model_dim = self.dit.blocks[0].adaLN_modulation[-1].in_features
        # static input buffers (fixed addresses the graphs read)
        self._z_buf = z(1, self.lp, self.latent_dim)
        self._hid_buf = z(2, self.hp, H)
        self._pos_act_buf = z(2, a, d=torch.float32)
        self._c_buf = z(ns, 2, model_dim)
        # batched extend over all ODE timesteps: 2*ns rows = (timestep, branch)
        self._rows_ext_buf = z(2 * ns, self.stride, H)
        self._pos_ext_all = z(2 * ns, self.stride, d=torch.float32)
        self._seqlens_ext = torch.zeros(2 * ns, device=dev, dtype=torch.int32)
        self._c_flat = self._c_buf.view(2 * ns, model_dim)   # view (extend reads all timesteps)
        self._dts_dev = self._dts
        self._gs = torch.tensor(self.guidance_scale, device=dev, dtype=dt)  # static (no in-capture H2D)
        self._c_built = False  # c_k = time_embedder(t_k)+g_cond is constant/stream -> build once
        self.warmup = warmup
        self._g_active: dict[int, dict] = {}   # k -> captured active graph
        self._g_extend_all: dict | None = None  # one graph: extend all timesteps

    # -- capture helpers ----------------------------------------------------
    def _capture(self, fn):
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        torch.cuda.synchronize()
        return graph, out

    def _active_fn(self, k):
        core = self.core
        zp = core.coordinate_proj(self._z_buf)
        h = torch.cat([
            torch.cat([self._hid_buf[0:1], zp], dim=1),
            torch.cat([self._hid_buf[1:2], zp], dim=1),
        ], dim=0)
        out = self._layer_pass(k, h, self._pos_act_buf, self._c_buf[k], causal=False)
        vel = final_velocity(self.dit, out, self._c_buf[k])[:, self.hp :]
        return vel[0:1] + self._gs * (vel[0:1] - vel[1:2])

    def _extend_all_fn(self):
        """Batched extend over ALL ODE timesteps in one pass: 2*ns rows (timestep,
        branch) through the layers, each layer one flash call over the cache viewed
        as [2*ns, cap, h, d]. The new prefix rows are the same for every timestep
        (only the modulation c[k] differs), so this is 18 layer-passes/patch
        instead of ns*18. Side-effecting cache write (causal, persist)."""
        dit, cache = self.dit, self.cache
        ns, cap = self.num_steps, self.cache.cap
        h = dit.input_layer(self._rows_ext_buf)              # [2ns, stride, model_dim]
        for l, block in enumerate(dit.blocks):
            s_a, sc_a, g_a, s_f, sc_f, g_f = block_modulation(block, self._c_flat)
            attn = block.attn
            q, kk, vv = _project_qkv_flash(attn, modulate(block.norm1(h), s_a, sc_a), self._pos_ext_all)
            kb = cache.kbuf[l].reshape(2 * ns, cap, self.heads, self.head_dim)
            vb = cache.vbuf[l].reshape(2 * ns, cap, self.heads, self.head_dim)
            o = flash_attn_with_kvcache(q, kb, vb, k=kk, v=vv,
                                        cache_seqlens=self._seqlens_ext, causal=True)
            h = h + g_a * _o_proj_flash(attn, o)
            h = h + g_f * block.ffn(modulate(block.norm2(h), s_f, sc_f))
        return self._z_buf  # dummy output

    @torch.no_grad()
    def prime_to(self, state, g_cond) -> None:
        """Bulk-extend the cache to cover the whole prefix [cached_P : fm_seq_len-1]
        in ONE eager pass (arbitrary length, not graphed). Used to switch into the
        cache mid-generation (the length-adaptive hybrid): cudagraph-full runs the
        short-history patches, then we prime the cache once and decode_patch the
        rest at O(1)/patch. O(L) one-time; subsequent extends are graphed stride."""
        L = int(state.fm_seq_len)
        if not self._c_built:
            self._c_buf.copy_(self._build_conditions(g_cond))
            self._c_built = True
        if L - 1 <= self.cached_P:
            return
        seq, cfg = state.fm_sequence, state.fm_cfg_sequence
        dev = self.device
        ns, cap = self.num_steps, self.cache.cap
        s, e = self.cached_P, L - 1
        m, H = e - s, seq.size(-1)
        rows2 = torch.cat([seq[:, s:e], cfg[:, s:e]], dim=0)              # [2, m, H]
        rows = rows2.unsqueeze(0).expand(ns, -1, -1, -1).reshape(2 * ns, m, H)
        pos = torch.arange(s, e, device=dev, dtype=torch.float32).reshape(1, -1).expand(2 * ns, -1)
        seqlens = torch.full((2 * ns,), s, device=dev, dtype=torch.int32)
        dit = self.dit
        h = dit.input_layer(rows)
        for l, block in enumerate(dit.blocks):
            s_a, sc_a, g_a, s_f, sc_f, g_f = block_modulation(block, self._c_flat)
            attn = block.attn
            q, kk, vv = _project_qkv_flash(attn, modulate(block.norm1(h), s_a, sc_a), pos)
            kb = self.cache.kbuf[l].reshape(2 * ns, cap, self.heads, self.head_dim)
            vb = self.cache.vbuf[l].reshape(2 * ns, cap, self.heads, self.head_dim)
            o = flash_attn_with_kvcache(q, kb, vb, k=kk, v=vv, cache_seqlens=seqlens, causal=True)
            h = h + g_a * _o_proj_flash(attn, o)
            h = h + g_f * block.ffn(modulate(block.norm2(h), s_f, sc_f))
        self.cache.seqlens.fill_(e)
        self.cached_P = e

    # -- overrides ----------------------------------------------------------
    @torch.no_grad()
    def extend_for_patch(self, state, g_cond) -> None:
        L = int(state.fm_seq_len)
        seq, cfg = state.fm_sequence, state.fm_cfg_sequence
        dev = self.device
        a = self.hp + self.lp
        if not self._c_built:   # constant across patches for a stream; build once
            self._c_buf.copy_(self._build_conditions(g_cond))
            self._c_built = True

        if L - 1 > self.cached_P:
            s, e = self.cached_P, L - 1
            m = e - s
            if m != self.stride:
                raise RuntimeError(
                    f"graphed extend expects stride={self.stride} rows, got {m}.")
            ns, H = self.num_steps, seq.size(-1)
            rows2 = torch.cat([seq[:, s:e], cfg[:, s:e]], dim=0)          # [2, stride, H]
            self._rows_ext_buf.view(ns, 2, self.stride, H).copy_(rows2.unsqueeze(0))
            pos_ext = torch.arange(s, e, device=dev, dtype=torch.float32)
            self._pos_ext_all.view(ns, 2, self.stride).copy_(pos_ext.view(1, 1, self.stride))
            self._seqlens_ext.fill_(s)                                    # write prefix at P_old
            if self._g_extend_all is None:
                graph, _ = self._capture(self._extend_all_fn)
                self._g_extend_all = {"graph": graph}
            self._g_extend_all["graph"].replay()                         # capture records only; execute now
            self.cache.seqlens.fill_(e)                                   # active reads P_new
            self.cached_P = e

        self._hid_buf.copy_(torch.cat([seq[:, L - 1 : L], cfg[:, L - 1 : L]], dim=0))
        self._pos_act_buf.copy_(
            torch.arange(L - 1, L - 1 + a, device=dev, dtype=torch.float32).reshape(1, -1).expand(2, -1))

    @torch.no_grad()
    def decode_patch(self, state, noise, g_cond) -> torch.Tensor:
        self.extend_for_patch(state, g_cond)
        z = noise.clone()
        for k in range(self.num_steps):
            self._z_buf.copy_(z)
            if k not in self._g_active:
                graph, out = self._capture(lambda k=k: self._active_fn(k))
                self._g_active[k] = {"graph": graph, "out": out}
            self._g_active[k]["graph"].replay()       # capture records only; execute now
            z = z + self._dts_dev[k] * self._g_active[k]["out"]
        return z
