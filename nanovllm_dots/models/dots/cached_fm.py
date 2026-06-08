"""Per-timestep KV cache for the FM/DiT head -- the O(P^2) -> O(P) lever.

The dots FM solver re-runs dense attention over the WHOLE growing history on
every ODE eval of every patch. But the history (prefix) rows attend ONLY
causally among themselves (`_build_fm_attn_mask`: rows `[0:fm_seq_len-1]` form a
causal block, never attending to the latent slot), so their per-layer attention
K/V depend only on the prefix -- NOT on the latent being denoised. At a FIXED ODE
timestep the adaLN modulation is also fixed (it is a function of the timestep and
g_cond, not position), so a prefix row's K/V are *identical across patches* at a
given timestep. We cache them.

Per patch, instead of O(L) attention we recompute only the ~5 active rows
(last-hidden 1 + latent patch 4) against the cached prefix K/V, then extend the
cache by the 5 rows that graduate from active to prefix. This makes per-patch FM
work O(1) in history length -> O(P) total instead of O(P^2).

The layer math here mirrors the reference DiT (`modules/backbone/dit.py` +
`layers.py`) exactly and is validated against the full forward by
`scripts/verify_fm_cache.py` (cos ~0.9999, the same bf16-reorder noise as the
split foundation in `verify_fm_split.py`).

Cache is indexed by (cfg branch, ODE timestep, layer). One `FMCache` holds all of
them for a single CFG branch's stream; the engine keeps a cond and an uncond
cache per request.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torchdiffeq import odeint

from dots_tts.modules.backbone.layers import apply_rotary_pos_emb


# --------------------------------------------------------------------------- #
# Layer-math helpers -- IDENTICAL to scripts/verify_fm_split.py (the validated
# faithful reimplementation of the reference DiT internals).
# --------------------------------------------------------------------------- #
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def project_qkv(attn, x, pos):
    """q/k/v projection + head split + qk_norm + rotary, exactly as the reference
    `MultiHeadAttention.forward`. `pos` is [B, n] float row positions."""
    q = rearrange(attn.q_proj(x), "b n (h d) -> b h n d", h=attn.num_heads)
    k = rearrange(attn.k_proj(x), "b n (h d) -> b h n d", h=attn.num_heads)
    v = rearrange(attn.v_proj(x), "b n (h d) -> b h n d", h=attn.num_heads)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rotary_bias:
        r = attn.rotary(pos)
        q, k = apply_rotary_pos_emb(r, q), apply_rotary_pos_emb(r, k)
    return q, k, v


def out_proj(attn, o):
    return attn.o_proj(rearrange(o, "b h n d -> b n (h d)"))


def block_modulation(block, c):
    """adaLN(c).chunk(6); gates pre-unsqueezed for broadcast over the seq dim."""
    s_a, sc_a, g_a, s_f, sc_f, g_f = block.adaLN_modulation(c).chunk(6, dim=1)
    return s_a, sc_a, g_a.unsqueeze(1), s_f, sc_f, g_f.unsqueeze(1)


def final_velocity(dit, x, c):
    """FinalLayer: adaLN(c).chunk(2) -> modulate(norm(x)) -> linear -> velocity."""
    ol = dit.output_layer
    sh, sc = ol.adaLN_modulation(c).chunk(2, dim=1)
    return ol.linear(modulate(ol.norm(x), sh, sc))


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class FMCache:
    """Per-(timestep, layer) prefix K/V for one CFG branch of one stream.

    Stored as a list (over the ODE's `num_steps` timesteps) of lists (over the
    DiT's layers) of [k, v] each [B, H, P, Dh]. P (prefix length) grows by the
    history stride (5) each patch via `extend`. Step-(a) validation uses python
    lists + cat; a later step swaps in a preallocated paged buffer for cudagraph.
    """

    def __init__(self, num_steps: int, num_layers: int):
        self.num_steps = num_steps
        self.num_layers = num_layers
        # kv[k][l] = (K, V) or None when empty
        self.kv: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = [
            [None] * num_layers for _ in range(num_steps)
        ]

    def length_at(self, k_index: int) -> int:
        slot = self.kv[k_index][0]
        return 0 if slot is None else slot[0].size(2)

    def length(self) -> int:
        # canonical prefix length; valid only when all timesteps are in lockstep
        # (i.e. between full decode_patch calls, not mid-extend-loop). Asserts it.
        lens = {self.length_at(k) for k in range(self.num_steps)}
        if len(lens) != 1:
            raise RuntimeError(f"FMCache timesteps out of lockstep: lengths={sorted(lens)}")
        return lens.pop()


def _build_c(dit, t_scalar: torch.Tensor, g_cond: torch.Tensor | None, batch: int):
    """DiT condition embedding c = time_embedder(t) (+ g_cond), broadcast to batch.

    `t_scalar` is the ODE timestep: a scalar/1-elem tensor (shared across the
    batch, the production case) or a [B] tensor. `g_cond` is [B, model_dim] or
    None. Returns [B, model_dim]."""
    t = t_scalar.reshape(-1)
    if t.numel() == 1:
        t = t.expand(batch)
    elif t.numel() != batch:
        raise ValueError(f"timestep size {t.numel()} != batch {batch}")
    c = dit.time_embedder(t)
    if g_cond is not None:
        c = c + g_cond.to(c.dtype)  # avoid an accidental upcast if g_cond dtype differs
    return c


@torch.no_grad()
def extend_cache(dit, cache: FMCache, x_new, pos_new, t_scalar, g_cond, k_index: int):
    """Append `x_new` (the new prefix rows) to `cache` at ODE-timestep index
    `k_index`, running them causally through all layers.

    x_new:   [B, m, in_dim]   new prefix rows (already in DiT input space, i.e.
             hidden_proj/latent_proj outputs -- NOT yet input_layer'd)
    pos_new: [B, m]           their rotary positions
    The new rows attend to [existing cache K/V (all)] + [causal among new rows],
    matching the reference prefix mask (a single growing causal block). Prefill
    is just this from an empty cache with m = L-1.
    """
    B = x_new.size(0)
    m = x_new.size(1)
    c = _build_c(dit, t_scalar, g_cond, B)
    h = dit.input_layer(x_new)

    # additive attn bias for the new rows' queries over [cache (P) | new (m)]:
    # cache fully visible, new rows causal among themselves. P is this timestep's
    # own cache length (timesteps grow independently within a decode_patch call).
    P = cache.length_at(k_index)
    neg = torch.finfo(h.dtype).min
    causal = torch.triu(torch.ones(m, m, device=h.device, dtype=torch.bool), 1)
    bias = torch.zeros(m, P + m, device=h.device, dtype=h.dtype)
    bias[:, P:].masked_fill_(causal, neg)
    bias = bias.view(1, 1, m, P + m)

    for l, block in enumerate(dit.blocks):
        s_a, sc_a, g_a, s_f, sc_f, g_f = block_modulation(block, c)
        attn = block.attn
        q, k, v = project_qkv(attn, modulate(block.norm1(h), s_a, sc_a), pos_new)
        slot = cache.kv[k_index][l]
        if slot is None:
            k_full, v_full = k, v
        else:
            ck, cv = slot
            k_full = torch.cat([ck, k], dim=2)
            v_full = torch.cat([cv, v], dim=2)
        o = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=bias)
        h = h + g_a * out_proj(attn, o)
        h = h + g_f * block.ffn(modulate(block.norm2(h), s_f, sc_f))
        cache.kv[k_index][l] = (k_full, v_full)


@torch.no_grad()
def active_forward(dit, cache: FMCache, x_active, pos_active, t_scalar, g_cond, k_index: int):
    """Run the active rows (last-hidden + latent patch) against the cached prefix
    K/V at ODE-timestep `k_index` and return their velocity.

    x_active: [B, a, in_dim]  active rows in DiT input space (last-hidden +
              coordinate_proj(z)); a = hidden_patch_size + latent_patch_size = 5
    pos_active: [B, a]
    Active rows attend to [cache K/V (all) | active K/V (all)] with NO mask --
    full attention, matching the reference mask for the active block (no padding
    in the cached path). Active K/V are NOT written to the cache.
    """
    B = x_active.size(0)
    c = _build_c(dit, t_scalar, g_cond, B)
    h = dit.input_layer(x_active)
    for l, block in enumerate(dit.blocks):
        s_a, sc_a, g_a, s_f, sc_f, g_f = block_modulation(block, c)
        attn = block.attn
        q, k, v = project_qkv(attn, modulate(block.norm1(h), s_a, sc_a), pos_active)
        slot = cache.kv[k_index][l]
        if slot is None:
            k_full, v_full = k, v
        else:
            ck, cv = slot
            k_full = torch.cat([ck, k], dim=2)
            v_full = torch.cat([cv, v], dim=2)
        o = F.scaled_dot_product_attention(q, k_full, v_full)  # full, no mask
        h = h + g_a * out_proj(attn, o)
        h = h + g_f * block.ffn(modulate(block.norm2(h), s_f, sc_f))
    return final_velocity(dit, h, c)


class CachedFMHead:
    """Per-stream cached flow-matching head: integrates the CFG ODE for one patch
    using the per-timestep prefix KV cache, recomputing only the 5 active rows.

    Holds a cond and an uncond `FMCache` (one prefix-K/V history per CFG branch).
    `decode_patch` is called once per audio patch; it (1) extends the caches by
    the prefix rows that have finalized since the last call (read from the
    ground-truth `fm_sequence`/`fm_cfg_sequence` buffers the engine maintains),
    then (2) runs the EULER ODE over the active rows against the caches.

    Scope: ode_method == "euler" (the production/golden path; midpoint/rk4 would
    need cache entries at the intermediate eval times -- the engine keeps the
    uncached batched_fm for those). num_steps/guidance_scale are fixed for the
    life of one stream (the caches are indexed by the euler timestep grid).
    """

    def __init__(self, core, *, num_steps: int, guidance_scale: float, dit=None):
        self.core = core
        self.dit = dit if dit is not None else core.velocity_field_predictor
        self.num_steps = num_steps
        self.guidance_scale = float(guidance_scale)
        self.hp = core.hidden_patch_size
        self.lp = core.latent_patch_size
        self.latent_dim = core.latent_dim
        n_layers = self.dit.num_layers
        self.cc = FMCache(num_steps, n_layers)   # cond branch prefix cache
        self.uc = FMCache(num_steps, n_layers)   # uncond branch prefix cache
        self.cached_P = 0                        # prefix rows already cached
        # euler grid, matching torchdiffeq's fixed-step solver on times=[0,1] with
        # step_size=1/num_steps: it builds the grid in the integ dtype (bf16), so
        # the eval times t_k = bf16(k/num_steps) and the per-step dt_k =
        # grid[k+1]-grid[k] both carry bf16 rounding. We replicate exactly (a plain
        # constant 0.1 step diverges ~1%/step -> several % on stiff patches).
        self._grid = None       # [num_steps+1] eval/endpoint times
        self._dts = None        # [num_steps] per-step dt

    def _ensure_grid(self, device, dtype) -> None:
        if self._grid is not None and self._grid.device == device and self._grid.dtype == dtype:
            return
        g = (torch.arange(self.num_steps + 1, device=device, dtype=torch.float32)
             / self.num_steps).to(dtype)
        self._grid = g
        self._dts = g[1:] - g[:-1]

    def _t(self, k: int, device, dtype) -> torch.Tensor:
        self._ensure_grid(device, dtype)
        return self._grid[k]

    @torch.no_grad()
    def extend_for_patch(self, state, g_cond) -> None:
        """(Step 1 of a patch) Extend the cond/uncond caches by the prefix rows
        that finalized since the last patch ([cached_P : fm_seq_len-1], read from
        the ground-truth fm buffers), at every ODE timestep. Then stash the active-
        row context (last hidden + positions) for `velocity`."""
        core, dit = self.core, self.dit
        L = int(state.fm_seq_len)
        seq, cfg = state.fm_sequence, state.fm_cfg_sequence
        dev, dt = seq.device, seq.dtype
        g_u = torch.zeros_like(g_cond)
        self._ensure_grid(dev, dt)

        if L - 1 > self.cached_P:
            s, e = self.cached_P, L - 1
            new_c, new_u = seq[:, s:e], cfg[:, s:e]
            pos_new = torch.arange(s, e, device=dev, dtype=torch.float32).reshape(1, -1)
            for k in range(self.num_steps):
                tk = self._grid[k]
                extend_cache(dit, self.cc, new_c, pos_new, tk, g_cond, k)
                extend_cache(dit, self.uc, new_u, pos_new, tk, g_u, k)
            self.cached_P = e

        a = self.hp + self.lp
        self._L = L
        self._hid_c = seq[:, L - 1 : L]                   # [1, hp, H]
        self._hid_u = cfg[:, L - 1 : L]
        self._pos_act = torch.arange(L - 1, L - 1 + a, device=dev, dtype=torch.float32).reshape(1, -1)
        self._g_cond, self._g_u = g_cond, g_u

    @torch.no_grad()
    def velocity(self, z, t_scalar, k_index: int) -> torch.Tensor:
        """One CFG ODE eval at timestep `k_index`: active rows (last hidden +
        coordinate_proj(z)) read the cached prefix K/V. Returns [B, lp, latent]."""
        core, dit = self.core, self.dit
        B = z.size(0)
        hid_c = self._hid_c.expand(B, -1, -1)
        hid_u = self._hid_u.expand(B, -1, -1)
        pos = self._pos_act.expand(B, -1)
        zp = core.coordinate_proj(z)                      # [B, lp, H]
        xc = torch.cat([hid_c, zp], dim=1)
        xu = torch.cat([hid_u, zp], dim=1)
        vc = active_forward(dit, self.cc, xc, pos, t_scalar, self._g_cond, k_index)[:, self.hp :]
        vu = active_forward(dit, self.uc, xu, pos, t_scalar, self._g_u, k_index)[:, self.hp :]
        gs = z.new_tensor(self.guidance_scale)
        return vc + gs * (vc - vu)

    @torch.no_grad()
    def decode_patch(self, state, noise, g_cond) -> torch.Tensor:
        """Integrate one patch's CFG euler ODE -> [B, lp, latent_dim] (raw z, pre
        denorm, matching `batched_flow_matching`'s return). Uses the same fixed
        euler grid/dt as torchdiffeq's `odeint` (bit-identical integrator) but as a
        static python loop (no per-step host sync; cudagraph-friendly).

        noise:  [B, lp, latent_dim] ODE init (the engine/caller owns the draw).
        g_cond: [B, model_dim] cond-branch global condition (uncond uses zeros).
        """
        self.extend_for_patch(state, g_cond)
        z = noise.clone()
        for k in range(self.num_steps):
            z = z + self._dts[k] * self.velocity(z, self._grid[k], k)
        return z
