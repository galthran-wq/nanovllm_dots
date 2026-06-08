"""Batched flow-matching ODE step across concurrent requests.

The reference FM solver (`DotsTtsCore.fm_solver_step` / `_flow_matching_step_fm`)
is hardcoded to batch size 1 (`t.reshape(1).repeat(len(z_branches))` yields a
length-2 timestep vector, i.e. only the CFG pair). This reimplements the same
math for N requests at once: every active sequence's 10-step ODE is integrated
together through a single batched DiT forward per step. Since FM is ~81% of
compute and the engine runs requests in lockstep, this is the throughput win.

Per-request histories differ in length, so each is padded to a common
`total_len = max_fm_seq_len + latent_patch_size` with the latent slot at the
end. The existing `_build_fm_attn_mask` already masks the padding gap (its
diagonal-fill for `[fm_seq_len, latent_start)`) and `_build_fm_pos_ids` keeps
each sequence's rotary positions correct regardless of physical padding, so the
padded batch is numerically identical to running each request alone -- in
particular N=1 reproduces the serial path (same noise draw, same DiT math).
"""
from __future__ import annotations

import torch
from torchdiffeq import odeint


@torch.no_grad()
def batched_flow_matching(
    core,
    *,
    input_sequence: torch.Tensor,   # [N, total_len, fm_hidden] (latent slot zeroed)
    cfg_sequence: torch.Tensor,     # [N, total_len, fm_hidden]
    attn_mask: torch.Tensor,        # [N, total_len, total_len] bool
    pos_ids: torch.Tensor,          # [N, total_len] float
    g_cond: torch.Tensor,           # [N, fm_hidden]
    num_steps: int = 10,
    guidance_scale: float = 1.2,
    ode_method: str = "euler",
    noise: torch.Tensor | None = None,
    vfp=None,
) -> torch.Tensor:
    """Integrate the CFG flow-matching ODE for N requests -> [N, patch, latent].

    `noise` (optional, [N, patch, latent]) overrides the random init -- used by
    tests to compare a ragged batch against per-request runs under matched noise.
    """
    n = input_sequence.size(0)
    patch_size = core.latent_patch_size
    latent_start = input_sequence.size(1) - patch_size
    device, dtype = input_sequence.device, input_sequence.dtype
    gs = input_sequence.new_tensor(float(guidance_scale))
    if vfp is None:
        vfp = core.velocity_field_predictor

    # CFG pairing: rows [0:n] conditioned, [n:2n] unconditioned. Masks/positions
    # are shared between a request's two branches; g_cond is zeroed for uncond.
    attn2 = torch.cat([attn_mask, attn_mask], dim=0)
    pos2 = torch.cat([pos_ids, pos_ids], dim=0)
    g2 = torch.cat([g_cond, torch.zeros_like(g_cond)], dim=0)

    def solver(t, z):
        zp = core.coordinate_proj(z)            # [n, patch, fm_hidden]
        z_c = input_sequence.clone()
        z_c[:, latent_start:] = zp
        z_u = cfg_sequence.clone()
        z_u[:, latent_start:] = zp
        z_z = torch.cat([z_c, z_u], dim=0)      # [2n, total_len, fm_hidden]
        t_t = t.reshape(1).expand(2 * n).to(z_z.dtype)
        vt = vfp(x=z_z, timesteps=t_t, attn_mask=attn2, pos_ids=pos2, g_cond=g2)
        # clone: under CUDA-graph (reduce-overhead) compile the output aliases a
        # static buffer that the next ODE eval overwrites; the solver feeds vt
        # back via z, so it must own its memory. Cheap (latent slice only).
        vt = vt[:, latent_start:].clone()       # [2n, patch, latent]
        vt_c, vt_u = vt[:n], vt[n:]
        return vt_c + gs * (vt_c - vt_u)

    if noise is None:
        noise = torch.randn((n, patch_size, core.latent_dim), device=device, dtype=dtype)
    times = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    options = {"step_size": 1.0 / num_steps} if ode_method in ("euler", "midpoint", "rk4") else {}
    traj = odeint(solver, noise, times, atol=1e-5, rtol=1e-5, method=ode_method, options=options)
    return traj[-1]                             # [n, patch, latent]
