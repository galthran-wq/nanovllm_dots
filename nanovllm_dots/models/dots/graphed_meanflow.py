"""Whole-patch CUDA graph for the MeanFlow sampler.

The per-patch MeanFlow solver runs `nfe` (~4) ODE steps; each step is a DiT
forward plus eager glue (coordinate_proj, a clone + slot-write, and the
`z += v*dt` update). Wrapping just the DiT in a CUDA graph (CudaGraphRunner)
still pays `nfe` separate graph launches per patch with eager kernels BETWEEN
them -- and FM is launch-bound, so those inter-replay gaps dominate.

This captures the ENTIRE per-patch solver -- all nfe steps, the coordinate_proj,
the clones, and the z-updates -- into ONE graph. A patch becomes a single replay.
The ODE grid (t, dt per step) is constant across patches, so it is baked into
static buffers once. Noise is drawn OUTSIDE the graph (RNG must not be captured)
and copied into a static buffer before replay.

One graph per (batch_size, total_len) shape, exactly like CudaGraphRunner: the FM
history grows by a fixed stride per patch, so a bounded set of shapes recurs and
each is captured once and replayed thereafter. Faithful to eager (records the
eager kernels): per-patch latent cos ~1.0 vs `batched_meanflow`.
"""
from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


class GraphedMeanflow:
    def __init__(self, core, *, num_steps: int = 4, dit=None, warmup: int = 3):
        self.core = core
        self.nfe = num_steps
        self.dit = dit if dit is not None else core.velocity_field_predictor
        self.warmup = warmup
        self.latent_dim = core.latent_dim
        self.patch = core.latent_patch_size
        self._graphs: dict[tuple, dict] = {}

    @torch.no_grad()
    def __call__(self, *, input_sequence, attn_mask, pos_ids, g_cond,
                 num_steps=None, noise=None):
        if num_steps is not None and num_steps != self.nfe:
            raise RuntimeError(
                f"GraphedMeanflow captured for nfe={self.nfe}, got {num_steps}."
            )
        n, total_len = input_sequence.shape[0], input_sequence.shape[1]
        key = (n, total_len)
        g = self._graphs.get(key)
        if g is None:
            g = self._capture(n, total_len, input_sequence.dtype, input_sequence.device)
            self._graphs[key] = g
        g["inp"].copy_(input_sequence)
        g["mask"].copy_(attn_mask)
        g["pos"].copy_(pos_ids)
        g["g"].copy_(g_cond)
        if noise is None:
            g["noise"].normal_()    # fresh noise each patch, drawn OUTSIDE the graph
        else:
            g["noise"].copy_(noise)  # matched noise (tests / reproducibility)
        g["graph"].replay()
        # the next replay overwrites the static output; the caller (engine) copies
        # it out immediately, but clone defensively.
        return g["out"].clone()

    def _capture(self, n, total_len, dtype, device) -> dict:
        core = self.core
        latent_start = total_len - self.patch
        H = core.fm_hidden_size

        inp = torch.zeros(n, total_len, H, device=device, dtype=dtype)
        mask = torch.zeros(n, total_len, total_len, dtype=torch.bool, device=device)
        pos = torch.zeros(n, total_len, dtype=torch.float32, device=device)
        g = torch.zeros(n, H, device=device, dtype=dtype)
        noise = torch.randn(n, self.patch, self.latent_dim, device=device, dtype=dtype)

        # Constant ODE grid baked into static per-step [n] tensors.
        grid = torch.linspace(0.0, 1.0, self.nfe + 1, device=device, dtype=dtype)
        t_static = [grid[s].expand(n).contiguous() for s in range(self.nfe)]
        dt_static = [(grid[s + 1] - grid[s]).expand(n).contiguous() for s in range(self.nfe)]

        def solver():
            # mem-efficient SDPA is mask-agnostic -- otherwise the kernel baked at
            # capture time mishandles a different fm_seq_len mask on replay.
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                z = noise
                for s in range(self.nfe):
                    zp = core.coordinate_proj(z)
                    z_c = inp.clone()
                    z_c[:, latent_start:] = zp
                    vt = self.dit(x=z_c, timesteps=t_static[s], duration=dt_static[s],
                                  attn_mask=mask, pos_ids=pos, g_cond=g)
                    z = z + vt[:, latent_start:] * dt_static[s].view(-1, 1, 1)
                return z

        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                solver()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = solver()
        torch.cuda.synchronize()
        return {"inp": inp, "mask": mask, "pos": pos, "g": g, "noise": noise,
                "out": out, "graph": graph}
