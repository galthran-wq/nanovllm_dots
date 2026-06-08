"""Manual CUDA-graph capture of the FM DiT forward.

Single-stream FM is overhead-bound (<1% MFU: tiny matrices, eager kernel
launches), so we capture the eager DiT forward into a CUDA graph and replay it.
Recording the EAGER kernels keeps the result bit-identical to eager (no inductor
drift), while the launch overhead is gone: single-stream RTF 1.69 -> 0.65, and
cos 1.0 vs the eager golden.

We own the static I/O buffers (copy each call's inputs into fixed buffers,
replay, read a fixed output buffer) -- `torch.compile(reduce-overhead)` instead
corrupts the FM (the ODE feeds the output back and its captured graph reads the
first call's input address) and was also slow here.

One graph per (batch, seq_len) shape. Run UNBUCKETED (fm_len_bucket=0): the FM
history grows by a fixed stride per patch, so only ~one shape per patch-count
recurs across requests -- a bounded graph set, captured once and replayed
EXACTLY. This is both faster and exact vs length bucketing, which pads the
history and (because the autoregressive FM amplifies the per-patch padding bf16
noise) diverges to cos ~0.73 -- a real effect present in eager too, NOT a
cudagraph bug. Capturing a new shape mid-generation is safe (cold first request
is correct). The returned tensor is a static buffer the next replay overwrites,
so the caller clones what it keeps (the ODE solver clones the latent slice).
"""
from __future__ import annotations

import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def make_dit_capture_safe(dit, device: torch.device) -> None:
    """Remove the one non-capturable op from the DiT timestep embedder.

    `TimestepEmbedder.timestep_embedding` builds `torch.arange(...).to(device)`
    every call -- an H2D copy from pageable CPU memory, which aborts CUDA-graph
    capture (and makes torch.compile reduce-overhead silently wrong). The freqs
    are constant, so precompute them on-device once and shadow the staticmethod
    with a capture-safe closure. Mathematically identical to the reference.
    """
    te = dit.time_embedder
    dim = te.frequency_embedding_size
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )

    def timestep_embedding(t, d):
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if d % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    te.timestep_embedding = timestep_embedding  # instance attr shadows the staticmethod


class CudaGraphRunner:
    def __init__(self, fn, *, warmup: int = 3):
        self.fn = fn                       # eager DiT forward (velocity_field_predictor)
        self.warmup = warmup
        self._graphs: dict[tuple, dict] = {}

    @torch.no_grad()
    def __call__(self, *, x, timesteps, attn_mask, pos_ids, g_cond):
        key = (x.shape[0], x.shape[1])     # (2N, total_len)
        g = self._graphs.get(key)
        if g is None:
            g = self._capture(x, timesteps, attn_mask, pos_ids, g_cond)
            self._graphs[key] = g
        g["x"].copy_(x)
        g["t"].copy_(timesteps)
        g["mask"].copy_(attn_mask)
        g["pos"].copy_(pos_ids)
        g["g"].copy_(g_cond)
        g["graph"].replay()
        return g["out"]

    def _capture(self, x, timesteps, attn_mask, pos_ids, g_cond) -> dict:
        # static input buffers (stable addresses the graph reads from)
        sx = x.clone()
        st = timesteps.clone()
        smask = attn_mask.clone()
        spos = pos_ids.clone()
        sg = g_cond.clone()

        # Force a single mask-agnostic SDPA backend. Otherwise SDPA picks its
        # kernel from the capture-time mask sparsity and bakes it into the graph,
        # so a graph captured at one fm_seq_len mishandles the real mask on replay
        # (the standalone-prewarm cos 0.73 bug). mem-efficient handles arbitrary
        # additive masks and is value-agnostic.
        def call():
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return self.fn(x=sx, timesteps=st, attn_mask=smask, pos_ids=spos, g_cond=sg)

        # Capturing while other GPU work is in flight corrupts the graph, so drain
        # the device first.
        torch.cuda.synchronize()
        # Warmup on a side stream so cuBLAS/cuDNN workspaces are allocated before
        # capture (capturing an uninitialized workspace alloc corrupts the graph).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                call()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        torch.cuda.synchronize()
        return {"x": sx, "t": st, "mask": smask, "pos": spos, "g": sg, "out": out, "graph": graph}
