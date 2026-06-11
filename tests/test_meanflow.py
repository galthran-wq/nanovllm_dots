"""Point of impact: the MeanFlow FM/DiT head (dots.tts-mf, the shipping path).

MeanFlow differs from flow-matching: no CFG, the DiT consumes duration=dt, and the
integrator is a few-step `z += v*dt` over a uniform [0,1] grid. We assert our
`batched_meanflow` matches the reference solver, is padding-transparent under
ragged batching, and that the cudagraph / whole-patch-graph paths equal eager.
"""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from conftest import EN1, MF, set_seed

pytestmark = pytest.mark.advanced
MODEL = MF
LENS = [12, 28, 7]
NFE = 4


def _meanflow_inputs(reference):
    """Reproduce the synthetic FM state the script built (seed 0)."""
    dots = reference.model
    core = dots.core
    assert getattr(core, "mode", None) == "meanflow", \
        f"expected a meanflow checkpoint, got mode={getattr(core, 'mode', None)}"
    H, patch, ld = core.fm_hidden_size, core.latent_patch_size, core.latent_dim
    dev, dt = torch.device("cuda"), torch.bfloat16

    set_seed(0)
    histories = [torch.randn(1, L, H, device=dev, dtype=dt) for L in LENS]
    noises = [torch.randn(1, patch, ld, device=dev, dtype=dt) for _ in LENS]

    def build(state_lens, hists):
        n, max_len = len(state_lens), max(state_lens)
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

    return dots, core, patch, histories, noises, build


def test_meanflow_matches_reference_solver(reference):
    """(A) our batched_meanflow == a hand-rolled loop over core.meanflow_solver_step."""
    from nanovllm_dots.models.dots.batched_fm import batched_meanflow
    dots, core, patch, histories, noises, build = _meanflow_inputs(reference)
    dev, dt = torch.device("cuda"), torch.bfloat16

    @torch.no_grad()
    def reference_meanflow(inp, mask, pos, g, noise):
        z = noise.clone()
        times = torch.linspace(0.0, 1.0, NFE + 1, device=dev, dtype=dt)
        for step in range(NFE):
            t = times[step].expand(inp.size(0))
            ddt = (times[step + 1] - times[step]).expand(inp.size(0))
            z = core.meanflow_solver_step(z, t=t, dt=ddt, input_sequence=inp,
                                          attn_mask=mask, pos_ids=pos, patch_size=patch, g_cond=g)
        return z

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dt):
        for i, L in enumerate(LENS):
            inp, mask, pos, g = build([L], [histories[i]])
            ours = batched_meanflow(core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                                    g_cond=g, num_steps=NFE, noise=noises[i])[0].float()
            ref = reference_meanflow(inp, mask, pos, g, noises[i])[0].float()
            cmin = F.cosine_similarity(ours, ref, dim=-1).min().item()
            assert cmin > 0.999, f"seq{i} L={L} ours-vs-ref cos min={cmin:.6f}"


def test_meanflow_batching_padding_transparent(reference):
    """(B) each ragged sequence batched together == run alone."""
    from nanovllm_dots.models.dots.batched_fm import batched_meanflow
    dots, core, patch, histories, noises, build = _meanflow_inputs(reference)
    dt = torch.bfloat16

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dt):
        alone = []
        for i, L in enumerate(LENS):
            inp, mask, pos, g = build([L], [histories[i]])
            alone.append(batched_meanflow(core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                                          g_cond=g, num_steps=NFE, noise=noises[i])[0].float())
        inp, mask, pos, g = build(LENS, histories)
        batched = batched_meanflow(core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                                   g_cond=g, num_steps=NFE, noise=torch.cat(noises, dim=0)).float()
        for i, L in enumerate(LENS):
            cmin = F.cosine_similarity(batched[i], alone[i], dim=-1).min().item()
            assert cmin > 0.995, f"seq{i} L={L} batch-vs-alone cos min={cmin:.6f}"


def test_meanflow_cudagraph_matches_eager(reference):
    """(C) CudaGraphRunner records the eager DiT kernels -> graphed == eager."""
    from nanovllm_dots.models.dots.batched_fm import batched_meanflow
    from nanovllm_dots.models.dots.cudagraph_dit import CudaGraphRunner, make_dit_capture_safe
    dots, core, patch, histories, noises, build = _meanflow_inputs(reference)
    dev, dt = torch.device("cuda"), torch.bfloat16

    make_dit_capture_safe(core.velocity_field_predictor, dev)
    runner = CudaGraphRunner(core.velocity_field_predictor)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dt):
        for i, L in enumerate(LENS):
            inp, mask, pos, g = build([L], [histories[i]])
            eager = batched_meanflow(core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                                     g_cond=g, num_steps=NFE, noise=noises[i])[0].float()
            graphed = batched_meanflow(core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                                       g_cond=g, num_steps=NFE, noise=noises[i], vfp=runner)[0].float()
            cmin = F.cosine_similarity(eager, graphed, dim=-1).min().item()
            assert cmin > 0.999, f"seq{i} L={L} graphed-vs-eager cos min={cmin:.6f}"


def test_wholepatch_graph_matches_eager(make_engine):
    """verify_mfgraph: the whole-patch MeanFlow CUDA graph (full nfe-step solver
    captured) == eager batched_meanflow, per patch, over a real trajectory."""
    from nanovllm_dots.models.dots.engine import DotsBatchEngine
    from nanovllm_dots.models.dots.batched_fm import batched_meanflow
    from nanovllm_dots.models.dots.graphed_meanflow import GraphedMeanflow
    from nanovllm_dots.models.dots.cudagraph_dit import make_dit_capture_safe

    class DualEngine(DotsBatchEngine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            make_dit_capture_safe(self.core.velocity_field_predictor, self.device)
            self._g = GraphedMeanflow(self.core, num_steps=NFE,
                                      dit=self.core.velocity_field_predictor)
            self.coss: list[float] = []

        def _batched_fm(self, payloads):
            assert len(payloads) == 1
            inp, _, mask, pos, gcond, p0, _ = self._pack_fm_inputs(payloads)
            ld = self.core.latent_dim
            noise = torch.randn(1, self.core.latent_patch_size, ld,
                                device=self.device, dtype=self.dtype)
            eager = batched_meanflow(self.core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                                     g_cond=gcond, num_steps=p0.num_steps, noise=noise.clone())
            graphed = self._g(input_sequence=inp, attn_mask=mask, pos_ids=pos,
                              g_cond=gcond, num_steps=p0.num_steps, noise=noise.clone())
            c = F.cosine_similarity(eager.reshape(-1, ld).float(),
                                    graphed.reshape(-1, ld).float(), dim=-1)
            self.coss.append(c.min().item())
            return eager

    set_seed(1234)
    eng = make_engine(cls=DualEngine, fm_accel="none")
    eng.add_request("r0", EN1, num_steps=NFE, guidance_scale=1.2)
    eng.run_all()

    assert eng.coss, "no patches were produced"
    cmin = min(eng.coss)
    assert cmin > 0.999, f"whole-patch-graph vs eager cos min={cmin:.5f}"
