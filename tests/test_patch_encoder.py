"""Point of impact: the FlashPatchEncoder (batched flash patch_encoder + prefill).

The flash path swaps the dense-mask O(capacity) SDPA for flash_attn_with_kvcache
over the actual length and runs N requests in one batched call (per-row
cache_seqlens / cache_batch_idx). It must match the reference `patch_encoder`
per patch -- as a kernel, after a prompt prefill, and inside the live engine.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import EN1, MF, ZH1, set_seed

pytestmark = pytest.mark.advanced
MODEL = MF


def test_flash_pe_matches_reference_decode(reference):
    """Batched FlashPatchEncoder.decode_patch == reference per-row decode_patch.

    Rows are advanced in a reversed permutation to exercise cache_batch_idx and
    different per-row history lengths (the continuous-batching case).
    """
    from nanovllm_dots.models.dots.flash_patch_encoder import FlashPatchEncoder

    core = reference.model.core
    pe = core.patch_encoder
    dev, dt = torch.device("cuda"), torch.bfloat16
    n, P = 3, 30
    ps, ld = core.latent_patch_size, core.latent_dim
    cap = 256 * pe.out_ds_rate

    set_seed(1234)
    ref_states = [pe.init_decode_state(max_audio_patch_count=256, batch_size=1, device=dev, dtype=dt)
                  for _ in range(n)]
    flash = FlashPatchEncoder(pe, max_batch=n, max_seq_len=cap, device=dev, dtype=dt)
    rows = torch.tensor(list(reversed(range(n))), device=dev, dtype=torch.int32)

    coss = []
    for _ in range(P):
        patches = [torch.randn(1, ps, ld, device=dev, dtype=dt) for _ in range(n)]
        ref_embeds = []
        for i in range(n):
            st = ref_states[i]
            positions = torch.arange(pe.out_ds_rate, device=dev, dtype=torch.long) + st.seq_len
            emb, conv_tail = pe.decode_patch(patches[i], st.conv_tail, st.layer_caches, positions)
            st.conv_tail.copy_(conv_tail)
            st.seq_len += pe.out_ds_rate
            ref_embeds.append(emb)
        ref = torch.cat(ref_embeds, dim=0)
        got = flash.decode_patch(torch.cat(patches, dim=0), rows)
        c = F.cosine_similarity(ref.reshape(n, -1).float(), got.reshape(n, -1).float(), dim=-1)
        coss.append(c.min().item())

    assert min(coss) > 0.999, f"flash-vs-ref decode cos min={min(coss):.5f}"


def test_flash_pe_prefill_matches_reference(reference):
    """FlashPatchEncoder.prefill == reference prefill, and decode AFTER prefill
    continues correctly (the bit voice-clone prompt seeding relies on)."""
    from nanovllm_dots.models.dots.flash_patch_encoder import FlashPatchEncoder

    core = reference.model.core
    pe = core.patch_encoder
    dev, dt = torch.device("cuda"), torch.bfloat16
    ps, ld = core.latent_patch_size, core.latent_dim
    cap = 256 * pe.out_ds_rate
    P, decode_patches = 22, 15

    set_seed(1234)
    prompt_latents = torch.randn(1, P * ps, ld, device=dev, dtype=dt)

    ref_state = pe.init_decode_state(max_audio_patch_count=256, batch_size=1, device=dev, dtype=dt)
    with torch.autocast("cuda", dtype=dt):
        emb_ref, ref_state = pe.prefill(prompt_latents, ref_state)
    flash = FlashPatchEncoder(pe, max_batch=1, max_seq_len=cap, device=dev, dtype=dt)
    with torch.autocast("cuda", dtype=dt):
        emb_ours = flash.prefill(prompt_latents, 0)

    n = min(emb_ref.size(1), emb_ours.size(1))
    pre_cos = F.cosine_similarity(emb_ref[:, :n].reshape(n, -1).float(),
                                  emb_ours[:, :n].reshape(n, -1).float(), dim=-1).min().item()
    assert pre_cos > 0.999, f"prefill embed cos={pre_cos:.5f}"

    rows = torch.tensor([0], device=dev, dtype=torch.int32)
    coss = []
    for _ in range(decode_patches):
        patch = torch.randn(1, ps, ld, device=dev, dtype=dt)
        positions = torch.arange(pe.out_ds_rate, device=dev, dtype=torch.long) + ref_state.seq_len
        with torch.autocast("cuda", dtype=dt):
            r_emb, ct = pe.decode_patch(patch, ref_state.conv_tail, ref_state.layer_caches, positions)
            ref_state.conv_tail.copy_(ct)
            ref_state.seq_len += pe.out_ds_rate
            o_emb = flash.decode_patch(patch, rows)
        coss.append(F.cosine_similarity(r_emb.reshape(-1).float(), o_emb.reshape(-1).float(), dim=0).item())

    assert min(coss) > 0.999, f"post-prefill decode cos min={min(coss):.5f}"


def test_flash_pe_engine_matches_reference(make_engine, reference):
    """End-to-end: the flash_pe engine path (row alloc/recycle, denormalize batch,
    history plumbing) == reference decode_patch, per patch, on the live trajectory."""
    from nanovllm_dots.models.dots.engine import DotsBatchEngine

    pe = reference.model.core.patch_encoder
    dev, dt = torch.device("cuda"), torch.bfloat16

    class DualEngine(DotsBatchEngine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._shadow: dict[int, object] = {}
            self.coss: list[float] = []

        def _finish_patches_flash(self, seqs, latents, stops):
            core, dots = self.core, self.dots
            rows = []
            for seq, latent in zip(seqs, latents):
                st = seq.custom_payload
                dots._append_history_chunk(st.gen_state, latent.unsqueeze(0))
                if st.pe_row is None:
                    st.pe_row = self._pe_free.pop()
                    self._flash_pe.reset_rows(
                        torch.tensor([st.pe_row], device=self.device, dtype=torch.int32))
                    self._shadow[st.pe_row] = pe.init_decode_state(
                        max_audio_patch_count=256, batch_size=1, device=dev, dtype=dt)
                rows.append(st.pe_row)
            rows_t = torch.tensor(rows, device=self.device, dtype=torch.int32)
            patches = core.io_helper.denormalize(latents)
            embeds = self._flash_pe.decode_patch(patches, rows_t)
            for i, (seq, latent, stop) in enumerate(zip(seqs, latents, stops)):
                st = seq.custom_payload
                sh = self._shadow[st.pe_row]
                positions = torch.arange(pe.out_ds_rate, device=dev, dtype=torch.long) + sh.seq_len
                ref_emb, conv_tail = pe.decode_patch(patches[i:i + 1], sh.conv_tail, sh.layer_caches, positions)
                sh.conv_tail.copy_(conv_tail)
                sh.seq_len += pe.out_ds_rate
                self.coss.append(F.cosine_similarity(
                    ref_emb.reshape(-1).float(), embeds[i].reshape(-1).float(), dim=-1).item())
                self._emit_patch(seq, st, latent.unsqueeze(0), embeds[i], stop)

    set_seed(1234)
    eng = make_engine(cls=DualEngine, fm_accel="cudagraph", flash_pe=True)
    texts = ([EN1, ZH1] * 2)[:3]
    for i, t in enumerate(texts):
        eng.add_request(f"r{i}", t, num_steps=4, guidance_scale=1.2)
    out = eng.run_all()

    assert eng.coss, "no patches produced"
    assert all(v.size(1) > 0 for v in out.values())
    assert min(eng.coss) > 0.999, f"flash-engine vs reference cos min={min(eng.coss):.5f}"
