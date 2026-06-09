"""Length-generic, batched decode for the patch_encoder (VAESemanticEncoder).

The reference `SuperviseEncoder.decode_step` (modules/backbone/layers.py,
`MultiHeadAttention.decode_step`) attends each step over the FULL fixed
`cache_capacity` (= max_audio_patch_count * out_ds_rate, e.g. 512) with a dense
`[B, heads, block, capacity]` bias mask -- regardless of how much history is
actually populated. That is O(capacity) per step even when only a few patches
exist, and it is run as a separate B=1 call per request, so under concurrency the
patch_encoder cost grows LINEARLY with the batch and dominates the pipeline
(profiled: 60-69% of wall at conc 8-16, vs 19% single-stream).

This replaces the dense-mask SDPA with `flash_attn_with_kvcache(..., k=, v=,
cache_seqlens=, causal=True)`, exactly the decode pattern already used by the LLM
(`batched_llm`) and the cached FM (`flash_cached_fm`):
  - the new k/v are appended into the cache at each row's `cache_seqlens`,
  - attention is computed only over the ACTUAL length (flash skips beyond
    cache_seqlens) -- O(actual_len), not O(capacity),
  - per-row `cache_seqlens` makes it a single batched call for N requests at
    DIFFERENT history lengths (continuous batching), no dense mask to build.

It reuses the reference modules verbatim (`q_proj`/`q_norm`/`rotary`/`ffn`/norms,
the `_downsample_step` causal conv, `_project_embeddings`/`out_proj`) -- only the
attention kernel changes, so it is faithful to `decode_patch` at the flash floor
(per-patch latent cos ~1.0; a tiny bf16/flash-vs-SDPA diff that does NOT amplify
because each embed feeds the LLM, not the next patch_encoder step).
"""
from __future__ import annotations

import torch
from flash_attn import flash_attn_with_kvcache

from .flash_cached_fm import _project_qkv_flash, _o_proj_flash


class FlashPatchEncoder:
    """Flash, batched, length-generic replacement for patch_encoder.decode_patch.

    Owns one flash-layout KV cache buffer ([max_batch, capacity, heads, head_dim]
    per layer) plus a per-row conv_tail and int32 cache_seqlens. Active requests
    occupy rows [0:n); the engine packs the n active patches each step.
    """

    def __init__(self, patch_encoder, *, max_batch: int, max_seq_len: int,
                 device, dtype):
        self.pe = patch_encoder                 # VAESemanticEncoder
        self.layers = patch_encoder.encoder.layers
        a0 = self.layers[0].attn
        self.heads, self.head_dim = a0.num_heads, a0.head_dim
        self.out_ds_rate = patch_encoder.out_ds_rate
        self.cap = max_seq_len
        self.device, self.dtype = device, dtype
        self.max_batch = max_batch

        self.kbuf = [torch.zeros(max_batch, self.cap, self.heads, self.head_dim,
                                 device=device, dtype=dtype)
                     for _ in self.layers]
        self.vbuf = [torch.zeros(max_batch, self.cap, self.heads, self.head_dim,
                                 device=device, dtype=dtype)
                     for _ in self.layers]
        self.seqlens = torch.zeros(max_batch, device=device, dtype=torch.int32)
        ds = patch_encoder.ds_proj
        self.conv_tail = torch.zeros(max_batch, ds.in_channels, ds.left_padding,
                                     device=device, dtype=dtype)

    def reset_rows(self, rows: torch.Tensor) -> None:
        """Zero cache/conv-tail/length for the given cache rows (on (re)assignment)."""
        self.seqlens[rows] = 0
        self.conv_tail[rows] = 0
        for k, v in zip(self.kbuf, self.vbuf):
            k[rows] = 0; v[rows] = 0

    @torch.no_grad()
    def decode_patch(self, latent_patches: torch.Tensor,
                     rows: torch.Tensor) -> torch.Tensor:
        """latent_patches: [n, patch_size, latent_dim] (denormalized, as fed to the
        reference decode_patch). `rows`: int32 [n] cache-row indices for these n
        requests (a persistent slot per request -> survives reordering / continuous
        batching). Returns embeds [n, 1, out_dim] and advances each row's cache by
        out_ds_rate. Each row attends over its OWN current seqlens[row] of history,
        so the n requests may be at DIFFERENT lengths in one call."""
        pe = self.pe
        # denormalize() promotes to f32 (mean/var are f32); the encoder weights and
        # cache are bf16 -- cast to match (reference relies on copy_'s implicit cast).
        latent_patches = latent_patches.to(self.dtype)
        # causal downsample conv -> [n, out_ds_rate, hidden]; persist new conv tail
        tail = self.conv_tail[rows]
        x, new_tail = pe._downsample_step(latent_patches, conv_tail=tail)
        self.conv_tail[rows] = new_tail
        block_len = x.size(1)                                  # = out_ds_rate

        seqlens = self.seqlens[rows]                           # int32 [n]
        # per-row query positions: seqlens[row] .. seqlens[row]+block_len-1
        pos = (seqlens[:, None].long()
               + torch.arange(block_len, device=self.device, dtype=torch.long))

        for l, layer in enumerate(self.layers):
            attn = layer.attn
            h = layer.attn_norm(x)
            q, k, v = _project_qkv_flash(attn, h, pos)         # flash layout [n, blk, H, D]
            o = flash_attn_with_kvcache(
                q, self.kbuf[l], self.vbuf[l], k=k, v=v,
                cache_seqlens=seqlens, cache_batch_idx=rows, causal=True,
            )
            x = x + _o_proj_flash(attn, o)
            x = x + layer.ffn(layer.ffn_norm(x))

        self.seqlens[rows] = seqlens + block_len
        return pe._project_embeddings(x)                       # [n, 1, out_dim]
