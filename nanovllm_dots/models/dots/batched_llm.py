"""Batched paged-KV driver for the dots.tts Qwen2 backbone.

This is the continuous-batching generalization of `PagedLLMRunner`: it advances
*many* sequences' LLM state in a single varlen flash-attn call. Each sequence
appends its own new chunk (the prefill prompt, or one patch-encoder embedding
token per decode step) against its own paged KV blocks; the chunks are packed
contiguously and attention is isolated per sequence by `cu_seqlens` + per-seq
`block_table`, so batching changes nothing about any sequence's result.

dots emits exactly one LLM token per audio patch (the patch encoder regroups
`out_ds_rate` frames into a single embedding), so this is structurally the same
1-token-per-step continuous batching the engine already does for VoxCPM -- the
only twist is that decode inputs are *embeddings* (from the patch encoder), not
token ids, so the driver consumes `inputs_embeds` directly.

The per-sequence inputs (`block_table`, `num_cached_tokens`, `seq_length`) are
exactly what `engine.sequence.Sequence` exposes, so the Phase-1 `DotsBatchEngine`
feeds this straight from the scheduler/block-manager. CUDA-graph capture is a
Phase-2 concern; this runs eager.
"""
from __future__ import annotations

import torch

from nanovllm_dots.layers.attention import Attention
from nanovllm_dots.models.dots.model_llm import QwenLLM
from nanovllm_dots.utils.context import reset_context, set_context


class BatchedPagedLLM:
    def __init__(
        self,
        llm: QwenLLM,
        *,
        num_blocks: int,
        block_size: int = 256,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        assert block_size % 256 == 0, "block_size must be a multiple of 256 for paged flash-attn"
        self.llm = llm
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        self._attn_modules = [
            m for m in llm.modules() if isinstance(m, Attention) and m.is_causal
        ]
        if not self._attn_modules:
            raise RuntimeError("QwenLLM exposes no causal Attention modules to cache.")
        for m in self._attn_modules:
            m.k_cache = torch.empty(
                num_blocks, block_size, m.num_kv_heads, m.head_dim,
                device=self.device, dtype=self.dtype,
            )
            m.v_cache = torch.empty(
                num_blocks, block_size, m.num_kv_heads, m.head_dim,
                device=self.device, dtype=self.dtype,
            )

    @torch.no_grad()
    def append_batch(
        self,
        chunks: list[torch.Tensor],
        block_tables: list[list[int]],
        cached_lens: list[int],
    ) -> list[torch.Tensor]:
        """Advance a batch of sequences by one varlen append.

        Args:
            chunks: per-seq ``[Li, hidden]`` (or ``[1, Li, hidden]``) new-token
                embeddings to append. ``Li`` may differ per sequence (prefill is
                long; decode is 1).
            block_tables: per-seq list of physical KV block ids (as the
                block-manager assigns them).
            cached_lens: per-seq count of tokens already resident in KV (the
                append starts at this absolute position). 0 for a fresh prefill.

        Returns:
            per-seq ``[Li, hidden]`` final-norm hidden states for the appended
            tokens (callers typically take the last row for eos / FM).
        """
        n = len(chunks)
        if not (n == len(block_tables) == len(cached_lens)):
            raise ValueError("chunks, block_tables, cached_lens must align.")
        if n == 0:
            return []

        positions_list: list[int] = []
        cu_q: list[int] = [0]
        cu_k: list[int] = [0]
        slot_mapping: list[int] = []
        max_q = max_k = 0
        embeds: list[torch.Tensor] = []
        lengths: list[int] = []
        for emb, bt, cached in zip(chunks, block_tables, cached_lens):
            if emb.dim() == 3:
                emb = emb[0]
            emb = emb.to(self.device, self.dtype)
            L = emb.size(0)
            embeds.append(emb)
            lengths.append(L)
            total = cached + L
            if len(bt) * self.block_size < total:
                raise RuntimeError(
                    f"block_table holds {len(bt)} blocks ({len(bt) * self.block_size} slots) "
                    f"but sequence needs {total}."
                )
            positions_list.extend(range(cached, total))
            cu_q.append(cu_q[-1] + L)
            cu_k.append(cu_k[-1] + total)
            max_q = max(max_q, L)
            max_k = max(max_k, total)
            # physical slot for each new absolute position p = block*bs + offset
            for p in range(cached, total):
                slot_mapping.append(bt[p // self.block_size] * self.block_size + p % self.block_size)

        embeds_t = torch.cat(embeds, dim=0)
        positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=self.device)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=self.device)
        slot_t = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        # Match the engine (model_runner.prepare_prefill_context): only pass a
        # block_table when some sequence has a cached prefix (seqlen_k >
        # seqlen_q). An all-fresh-prefill batch then takes the plain packed-varlen
        # flash-attn path (block_tables=None) instead of the paged-prefix path,
        # byte-for-byte matching the validated engine behavior.
        bt_t = None
        if cu_k[-1] > cu_q[-1]:
            max_bt = max(len(bt) for bt in block_tables)
            bt_t = torch.tensor(
                [bt + [-1] * (max_bt - len(bt)) for bt in block_tables],
                dtype=torch.int32, device=self.device,
            )

        set_context(True, cu_q_t, cu_k_t, max_q, max_k, slot_t, None, bt_t)
        try:
            hidden = self.llm(embeds_t, positions)  # [sum Li, hidden]
        finally:
            reset_context()

        out: list[torch.Tensor] = []
        off = 0
        for L in lengths:
            out.append(hidden[off : off + L])
            off += L
        return out
