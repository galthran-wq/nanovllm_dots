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
        graph_decode: bool = False,
    ) -> None:
        assert block_size % 256 == 0, "block_size must be a multiple of 256 for paged flash-attn"
        self.llm = llm
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype
        # CUDA-graph capture of the one-token decode forward (per batch size).
        # The only varying device inputs are copied into static buffers each
        # step (embeds/positions/slots/context_lens/block_table); the int32
        # context_lens carrying the growing KV length is the capturable
        # decode pattern, so one graph per batch size serves every length.
        self.graph_decode = graph_decode
        self._decode_graphs: dict[int, dict] = {}

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

        # Pure-decode batch: every sequence appends exactly one token. Route it
        # through flash_attn_with_kvcache (the decode kernel + capturable pattern)
        # instead of the varlen prefill kernel. A 1-token "prefill" against a
        # cached prefix is identical to a decode, so keying on Li==1 is safe.
        embed_dims = [c.dim() for c in chunks]
        sizes = [(c[0] if d == 3 else c).size(0) for c, d in zip(chunks, embed_dims)]
        if all(s == 1 for s in sizes):
            return self._decode_batch(chunks, block_tables, cached_lens)
        return self._varlen_batch(chunks, block_tables, cached_lens)

    @torch.no_grad()
    def _varlen_batch(
        self,
        chunks: list[torch.Tensor],
        block_tables: list[list[int]],
        cached_lens: list[int],
    ) -> list[torch.Tensor]:
        """Generic packed-varlen append (prefill, or mixed-length batches)."""
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

    @torch.no_grad()
    def _decode_batch(
        self,
        chunks: list[torch.Tensor],
        block_tables: list[list[int]],
        cached_lens: list[int],
    ) -> list[torch.Tensor]:
        """One-token-per-seq decode via flash_attn_with_kvcache (paged).

        Each new token sits at absolute position ``cached`` in its sequence; we
        store its K/V into the paged cache at that slot and attend with
        ``cache_seqlens = cached + 1`` (inclusive of self). This is the
        CUDA-graph-capturable decode pattern (the only varying device input is
        the int32 ``context_lens``; the block table is copied in each step).
        """
        bs = self.block_size
        n = len(chunks)
        embeds = []
        positions_list: list[int] = []
        ctx_lens: list[int] = []
        slot_mapping: list[int] = []
        bt_widths = []
        for emb, bt, cached in zip(chunks, block_tables, cached_lens):
            if emb.dim() == 3:
                emb = emb[0]
            embeds.append(emb.to(self.device, self.dtype))   # [1, H]
            pos = cached                                      # new token position
            if len(bt) * bs <= pos:
                raise RuntimeError(
                    f"block_table holds {len(bt)} blocks ({len(bt) * bs} slots) "
                    f"but decode needs position {pos}."
                )
            positions_list.append(pos)
            ctx_lens.append(pos + 1)
            slot_mapping.append(bt[pos // bs] * bs + pos % bs)
            bt_widths.append(len(bt))

        embeds_t = torch.cat(embeds, dim=0)                   # [n, H]

        g = self._decode_graphs.get(n) if self.graph_decode else None
        if g is not None and max(bt_widths) <= g["bt"].size(1):
            # capturable decode: copy inputs into static buffers, replay.
            g["embeds"].copy_(embeds_t)
            g["pos"].copy_(torch.tensor(positions_list, dtype=torch.int64, device=self.device))
            g["ctx"].copy_(torch.tensor(ctx_lens, dtype=torch.int32, device=self.device))
            g["slot"].copy_(torch.tensor(slot_mapping, dtype=torch.int32, device=self.device))
            width = g["bt"].size(1)
            bt_cpu = torch.full((n, width), -1, dtype=torch.int32)
            for i, bt in enumerate(block_tables):
                bt_cpu[i, : len(bt)] = torch.tensor(bt, dtype=torch.int32)
            g["bt"].copy_(bt_cpu.to(self.device))
            g["graph"].replay()
            out = g["out"]
            return [out[i : i + 1] for i in range(n)]

        positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
        ctx_lens_t = torch.tensor(ctx_lens, dtype=torch.int32, device=self.device)
        slot_t = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        max_bt = max(bt_widths)
        bt_t = torch.tensor(
            [bt + [-1] * (max_bt - len(bt)) for bt in block_tables],
            dtype=torch.int32, device=self.device,
        )
        set_context(False, slot_mapping=slot_t, context_lens=ctx_lens_t, block_tables=bt_t)
        try:
            hidden = self.llm(embeds_t, positions)            # [n, H]
        finally:
            reset_context()
        return [hidden[i : i + 1] for i in range(n)]

    @torch.no_grad()
    def warmup_decode(self, batch_sizes=(1,), bt_width: int | None = None) -> None:
        """Pre-capture the decode CUDA graph for each batch size, BEFORE serving.

        store_kvcache writes into the real paged K/V cache at the captured slots,
        so capturing must happen while the cache holds no live sequence data
        (i.e. right after construction). Capturing lazily mid-generation would
        corrupt whatever sequence owns the scratch block. We capture against
        block 0 with context_len=1; the first real prefill overwrites it.
        """
        if not self.graph_decode:
            return
        width = bt_width if bt_width is not None else min(self.num_blocks, 64)
        for n in batch_sizes:
            if n not in self._decode_graphs:
                self._decode_graphs[n] = self._capture_decode(n, width)

    @torch.no_grad()
    def _capture_decode(self, n: int, width: int) -> dict:
        # NOTE: the model forward run under capture includes a @torch.compile RoPE
        # (model_llm rotary). The 3-iteration side-stream warmup below forces its
        # Inductor compilation BEFORE capture, per batch size n. Keep RoPE compiled
        # with mode="default" (NOT reduce-overhead, which nests CUDA graphs and
        # would abort this capture) -- same constraint as the FM DiT graph.
        H = self._attn_modules[0].num_heads * self._attn_modules[0].head_dim
        dev = self.device
        embeds_buf = torch.zeros(n, H, device=dev, dtype=self.dtype)
        pos_buf = torch.zeros(n, dtype=torch.int64, device=dev)
        ctx_buf = torch.ones(n, dtype=torch.int32, device=dev)        # 1 token, block 0
        slot_buf = torch.arange(n, dtype=torch.int32, device=dev)     # distinct slots in block 0
        bt_buf = torch.full((n, width), -1, dtype=torch.int32, device=dev)
        bt_buf[:, 0] = 0                                              # all read block 0

        set_context(False, slot_mapping=slot_buf, context_lens=ctx_buf, block_tables=bt_buf)
        try:
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.llm(embeds_buf, pos_buf)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.llm(embeds_buf, pos_buf)
            torch.cuda.synchronize()
        finally:
            reset_context()
        return {"embeds": embeds_buf, "pos": pos_buf, "ctx": ctx_buf,
                "slot": slot_buf, "bt": bt_buf, "out": out, "graph": graph}
