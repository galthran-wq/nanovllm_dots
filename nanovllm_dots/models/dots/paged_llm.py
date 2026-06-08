"""Paged-KV driver for the dots.tts Qwen2 backbone.

dots.tts never decodes one token at a time the way a chat LLM does. Its
generation schedule is fixed up front, and the LLM is fed in *chunks*:

  1. a prefill chunk (text prefix + any injected prompt-patch embeddings),
  2. per generated audio patch, a chunk of ``patch_encoder.out_ds_rate``
     embeddings (the patch-encoder output), and
  3. (interleaved schedules only) text chunks between audio spans.

Every one of those is the same operation: *append L new tokens to a growing
sequence and read out their hidden states*, where the new tokens attend
causally over the whole history. That maps exactly onto the engine's
"chunked-prefill against a paged KV cache" path (`flash_attn_varlen_func`
with a `block_table`): we write the new chunk's K/V into the paged cache at
the next free slots, then attend queries=L against keys=past_len+L. flash-attn
aligns the shorter query block to the *suffix* of the keys (bottom-right
causal), which is precisely append semantics.

This class manages a single sequence's paged cache. Phase 1 uses it per-request
(the batched scheduler comes in Phase 3); the cache layout and append path are
already the ones the continuous-batching runner will share.
"""
from __future__ import annotations

import torch

from nanovllm_dots.layers.attention import Attention
from nanovllm_dots.models.dots.model_llm import QwenLLM
from nanovllm_dots.utils.context import reset_context, set_context


class PagedLLMRunner:
    """Drives a `QwenLLM` over a single-sequence paged KV cache.

    Args:
        llm: the ported Qwen2 backbone (already on device, weights loaded).
        block_size: KV-cache page size (tokens per block).
        max_seq_len: upper bound on a single request's token count; sizes the
            block table. dots schedules are bounded by ``max_audio_patch_count``
            patches * ``out_ds_rate`` + text, comfortably under a few thousand.
        device / dtype: cache placement. dtype must match the LLM compute dtype
            (bf16) so flash-attn sees consistent K/V.
    """

    def __init__(
        self,
        llm: QwenLLM,
        *,
        block_size: int = 256,
        max_seq_len: int = 8192,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        # flash-attn paged-KV requires a page size that is a multiple of 256
        # (same constraint the engine enforces in config.py).
        assert block_size % 256 == 0, "block_size must be a multiple of 256 for paged flash-attn"
        self.llm = llm
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        self._attn_modules = [
            m for m in llm.modules() if isinstance(m, Attention) and m.is_causal
        ]
        if not self._attn_modules:
            raise RuntimeError("QwenLLM exposes no causal Attention modules to cache.")

        self.num_blocks = (max_seq_len + block_size - 1) // block_size
        for m in self._attn_modules:
            m.k_cache = torch.empty(
                self.num_blocks, block_size, m.num_kv_heads, m.head_dim,
                device=self.device, dtype=self.dtype,
            )
            m.v_cache = torch.empty(
                self.num_blocks, block_size, m.num_kv_heads, m.head_dim,
                device=self.device, dtype=self.dtype,
            )
        # Sequential physical blocks for the one resident sequence: token at
        # absolute position p lives at physical slot p (block p//bs, off p%bs).
        self._block_table = torch.arange(
            self.num_blocks, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        self.past_len = 0

    def reset(self) -> None:
        """Start a fresh sequence (drop history; cache is overwritten in place)."""
        self.past_len = 0

    @torch.no_grad()
    def append(self, input_embeds: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Append a chunk and return its hidden states.

        Args:
            input_embeds: ``[L, hidden]`` or ``[1, L, hidden]`` chunk embeddings.
            positions: optional ``[L]`` absolute positions; defaults to the
                contiguous range following the current history.

        Returns:
            ``[L, hidden]`` final-norm hidden states for the appended tokens.
        """
        if input_embeds.dim() == 3:
            if input_embeds.size(0) != 1:
                raise ValueError("PagedLLMRunner handles a single sequence (batch=1).")
            input_embeds = input_embeds[0]
        input_embeds = input_embeds.to(self.device, self.dtype)
        L = input_embeds.size(0)
        start = self.past_len
        total = start + L
        if total > self.num_blocks * self.block_size:
            raise RuntimeError(
                f"sequence length {total} exceeds paged cache capacity "
                f"{self.num_blocks * self.block_size}."
            )

        if positions is None:
            positions = torch.arange(start, total, device=self.device)
        else:
            positions = positions.to(self.device)
        slot_mapping = torch.arange(start, total, dtype=torch.int32, device=self.device)
        cu_q = torch.tensor([0, L], dtype=torch.int32, device=self.device)
        cu_k = torch.tensor([0, total], dtype=torch.int32, device=self.device)
        num_blocks = (total + self.block_size - 1) // self.block_size

        set_context(
            True,
            cu_q,
            cu_k,
            L,
            total,
            slot_mapping,
            None,
            self._block_table[:, :num_blocks],
        )
        try:
            hidden = self.llm(input_embeds, positions)
        finally:
            reset_context()
        self.past_len = total
        return hidden
