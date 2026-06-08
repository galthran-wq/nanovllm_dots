"""Qwen2.5-1.5B LLM backbone for dots.tts, ported onto the nano-vllm runtime.

This is the *only* part of dots.tts that flows through the paged-KV / flash-attn
inference engine. It is a standard Qwen2 decoder stack:
  - QKV projection WITH bias (Qwen2 has q/k/v bias), o_proj without bias
  - GQA (12 query heads, 2 kv heads), head_dim = 128
  - standard RoPE (theta = 1e6), no qk-norm, no LongRoPE scaling
  - RMSNorm (eps 1e-6), SiLU gated MLP

Differences vs the VoxCPM `Cpm4Model` template we forked from:
  - standard `get_rope` instead of MiniCPM LongRoPE
  - `qkv_bias=True`, `apply_qk_norm=False`, no `scale_depth`
  - causal-only (no non-causal encoder branch)

At inference dots follows a fixed `generation_schedule` and never samples from
LLM logits (the eos signal comes from a separate `eos_proj` head on the hidden
state), so this backbone exposes hidden states only — no vocab/lm_head.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from nanovllm_dots.layers.attention import Attention
from nanovllm_dots.layers.layernorm import RMSNorm
from nanovllm_dots.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm_dots.layers.activation import SiluAndMul
from nanovllm_dots.layers.embed_head import VocabParallelEmbedding
from nanovllm_dots.layers.rotary_embedding import get_rope


class QwenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rope_theta: float,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        assert num_heads % tp_size == 0
        assert num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            is_causal=True,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Engine RoPE expects [tokens, heads, head_dim]; reshape before rotary.
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o.view(-1, self.num_heads * self.head_dim)
        return self.o_proj(o)


class QwenMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class QwenDecoderLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        self.self_attn = QwenAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            max_position=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
            # Qwen2 always uses bias on q/k/v projections (o_proj has none),
            # regardless of llm_config's attention_bias field.
            qkv_bias=True,
        )
        self.mlp = QwenMLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(positions, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class QwenLLM(nn.Module):
    """Embedding + decoder stack + final norm. Returns hidden states.

    `forward` accepts precomputed `input_embeds` because dots replaces the
    embeddings at audio-span positions with patch-encoder outputs before the
    LLM sees them. Use `embed_tokens` to get text-token embeddings.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, cfg) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)
