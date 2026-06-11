"""Native DotsCore: the central hub the engine + accelerators read
(velocity_field_predictor, patch_encoder, the projections, io_helper, config
scalars, audio-span token ids, and an embeddings shim). Replaces the reference
DotsTtsCore at runtime.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .config import load_config
from .io_helper import IOHelper
from .loader import (_llm_hidden_size, build_native_dit,
                     build_native_patch_encoder, load_component)

AUDIO_GEN_SPAN_TOKEN = "<|audio_gen_span|>"
AUDIO_COMP_SPAN_TOKEN = "<|audio_comp_span|>"
TEXT_COND_END_TOKEN = "<|text_cond_end|>"


def require_token_id(tokenizer, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"tokenizer is missing required special token {token!r}")
    return int(tid)


class _LLMEmbedShim:
    """The engine only calls ``core.llm.get_input_embeddings()``."""

    def __init__(self, embed: nn.Embedding):
        self._embed = embed

    def get_input_embeddings(self) -> nn.Embedding:
        return self._embed


class DotsCore(nn.Module):
    @classmethod
    def from_pretrained(cls, model_dir: str, tokenizer, *, device="cuda",
                        dtype=torch.bfloat16) -> "DotsCore":
        self = cls()
        cfg = load_config(model_dir)
        ckpt = os.path.join(model_dir, "model.safetensors")
        llm_hidden = _llm_hidden_size(model_dir)

        # config scalars (mirror reference DotsTtsCore)
        self.latent_dim = cfg["latent_dim"]
        self.latent_patch_size = cfg["patch_size"]
        self.hidden_patch_size = 1
        self.fm_hidden_size = cfg["DiT"]["hidden_size"]
        self.llm_hidden_size = llm_hidden
        self.xvec_dim = cfg["campplus_embedding_size"]

        # neural nets (built + loaded by the phase 1/2 helpers)
        self.velocity_field_predictor = build_native_dit(model_dir, device=device, dtype=dtype)
        self.patch_encoder = build_native_patch_encoder(model_dir, device=device, dtype=dtype)

        # projections + eos head + token embeddings, built then loaded from model.safetensors
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                self.coordinate_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
                self.hidden_proj = nn.Linear(llm_hidden, self.fm_hidden_size)
                self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
                self.xvec_proj = nn.Sequential(
                    nn.Linear(self.xvec_dim, self.fm_hidden_size),
                    nn.LayerNorm(self.fm_hidden_size))
                self.eos_proj = nn.Sequential(
                    nn.Linear(llm_hidden, llm_hidden), nn.SiLU(), nn.Linear(llm_hidden, 2))
                self.embed_tokens = nn.Embedding(len(tokenizer), llm_hidden)
        finally:
            torch.set_default_dtype(prev)

        load_component(self.coordinate_proj, ckpt, "coordinate_proj.")
        load_component(self.hidden_proj, ckpt, "hidden_proj.")
        load_component(self.latent_proj, ckpt, "latent_proj.")
        load_component(self.xvec_proj, ckpt, "xvec_proj.")
        load_component(self.eos_proj, ckpt, "eos_proj.")
        load_component(self.embed_tokens, ckpt, "llm.model.embed_tokens.")

        self.llm = _LLMEmbedShim(self.embed_tokens)
        self.io_helper = IOHelper(os.path.join(model_dir, "latent_stats.pt"))
        self.tokenizer = tokenizer
        self.audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)
        self.audio_comp_span_id = require_token_id(tokenizer, AUDIO_COMP_SPAN_TOKEN)
        self.text_cond_end_id = require_token_id(tokenizer, TEXT_COND_END_TOKEN)
        self.audio_span_token_ids = [self.audio_gen_span_id, self.audio_comp_span_id]
        return self
