"""Native DotsModel: the per-generation glue the engine calls on ``self.dots``
(FM mask/pos builders, history appends, prefill-embeds, span finders, patch-encoder
capacity, eos stop, the vocoder-stream wrappers, and prompt conditioning). Ported
from the reference DotsTtsModel methods (pure tensor logic), minus the
profiling/compile/logging wrappers. Runs the engine entirely on native modules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange


@dataclass
class _GenerateState:
    llm_cache: Any | None = None
    llm_hiddens: torch.Tensor | None = None
    patch_encoder_state: Any | None = None
    fm_seq_len: int = 0
    fm_capacity: int = 0
    fm_sequence: torch.Tensor | None = None
    fm_cfg_sequence: torch.Tensor | None = None
    fm_null_g_cond: torch.Tensor | None = None
    end_flag: bool = False


@dataclass(frozen=True)
class _PromptConditioning:
    prompt_patches: torch.Tensor | None = None
    prompt_latents: torch.Tensor | None = None
    g_cond: torch.Tensor | None = None


class _ModelConfig:
    def __init__(self, patch_size: int):
        self.patch_size = patch_size


class DotsModel:
    """Holds the native core + vocoder + speaker encoder and the generation glue.
    Not an nn.Module (its sub-models are); the engine reads .core/.vocoder/.hop_size
    and calls the private helpers below."""

    def __init__(self, core, vocoder, xvector_extractor, *, patch_size: int):
        self.core = core
        self.vocoder = vocoder
        self.xvector_extractor = xvector_extractor
        self.hop_size = vocoder.hop_size
        self.config = _ModelConfig(patch_size)
        self._optimize_enabled = False

    # ---------------------------------------------------------------- FM mask/pos
    def _build_fm_attn_mask(self, *, state: _GenerateState, attn_mask: torch.Tensor):
        if state.fm_seq_len <= 0:
            raise RuntimeError("FM sequence length must be positive before decode.")
        hidden_patch_size = self.core.hidden_patch_size
        latent_start = attn_mask.size(-1) - self.core.latent_patch_size
        attn_mask.zero_()
        block_start = state.fm_seq_len - hidden_patch_size
        if block_start > 0:
            causal_mask = torch.ones((block_start, block_start), device=attn_mask.device,
                                     dtype=torch.bool).triu(1).logical_not()
            attn_mask[:, :block_start, :block_start] = causal_mask
        attn_mask[:, block_start:state.fm_seq_len, :state.fm_seq_len] = True
        attn_mask[:, block_start:state.fm_seq_len, latent_start:] = True
        attn_mask[:, latent_start:, :state.fm_seq_len] = True
        attn_mask[:, latent_start:, latent_start:] = True
        if latent_start > state.fm_seq_len:
            padding_indices = torch.arange(state.fm_seq_len, latent_start, device=attn_mask.device)
            attn_mask[:, padding_indices, padding_indices] = True
        return attn_mask

    def _build_fm_pos_ids(self, *, state: _GenerateState, pos_ids: torch.Tensor):
        if state.fm_seq_len <= 0:
            raise RuntimeError("FM sequence length must be positive before decode.")
        pos_ids.zero_()
        latent_start = pos_ids.size(-1) - self.core.latent_patch_size
        pos_ids[:, :state.fm_seq_len] = torch.arange(
            state.fm_seq_len, device=pos_ids.device, dtype=pos_ids.dtype)
        pos_ids[:, latent_start:] = torch.arange(
            state.fm_seq_len, state.fm_seq_len + self.core.latent_patch_size,
            device=pos_ids.device, dtype=pos_ids.dtype)
        return pos_ids

    # ---------------------------------------------------------------- FM history
    def _append_to_fm_buffer(self, buffer, state: _GenerateState, chunk):
        if buffer is None:
            raise RuntimeError("FM static buffer is not initialized.")
        start = state.fm_seq_len
        end = start + chunk.size(1)
        if end > state.fm_capacity:
            raise RuntimeError("FM StaticBuffer capacity exceeded: "
                               f"next_length={end} capacity={state.fm_capacity}.")
        buffer[:, start:end].copy_(chunk.to(buffer.dtype))
        return start, end

    def _append_hidden_chunk(self, state: _GenerateState, hidden_chunk):
        last_hidden = hidden_chunk[:, -self.core.hidden_patch_size:, :]
        projected = self.core.hidden_proj(last_hidden)
        null_projected = self.core.hidden_proj(torch.zeros_like(last_hidden))
        _start, end = self._append_to_fm_buffer(state.fm_sequence, state, projected)
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len:end].copy_(null_projected.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    def _append_history_chunk(self, state: _GenerateState, latent_chunk):
        history_latent = self.core.latent_proj(latent_chunk)
        _start, end = self._append_to_fm_buffer(state.fm_sequence, state, history_latent)
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len:end].copy_(history_latent.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    # ---------------------------------------------------------------- spans / prefill
    @staticmethod
    def _find_audio_span_positions(generation_schedule, *, audio_placeholder_ids):
        schedule = generation_schedule[0]
        placeholder_ids = torch.tensor(sorted(audio_placeholder_ids),
                                       device=schedule.device, dtype=schedule.dtype)
        return torch.nonzero(torch.isin(schedule, placeholder_ids), as_tuple=False).squeeze(-1)

    @staticmethod
    def _next_token_is_audio_span(generation_schedule, *, position, audio_placeholder_ids):
        next_position = position + 1
        if next_position >= generation_schedule.size(1):
            return False
        return int(generation_schedule[0, next_position].item()) in audio_placeholder_ids

    def _locate_prefill_boundary(self, *, span_positions, prompt_patch_count):
        if span_positions.numel() > prompt_patch_count:
            return int(span_positions[prompt_patch_count].item()), span_positions[:prompt_patch_count]
        raise RuntimeError("Prefill boundary discovery failed despite prior schedule validation.")

    def _build_prefill_inputs_embeds(self, generation_schedule, *,
                                     prompt_patch_embeddings, prompt_span_positions):
        inputs_embeds = self.core.llm.get_input_embeddings()(generation_schedule).clone()
        if prompt_span_positions.numel() > 0:
            if prompt_patch_embeddings is None:
                raise RuntimeError(
                    "Prompt patch embeddings are required when prefill includes prompt audio spans.")
            patch_embeddings = prompt_patch_embeddings[:, :prompt_span_positions.numel()].to(
                inputs_embeds.dtype)
            if patch_embeddings.size(1) != prompt_span_positions.numel():
                raise RuntimeError(
                    f"Prompt patch embeddings ({patch_embeddings.size(1)}) do not match prompt "
                    f"span count ({prompt_span_positions.numel()}).")
            inputs_embeds[:, prompt_span_positions, :] = patch_embeddings
        return inputs_embeds

    # ---------------------------------------------------------------- patch-encoder capacity
    def _resolve_state_audio_patch_count(self, max_audio_patch_count: int) -> int:
        requested = int(max_audio_patch_count)
        if requested <= 0:
            raise ValueError("max_audio_patch_count must be positive.")
        return requested  # _optimize_enabled is False -> no bucketing

    def _resolve_patch_encoder_audio_bucket(self, required_seq_len: int) -> int:
        requested = int(required_seq_len)
        if requested <= 0:
            raise ValueError("required_seq_len must be positive.")
        return math.ceil(requested / self.core.patch_encoder.out_ds_rate)

    def _copy_patch_encoder_state(self, source, target) -> None:
        seq_len = source.seq_len
        target_capacity = int(target.layer_caches[0][0].size(2))
        if seq_len > target_capacity:
            raise ValueError("Patch encoder state copy exceeds target capacity: "
                             f"seq_len={seq_len} capacity={target_capacity}.")
        target.conv_tail.copy_(source.conv_tail)
        target.seq_len = seq_len
        for (sk, sv), (tk, tv) in zip(source.layer_caches, target.layer_caches, strict=True):
            if seq_len > 0:
                tk[:, :, :seq_len, :].copy_(sk[:, :, :seq_len, :])
                tv[:, :, :seq_len, :].copy_(sv[:, :, :seq_len, :])

    def _ensure_patch_encoder_state_capacity(self, state: _GenerateState, *,
                                             required_seq_len, device, dtype) -> None:
        current_state = state.patch_encoder_state
        if current_state is not None:
            if int(current_state.layer_caches[0][0].size(2)) >= required_seq_len:
                return
        target = self._resolve_patch_encoder_audio_bucket(required_seq_len)
        next_state = self.core.patch_encoder.init_decode_state(
            max_audio_patch_count=target, batch_size=1, device=device, dtype=dtype)
        if current_state is not None:
            self._copy_patch_encoder_state(current_state, next_state)
        state.patch_encoder_state = next_state

    # ---------------------------------------------------------------- eos
    def _should_stop_after_current_audio(self, state: _GenerateState, *, eos_threshold) -> bool:
        if state.llm_hiddens is None:
            return False
        eos = self.core.eos_proj(state.llm_hiddens).softmax(dim=-1)[:, -1, 1] > eos_threshold
        return state.end_flag or bool(eos.item())

    # ---------------------------------------------------------------- vocoder glue
    @torch.no_grad()
    def _decode_latents(self, latents):
        return self.vocoder.inference_from_latents(latents.transpose(1, 2).float(), do_sample=False)

    @torch.no_grad()
    def _init_vocoder_stream_state(self):
        return self.vocoder.init_stream_state(batch_size=1, chunk_size=self.core.latent_patch_size)

    @torch.no_grad()
    def _stream_vocoder_patch(self, latent_patch, *, stream_state):
        return self.vocoder.stream_step(latent_patch.transpose(1, 2), stream_state)

    @torch.no_grad()
    def _flush_vocoder_stream(self, stream_state):
        return self.vocoder.stream_flush(stream_state)

    # ---------------------------------------------------------------- prompt conditioning
    @torch.no_grad()
    def _prepare_prompt_conditioning(self, prompt_audio, *, use_prompt_prefill,
                                     speaker_scale: float = 1.5) -> _PromptConditioning:
        if prompt_audio is None:
            return _PromptConditioning()
        self.vocoder.eval()
        self.xvector_extractor.eval()
        device = next(self.core.parameters()).device
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        prompt_audio = prompt_audio.to(device=device)

        target_len = math.ceil(
            prompt_audio.size(1) / (self.config.patch_size * self.hop_size)
        ) * (self.config.patch_size * self.hop_size)
        pad_len = target_len - prompt_audio.size(1)
        if pad_len > 0:
            prompt_audio = F.pad(prompt_audio, (0, pad_len))

        speaker_embedding = self.xvector_extractor(prompt_audio[None, :]) * float(speaker_scale)
        g_cond = self.core.xvec_proj(speaker_embedding)
        if not use_prompt_prefill:
            return _PromptConditioning(g_cond=g_cond)

        prompt_latents = self.vocoder.extract_latents(prompt_audio[None, :])
        prompt_latents_sampled = self.core.io_helper.sample_from_latent(prompt_latents)
        prompt_latents_sampled = prompt_latents_sampled[:, : -self.config.patch_size]
        prompt_patches = rearrange(
            self.core.io_helper.normalize(prompt_latents_sampled),
            "b (s p) d -> b s p d", p=self.config.patch_size)
        return _PromptConditioning(prompt_patches=prompt_patches,
                                   prompt_latents=prompt_latents_sampled, g_cond=g_cond)
