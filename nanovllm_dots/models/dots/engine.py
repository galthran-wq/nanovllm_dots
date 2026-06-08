"""Phase-1 continuous-batching engine for dots.tts.

Drives the LLM step for many requests at once through the paged-KV engine
(`BatchedPagedLLM` + the engine `Scheduler`/`BlockManager`), while the heavy
flow-matching head, patch encoder and vocoder run *per request* from the
original dots modules (exact, for correctness). Phase 2 replaces the serial
per-request FM with a single batched FM call -- the loop shape here is the one
it fills in.

Structure (lockstep): each engine step runs ONE batched LLM call across the
active sequences, then each sequence's FM ODE serially. dots emits exactly one
LLM token per audio patch, so this maps onto the standard prefill-then-decode
scheduler: the prefill step produces patch 0, each decode step produces the
next patch.

Per-patch ordering mirrors the reference `DotsTtsModel._decode` exactly, just
with the LLM hoisted out and batched:

    prefill step:  LLM(text prefix) -> hidden_p
                   append_hidden(hidden_p)            # seed FM history
                   eos? ; FM -> patch0 ; history+patch_encoder -> embed0
    decode step k: LLM(embed_{k-1})  -> hidden_k       # the deferred consume LLM
                   append_hidden(hidden_k)
                   eos? ; FM -> patch_k ; history+patch_encoder -> embed_k

State isolation: each sequence gets its OWN FM buffers. The reference
`_allocate_generate_state` hands out a *shared* cached workspace (fine for one
request, aliasing for many), so we build the `_GenerateState` with private
tensors here.

Phase-1 scope: no prompt-audio conditioning (g_cond=None); eager; latents are
collected per request (streaming vocoder is Phase 3). Reuses a loaded
`DotsTtsRuntime` for tokenization / schedule building / the non-LLM modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from nanovllm_dots.config import Config
from nanovllm_dots.engine.scheduler import Scheduler
from nanovllm_dots.engine.sequence import Sequence
from nanovllm_dots.models.dots.batched_llm import BatchedPagedLLM


@dataclass
class DotsSeqState:
    """Per-request payload carried on `Sequence.custom_payload`."""
    schedule: torch.Tensor            # [1, S] generation schedule
    prefill_end: int
    span_count: int
    g_cond: torch.Tensor | None
    gen_state: Any                    # dots `_GenerateState` (private FM buffers)
    prefill_embed: torch.Tensor       # [prefill_end, H] LLM input for prefill
    next_embed: torch.Tensor | None = None   # [1, H] LLM input for the next decode step
    position: int = 0                 # cursor into `schedule`
    patches_emitted: int = 0
    done_prefill: bool = False
    latents: list[torch.Tensor] = field(default_factory=list)
    # decode controls
    num_steps: int = 10
    guidance_scale: float = 1.2
    ode_method: str = "euler"
    eos_threshold: float = 0.8


class DotsBatchEngine:
    def __init__(
        self,
        runtime,                      # DotsTtsRuntime (optimize=False)
        paged_llm,                    # QwenLLM with weights loaded
        *,
        model_dir: str,
        num_kvcache_blocks: int = 128,
        block_size: int = 256,
        max_num_seqs: int = 32,
        max_model_len: int = 4096,
        compile_fm: bool = True,
        fm_vfp=None,
        fm_len_bucket: int = 0,
        fm_compile_mode: str = "default",
        fm_accel: str | None = None,   # "compile" | "cudagraph" | "none"
    ) -> None:
        self.runtime = runtime
        self.dots = runtime.model                 # DotsTtsModel: core/patch_encoder/FM/vocoder
        self.core = self.dots.core
        self.device = next(self.core.parameters()).device
        self.dtype = torch.bfloat16
        self.block_size = block_size

        # FM is overhead-bound at single-stream (<1% MFU: tiny matrices, eager
        # kernel launches), so compile the DiT forward (inductor fusion -> fewer,
        # bigger kernels). History length is bucketed (mask/pos already pad
        # correctly) so a bounded set of shapes is captured. `fm_vfp` lets a
        # caller share ONE compiled wrapper across engines (stable graph cache).
        # NOTE: mode="reduce-overhead" (CUDA graphs) corrupts results here -- the
        # ODE solver feeds each vfp output back as the next input, and the graph
        # reuses output memory. Use "default" until outputs are cloned/captured
        # safely (separate step).
        if fm_accel is None:
            fm_accel = "compile" if compile_fm else "none"
        self.fm_accel = fm_accel
        self.fm_len_bucket = fm_len_bucket
        if fm_vfp is not None:
            self._vfp = fm_vfp
        elif fm_accel == "cudagraph":
            # Manual CUDA-graph capture of the EAGER DiT (faithful to eager: the
            # graph just removes launch overhead). Capture-safe first (the timestep
            # embedder's torch.arange().to(device) aborts capture). Faithfulness
            # is exact -- it reproduces eager+bucketed FM. (The cos ~0.73 vs the
            # unbucketed golden is the PADDING divergence, present in eager too;
            # set fm_len_bucket=0 for the exact-but-graph-per-patch variant.)
            from nanovllm_dots.models.dots.cudagraph_dit import (
                CudaGraphRunner, make_dit_capture_safe,
            )
            # Default UNBUCKETED (fm_len_bucket=0): the FM history grows by a fixed
            # stride per patch, so only ~one unique shape per patch-count recurs --
            # a bounded set of graphs, captured once and reused EXACTLY (cos 1.0 vs
            # eager) with no padding. Faster AND exact vs bucketed (which pads ->
            # cos ~0.73 divergence). Bucketing stays available for memory-bound use.
            make_dit_capture_safe(self.core.velocity_field_predictor, self.device)
            self._vfp = CudaGraphRunner(self.core.velocity_field_predictor)
        elif fm_accel == "compile":
            # dynamic=True => one graph handles every history length (no recompile,
            # no length padding). Bucketing is only needed for CUDA-graph capture.
            self._vfp = torch.compile(
                self.core.velocity_field_predictor, mode=fm_compile_mode,
                dynamic=(fm_len_bucket == 0),
            )
        else:
            self._vfp = self.core.velocity_field_predictor

        self.batched_llm = BatchedPagedLLM(
            paged_llm, num_blocks=num_kvcache_blocks, block_size=block_size,
            device=self.device, dtype=self.dtype,
        )
        cfg = Config(
            model=model_dir,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max(max_model_len, 16384),
            kvcache_block_size=block_size,
            num_kvcache_blocks=num_kvcache_blocks,
        )
        self.scheduler = Scheduler(cfg, callbacks=None)
        self._audio_ids = set(self.core.audio_span_token_ids)
        self._embed = self.core.llm.get_input_embeddings()
        self._results: dict[str, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def warmup_graphs(self, concurrencies=(1,), max_fm_len: int = 512) -> None:
        """Pre-capture the FM CUDA graphs for every (batch, length) shape, with
        dummy inputs, BEFORE serving. Capturing a new bucket mid-generation (lazy)
        corrupts that run; pre-warming in isolation makes the first real request
        correct. No-op unless fm_accel == "cudagraph"."""
        if self.fm_accel != "cudagraph":
            return
        import types

        core = self.core
        H, lp, b = core.fm_hidden_size, core.latent_patch_size, self.fm_len_bucket
        dev, dt = self.device, self.dtype
        buckets = list(range(b, max_fm_len + b, b))
        with torch.autocast(device_type=dev.type, dtype=dt):
            for nconc in concurrencies:
                bs = 2 * nconc  # CFG doubles the batch
                for L in buckets:
                    total = L + lp
                    # Capture with REALISTIC inputs (random x, the real structured
                    # FM mask/pos), not zeros: an all-zero x degenerates SDPA at
                    # capture and the replayed graph stays wrong. Mask/pos values
                    # are still copied per call; only the captured kernels matter.
                    st = types.SimpleNamespace(fm_seq_len=L)
                    mask = torch.zeros(1, total, total, dtype=torch.bool, device=dev)
                    pos = torch.zeros(1, total, dtype=torch.float32, device=dev)
                    self.dots._build_fm_attn_mask(state=st, attn_mask=mask)
                    self.dots._build_fm_pos_ids(state=st, pos_ids=pos)
                    self._vfp(
                        x=torch.randn(bs, total, H, device=dev, dtype=dt),
                        timesteps=torch.rand(bs, device=dev, dtype=dt),
                        attn_mask=mask.expand(bs, -1, -1).contiguous(),
                        pos_ids=pos.expand(bs, -1).contiguous(),
                        g_cond=torch.zeros(bs, H, device=dev, dtype=dt),
                    )
    def _alloc_private_state(self, span_count: int):
        """dots `_GenerateState` with PRIVATE FM buffers (not the shared cache)."""
        from dots_tts.models.dots_tts.model import _GenerateState

        core = self.core
        patch_count = self.dots._resolve_state_audio_patch_count(span_count)
        fm_capacity = patch_count * (core.hidden_patch_size + core.latent_patch_size)
        z = lambda *s: torch.zeros(*s, dtype=self.dtype, device=self.device)
        patch_encoder_state = core.patch_encoder.init_decode_state(
            max_audio_patch_count=patch_count, batch_size=1,
            device=self.device, dtype=self.dtype,
        )
        return _GenerateState(
            patch_encoder_state=patch_encoder_state,
            fm_seq_len=0,
            fm_capacity=fm_capacity,
            fm_sequence=z(1, fm_capacity, core.fm_hidden_size),
            fm_cfg_sequence=z(1, fm_capacity, core.fm_hidden_size),
            fm_null_g_cond=z(1, core.fm_hidden_size),
        )

    def add_request(
        self,
        seq_id: str,
        text: str,
        *,
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        eos_threshold: float = 0.8,
    ) -> None:
        inputs = self.runtime._prepare_inputs(
            text=text, prompt_audio_path=None, prompt_text=None,
            template_name=None, language=None, normalize_text=False,
        )
        schedule = inputs["generation_schedule"].to(self.device)
        if schedule.size(0) != 1:
            raise ValueError("expected batch-1 generation_schedule")

        span_positions = self.dots._find_audio_span_positions(
            schedule, audio_placeholder_ids=self._audio_ids
        )
        span_count = int(span_positions.numel())
        if span_count < 1:
            raise ValueError("schedule has no audio spans to generate")
        # Phase-1 scope: only the default contiguous-audio template (a single run
        # of spans after the prefix). Interleaved templates weave text tokens
        # between spans and need _consume_text_schedule handling we don't do yet.
        if span_count > 1 and int(span_positions[-1] - span_positions[0]) != span_count - 1:
            raise NotImplementedError(
                "DotsBatchEngine supports only contiguous-audio (non-interleave) "
                "templates in Phase 1; got interleaved audio spans."
            )
        prefill_end = int(span_positions[0].item())  # no prompt audio => first span

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            prefill_embed = self.dots._build_prefill_inputs_embeds(
                schedule[:, :prefill_end],
                prompt_patch_embeddings=None,
                prompt_span_positions=span_positions[:0],
            )[0]  # [prefill_end, H]

        state = self._alloc_private_state(span_count)
        payload = DotsSeqState(
            schedule=schedule,
            prefill_end=prefill_end,
            span_count=span_count,
            g_cond=None,
            gen_state=state,
            prefill_embed=prefill_embed,
            position=prefill_end,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            eos_threshold=eos_threshold,
        )
        # KV hash tokens = the actual prefill token ids (enables prefix caching
        # across requests with identical text prefixes).
        token_ids = schedule[0, :prefill_end].tolist()
        seq = Sequence(seq_id, token_ids, self.block_size, payload)
        self._results[seq_id] = payload.latents
        self.scheduler.add(seq)

    # ------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self) -> list[Sequence]:
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return seqs

        # 1) batched LLM over the active sequences' current chunk
        chunks, block_tables, cached_lens = [], [], []
        for seq in seqs:
            st: DotsSeqState = seq.custom_payload
            if is_prefill:
                # Preemption would deallocate KV and re-enter prefill mid-stream;
                # we have no re-prefill path, so fail loud rather than corrupt.
                if st.done_prefill:
                    raise RuntimeError(
                        f"seq {seq.seq_id} re-entered prefill (preemption is not "
                        "supported in Phase 1; size num_kvcache_blocks to avoid it)."
                    )
                chunk = st.prefill_embed[seq.num_cached_tokens :]
                if chunk.size(0) == 0:
                    raise RuntimeError(
                        f"seq {seq.seq_id} prefix is fully KV-cached "
                        "(unsupported in Phase 1: there is no token left to seed patch 0)."
                    )
                st.done_prefill = True
            else:
                if st.next_embed is None:
                    raise RuntimeError(
                        f"seq {seq.seq_id} reached decode without a prefill hidden "
                        "(next_embed is None)."
                    )
                chunk = st.next_embed  # [1, H]
            chunks.append(chunk)
            block_tables.append(seq.block_table)
            cached_lens.append(seq.num_cached_tokens if is_prefill else len(seq) - 1)
        dots = self.dots
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            hiddens = self.batched_llm.append_batch(chunks, block_tables, cached_lens)

            # 2) pre-FM (per seq, cheap): seed/append the hidden chunk, eos check.
            stops = []
            for seq, hidden in zip(seqs, hiddens):
                state = seq.custom_payload.gen_state
                state.llm_hiddens = hidden[-1:].unsqueeze(0)  # [1, 1, H]
                dots._append_hidden_chunk(state, state.llm_hiddens)
                stops.append(
                    dots._should_stop_after_current_audio(
                        state, eos_threshold=seq.custom_payload.eos_threshold
                    )
                )

            # 3) batched FM ODE over ALL active sequences at once (the 81% hog).
            latents = self._batched_fm([s.custom_payload for s in seqs])

            # 4) post-FM (per seq): history append + patch encoder + emit + stop.
            for seq, latent, stop in zip(seqs, latents, stops):
                self._finish_patch(seq, seq.custom_payload, latent.unsqueeze(0), stop)

        for seq in seqs:
            if seq.stoped:
                self.scheduler.finish(seq)
        return seqs

    def _batched_fm(self, payloads: list[DotsSeqState]) -> torch.Tensor:
        """Pad every active sequence's FM history into one batch and integrate
        the CFG ODE together. Returns [N, patch_size, latent_dim] (normalized)."""
        from nanovllm_dots.models.dots.batched_fm import batched_flow_matching

        core, dots = self.core, self.dots
        states = [p.gen_state for p in payloads]
        n = len(states)
        max_len = max(st.fm_seq_len for st in states)
        # Optional length bucketing (for CUDA-graph shape stability); adds padding
        # and a bit of numerical drift, so off by default (dynamic compile instead).
        if self.fm_len_bucket > 0:
            b = self.fm_len_bucket
            max_len = ((max_len + b - 1) // b) * b
        # One odeint call integrates the whole batch, so the ODE schedule and CFG
        # must be uniform across co-active requests. Fail loud rather than silently
        # decode some requests with the wrong NFE/guidance.
        p0 = payloads[0]
        for p in payloads:
            if (p.num_steps, p.guidance_scale, p.ode_method) != (
                p0.num_steps, p0.guidance_scale, p0.ode_method
            ):
                raise RuntimeError(
                    "batched FM requires uniform (num_steps, guidance_scale, ode_method) "
                    "across co-active requests; got mismatched decode params."
                )
        total_len = max_len + core.latent_patch_size
        H = core.fm_hidden_size
        dev, dt = self.device, self.dtype

        inp = torch.zeros(n, total_len, H, device=dev, dtype=dt)
        cfg = torch.zeros(n, total_len, H, device=dev, dtype=dt)
        mask = torch.zeros(n, total_len, total_len, dtype=torch.bool, device=dev)
        pos = torch.zeros(n, total_len, dtype=torch.float32, device=dev)
        gcond = torch.zeros(n, H, device=dev, dtype=dt)
        for i, (st, p) in enumerate(zip(states, payloads)):
            L = st.fm_seq_len
            inp[i, :L] = st.fm_sequence[0, :L]
            cfg[i, :L] = st.fm_cfg_sequence[0, :L]
            # _build_fm_* fill a [1, total_len, ...] view in place (zero + structure);
            # padding gap [L:max_len] is self-masked, positions stay per-seq correct.
            dots._build_fm_attn_mask(state=st, attn_mask=mask[i : i + 1])
            dots._build_fm_pos_ids(state=st, pos_ids=pos[i : i + 1])
            if p.g_cond is not None:
                gcond[i] = p.g_cond.to(dev, dt)
            elif st.fm_null_g_cond is not None:
                gcond[i] = st.fm_null_g_cond[0]

        return batched_flow_matching(
            core, input_sequence=inp, cfg_sequence=cfg, attn_mask=mask, pos_ids=pos,
            g_cond=gcond, num_steps=p0.num_steps, guidance_scale=p0.guidance_scale,
            ode_method=p0.ode_method, vfp=self._vfp,
        )

    def _finish_patch(self, seq: Sequence, st: DotsSeqState, patch: torch.Tensor, stop: bool) -> None:
        core = self.core
        # next LLM input = patch encoder over this patch (history append + encode)
        st.next_embed = self._patch_to_embed(st.gen_state, patch)

        latent_cpu = core.io_helper.denormalize(patch).detach().float().cpu()
        st.latents.append(latent_cpu)
        st.patches_emitted += 1
        st.position += 1
        # advance the KV sequence by one token (this patch's encoder embedding,
        # consumed by the LLM on the next decode step)
        seq.append_token(latent_cpu.reshape(-1).numpy().tobytes())

        schedule_exhausted = st.position >= st.schedule.size(1)
        if stop or schedule_exhausted or st.patches_emitted >= st.span_count:
            seq.stoped = True

    def _patch_to_embed(self, state, patch: torch.Tensor) -> torch.Tensor:
        """`_consume_audio_patch` minus the LLM step (the LLM is batched)."""
        dots, core = self.dots, self.core
        dots._append_history_chunk(state, patch)
        cur = 0 if state.patch_encoder_state is None else state.patch_encoder_state.seq_len
        dots._ensure_patch_encoder_state_capacity(
            state,
            required_seq_len=cur + core.patch_encoder.out_ds_rate,
            device=self.device, dtype=self.dtype,
        )
        patch_for_llm = core.io_helper.denormalize(patch)
        positions = torch.arange(
            core.patch_encoder.out_ds_rate, device=self.device, dtype=torch.long
        ) + state.patch_encoder_state.seq_len
        embed, conv_tail = core.patch_encoder.decode_patch(
            patch_for_llm,
            state.patch_encoder_state.conv_tail,
            state.patch_encoder_state.layer_caches,
            positions,
        )
        state.patch_encoder_state.conv_tail.copy_(conv_tail)
        state.patch_encoder_state.seq_len += core.patch_encoder.out_ds_rate
        return embed[0]  # [1, H]

    # ------------------------------------------------------------------- run
    @torch.no_grad()
    def run_all(self) -> dict[str, torch.Tensor]:
        """Drive the scheduler to completion; return per-request latent tensors."""
        while not self.scheduler.is_finished():
            seqs = self.step()
            if not seqs:
                raise RuntimeError(
                    "scheduler made no progress with pending requests "
                    "(KV exhaustion / preemption livelock)."
                )
        return {
            sid: (torch.cat(lat, dim=1) if lat else torch.zeros((1, 0, 0)))
            for sid, lat in self._results.items()
        }
