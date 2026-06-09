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
    max_patches: int | None = None    # hard cap on emitted patches (benchmarking)
    fm_head: Any = None               # per-seq CachedFMHead (fm_accel="kvcache")
    pe_row: int | None = None         # cache row in the shared FlashPatchEncoder
    vocoder_state: Any = None         # per-seq BigVGAN streaming state (lazy)
    vocoded: int = 0                  # # of emitted patches already vocoded
    stream: bool = True               # True: per-patch streaming vocode; False:
                                      # one-shot vocode the full latents at finish


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
        fm_accel: str | None = None,   # "compile"|"cudagraph"|"kvcache"|"hybrid"|"none"
        kvcache_graphed: bool = True,  # CUDA-graph the kvcache FM head
        hybrid_threshold: int = 400,   # fm_seq_len to switch cudagraph-full -> kvcache
                                       # (~per-patch crossover; clean bench: kvcache
                                       # already wins by L~250/16s, cudagraph by L~105/3.5s)
        graph_decode: bool = False,    # CUDA-graph the one-token LLM decode forward
        compile_pe: bool = False,      # torch.compile the patch_encoder decode_patch
        flash_pe: bool = False,        # flash/varlen BATCHED patch_encoder decode
        pe_max_patches: int = 256,     # shared FlashPatchEncoder history capacity
        pe_max_batch: int | None = None,  # flash_pe cache rows (default max_num_seqs);
                                          # must bound CONCURRENTLY generating seqs
    ) -> None:
        self.kvcache_graphed = kvcache_graphed
        self.hybrid_threshold = hybrid_threshold
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
        if fm_accel == "hybrid":
            # Length-adaptive: cudagraph-full DiT for short history, KV cache once it
            # exceeds hybrid_threshold. Needs BOTH a CudaGraphRunner (short path,
            # shareable across engines via fm_vfp so it isn't re-captured) and the
            # eager DiT (the kvcache heads capture their own layer graphs).
            from nanovllm_dots.models.dots.cudagraph_dit import (
                CudaGraphRunner, make_dit_capture_safe,
            )
            make_dit_capture_safe(self.core.velocity_field_predictor, self.device)
            self._vfp_full = fm_vfp if fm_vfp is not None else CudaGraphRunner(
                self.core.velocity_field_predictor)
            self._vfp = self.core.velocity_field_predictor
        elif fm_vfp is not None and fm_accel != "mfgraph":
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
        elif fm_accel == "kvcache":
            # Per-timestep prefix KV cache: O(P) instead of O(P^2). FM runs through
            # per-seq CachedFMHead objects (built lazily in _kvcache_fm) that capture
            # their own graphs of the eager DiT layers. self._vfp is that eager DiT.
            self._vfp = self.core.velocity_field_predictor
        elif fm_accel == "mfgraph":
            # MeanFlow whole-patch graph: capture the entire nfe-step solver into
            # one graph (built lazily in _mf_graph once the nfe is known). The DiT
            # must be capture-safe (arange().to() in the embedders aborts capture).
            from nanovllm_dots.models.dots.cudagraph_dit import make_dit_capture_safe
            make_dit_capture_safe(self.core.velocity_field_predictor, self.device)
            self._vfp = self.core.velocity_field_predictor
            # A shared GraphedMeanflow (via fm_vfp) POOLS the whole-patch graphs
            # across engines/requests -- essential, since each history length is
            # captured once and the capture is ~4x a single-DiT graph; reuse only
            # pays off across requests of recurring lengths. None -> built lazily.
            self._mfgraph = fm_vfp
        else:
            self._vfp = self.core.velocity_field_predictor

        # patch_encoder.decode_patch has fully STATIC shapes (it attends over the
        # fixed cache_capacity with a positions-derived mask), so a single
        # torch.compile specialization (dynamic=False) fuses its many tiny eager
        # kernels -- ~16 -> ~8 ms/patch. Compiled once on the shared core.
        if compile_pe:
            pe = self.core.patch_encoder
            if not getattr(pe, "_decode_patch_compiled", False):
                pe.decode_patch = torch.compile(pe.decode_patch, dynamic=False)
                pe._decode_patch_compiled = True

        # flash/varlen BATCHED patch_encoder: replaces the per-seq dense-mask SDPA
        # over the FIXED cache_capacity (O(capacity)/patch, run once per request ->
        # LINEAR in concurrency, the throughput wall) with ONE flash_attn_with_kvcache
        # call over ALL active requests, each attending only its own actual history
        # (cache_batch_idx -> a persistent cache row per request). Quality-free
        # (faithful to decode_patch, cos 0.99999). See flash_patch_encoder.py.
        self.flash_pe = flash_pe
        self._pe_max_patches = pe_max_patches
        if flash_pe:
            from nanovllm_dots.models.dots.flash_patch_encoder import FlashPatchEncoder
            pe = self.core.patch_encoder
            # Rows are claimed for a request's whole generation (not per step), so the
            # pool must cover every CONCURRENTLY generating sequence. The decode batch
            # is capped at max_num_seqs, but the scheduler's running set can exceed it
            # when many requests are queued (prefill admits up to max_num_seqs/call and
            # returns early). Size to pe_max_batch (>= the true running cap) and fail
            # loudly rather than corrupt memory; a server should set max_num_seqs to
            # bound concurrency and pe_max_batch to match.
            n_rows = pe_max_batch if pe_max_batch is not None else max_num_seqs
            self._pe_max_batch = n_rows
            self._flash_pe = FlashPatchEncoder(
                pe, max_batch=n_rows,
                max_seq_len=pe_max_patches * pe.out_ds_rate,
                device=self.device, dtype=self.dtype,
            )
            self._pe_free = list(range(n_rows))

        self.batched_llm = BatchedPagedLLM(
            paged_llm, num_blocks=num_kvcache_blocks, block_size=block_size,
            device=self.device, dtype=self.dtype, graph_decode=graph_decode,
        )
        if graph_decode:
            # Capture now, while the paged cache holds no live sequence (the
            # decode graph writes K/V into it at the captured slots). Covers
            # every co-active batch size the scheduler can form. The block-table
            # width is sized to the longest sequence (ceil(max_model_len/block))
            # so the graph -- not the eager fallback -- serves every real length.
            bt_width = -(-max_model_len // block_size)
            self.batched_llm.warmup_decode(
                batch_sizes=tuple(range(1, max_num_seqs + 1)), bt_width=bt_width)
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
        # flash_pe shares a fixed-capacity history cache (cap = pe_max_patches *
        # out_ds_rate). Bound each request's emitted patches here (host-side, no
        # per-step GPU sync) so flash_attn_with_kvcache can never write past the
        # cache. The stop rule caps emitted patches at <= patch_count, so this is
        # sufficient.
        if self.flash_pe and patch_count > self._pe_max_patches:
            raise RuntimeError(
                f"flash_pe: request needs {patch_count} patches but pe_max_patches="
                f"{self._pe_max_patches}. Increase pe_max_patches.")
        fm_capacity = patch_count * (core.hidden_patch_size + core.latent_patch_size)
        z = lambda *s: torch.zeros(*s, dtype=self.dtype, device=self.device)
        # flash_pe owns a shared patch_encoder cache (FlashPatchEncoder); the
        # per-seq reference cache (~100MB/seq at 256 patches) is then unused.
        patch_encoder_state = None if self.flash_pe else \
            core.patch_encoder.init_decode_state(
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
        max_patches: int | None = None,
        stream: bool = True,
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
            max_patches=max_patches,
            stream=stream,
        )
        # KV hash tokens = the actual prefill token ids (enables prefix caching
        # across requests with identical text prefixes).
        token_ids = schedule[0, :prefill_end].tolist()
        seq = Sequence(seq_id, token_ids, self.block_size, payload)
        self._results[seq_id] = payload.latents
        self.scheduler.add(seq)

    def cancel(self, seq_id: str) -> bool:
        """Abort an in-flight request: free its KV blocks and flash_pe cache row and
        drop it from the scheduler. Called (via the server's intake) when a client
        disconnects, so a zombie sequence can't hold a batch slot forever. Must run
        on the engine-owning thread (between steps), like every other engine call."""
        seq = self.scheduler._id_to_seq.get(seq_id)
        if seq is None:
            return False
        p = seq.custom_payload
        if self.flash_pe and p is not None and p.pe_row is not None:
            self._pe_free.append(p.pe_row)
            p.pe_row = None
        self.scheduler.cancel(seq_id)
        return True

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

            # 4) post-FM: history append + patch encoder + emit + stop. The patch
            # encoder is BATCHED over all active seqs in one flash call (flash_pe)
            # or run per-seq (reference dense-mask path).
            if self.flash_pe:
                self._finish_patches_flash(seqs, latents, stops)
            else:
                for seq, latent, stop in zip(seqs, latents, stops):
                    self._finish_patch(seq, seq.custom_payload, latent.unsqueeze(0), stop)

        for seq in seqs:
            if seq.stoped:
                p = seq.custom_payload
                if self.flash_pe and p.pe_row is not None:
                    self._pe_free.append(p.pe_row)   # recycle the cache row
                    p.pe_row = None
                self.scheduler.finish(seq)
        return seqs

    def _kvcache_fm(self, payloads: list[DotsSeqState], *, prime: bool = False) -> torch.Tensor:
        """Per-timestep KV-cache FM. Each request carries its own CachedFMHead
        (own cond/uncond prefix cache that grows with its history), so per-patch
        work is O(1) in history length. `prime=True` (hybrid) bulk-fills a head's
        cache to the current length on first use (switching in mid-generation).
        Returns [N, patch_size, latent_dim]."""
        from nanovllm_dots.models.dots.flash_cached_fm import (
            FlashCachedFMHead, GraphedFlashCachedFMHead,
        )
        HeadCls = GraphedFlashCachedFMHead if self.kvcache_graphed else FlashCachedFMHead

        core = self.core
        lp, ld = core.latent_patch_size, core.latent_dim
        stride = core.hidden_patch_size + core.latent_patch_size
        outs = []
        for p in payloads:
            st = p.gen_state
            if p.ode_method != "euler":
                raise RuntimeError("fm_accel='kvcache' supports ode_method='euler' only.")
            if p.fm_head is None:
                p.fm_head = HeadCls(
                    core, num_steps=p.num_steps, guidance_scale=p.guidance_scale,
                    max_patches=st.fm_capacity // stride + 1, dit=self._vfp,
                )
                if prime:   # switching in mid-generation: build the prefix cache once
                    p.fm_head.prime_to(st, self._seq_gcond(p))
            z = p.fm_head.decode_patch(st, torch.randn(
                (1, lp, ld), device=self.device, dtype=self.dtype), self._seq_gcond(p))
            outs.append(z[0])
        return torch.stack(outs, dim=0)                    # [N, lp, ld]

    def _seq_gcond(self, p: DotsSeqState) -> torch.Tensor:
        st = p.gen_state
        if p.g_cond is not None:
            return p.g_cond.to(self.device, self.dtype).reshape(1, -1)
        return st.fm_null_g_cond.to(self.device, self.dtype)

    def _hybrid_fm(self, payloads: list[DotsSeqState]) -> torch.Tensor:
        """Length-adaptive: cudagraph-full DiT while the FM history is short (it
        wins, O(P²) hasn't bitten), then the O(1)/patch KV cache once history
        exceeds `hybrid_threshold` (cudagraph-full degrades past realtime there).
        Best of both across all audio lengths."""
        thr = self.hybrid_threshold
        results: list[torch.Tensor | None] = [None] * len(payloads)
        short = [i for i, p in enumerate(payloads) if p.gen_state.fm_seq_len < thr]
        long = [i for i, p in enumerate(payloads) if p.gen_state.fm_seq_len >= thr]
        if short:
            res = self._full_fm([payloads[i] for i in short], self._vfp_full)
            for j, i in enumerate(short):
                results[i] = res[j]
        if long:
            res = self._kvcache_fm([payloads[i] for i in long], prime=True)
            for j, i in enumerate(long):
                results[i] = res[j]
        return torch.stack(results, dim=0)

    def _batched_fm(self, payloads: list[DotsSeqState]) -> torch.Tensor:
        """Dispatch the FM head per `fm_accel`. Returns [N, patch_size, latent_dim]."""
        if getattr(self.core, "mode", "flow_matching") == "meanflow":
            # MeanFlow (dots.tts-mf): single-branch + duration embedding. The
            # per-timestep KV cache encodes the CFG pair in its layer math, so it
            # isn't wired for meanflow yet -> use the batched full path.
            if self.fm_accel in ("kvcache", "hybrid"):
                raise RuntimeError(
                    f"fm_accel={self.fm_accel!r} not yet supported for MeanFlow "
                    "(dots.tts-mf); use 'none', 'compile', 'cudagraph', or 'mfgraph'."
                )
            if self.fm_accel == "mfgraph":
                return self._mf_graph(payloads)
            return self._full_fm(payloads, self._vfp)
        if self.fm_accel == "kvcache":
            return self._kvcache_fm(payloads)
        if self.fm_accel == "hybrid":
            return self._hybrid_fm(payloads)
        return self._full_fm(payloads, self._vfp)

    def _pack_fm_inputs(self, payloads: list[DotsSeqState]):
        """Pad every active sequence's FM history into one batch.

        Returns (inp, cfg, mask, pos, gcond, p0, meanflow). `cfg` (the CFG uncond
        sequence) is None for MeanFlow. The padding gap is self-masked and rotary
        positions stay per-seq correct, so the padded batch == running each alone.
        """
        meanflow = getattr(self.core, "mode", "flow_matching") == "meanflow"
        core, dots = self.core, self.dots
        states = [p.gen_state for p in payloads]
        n = len(states)
        max_len = max(st.fm_seq_len for st in states)
        # Optional length bucketing (for CUDA-graph shape stability); adds padding
        # and a bit of numerical drift, so off by default (dynamic compile instead).
        if self.fm_len_bucket > 0:
            b = self.fm_len_bucket
            max_len = ((max_len + b - 1) // b) * b
        # A single batched integration shares the ODE schedule (and, for FM, CFG),
        # so those must be uniform across co-active requests. Fail loud rather than
        # silently decode some requests with the wrong NFE/guidance.
        p0 = payloads[0]
        for p in payloads:
            mism = (p.num_steps != p0.num_steps) if meanflow else (
                (p.num_steps, p.guidance_scale, p.ode_method)
                != (p0.num_steps, p0.guidance_scale, p0.ode_method)
            )
            if mism:
                raise RuntimeError(
                    "batched FM requires uniform decode params (num_steps"
                    + ("" if meanflow else ", guidance_scale, ode_method")
                    + ") across co-active requests; got mismatched decode params."
                )
        total_len = max_len + core.latent_patch_size
        H = core.fm_hidden_size
        dev, dt = self.device, self.dtype

        inp = torch.zeros(n, total_len, H, device=dev, dtype=dt)
        # MeanFlow has no CFG branch, so the uncond (cfg) sequence is never used.
        cfg = None if meanflow else torch.zeros(n, total_len, H, device=dev, dtype=dt)
        mask = torch.zeros(n, total_len, total_len, dtype=torch.bool, device=dev)
        pos = torch.zeros(n, total_len, dtype=torch.float32, device=dev)
        gcond = torch.zeros(n, H, device=dev, dtype=dt)
        for i, (st, p) in enumerate(zip(states, payloads)):
            L = st.fm_seq_len
            inp[i, :L] = st.fm_sequence[0, :L]
            if not meanflow:
                cfg[i, :L] = st.fm_cfg_sequence[0, :L]
            dots._build_fm_attn_mask(state=st, attn_mask=mask[i : i + 1])
            dots._build_fm_pos_ids(state=st, pos_ids=pos[i : i + 1])
            if p.g_cond is not None:
                gcond[i] = p.g_cond.to(dev, dt)
            elif st.fm_null_g_cond is not None:
                gcond[i] = st.fm_null_g_cond[0]
        return inp, cfg, mask, pos, gcond, p0, meanflow

    def _full_fm(self, payloads: list[DotsSeqState], vfp) -> torch.Tensor:
        """Pad every active sequence's FM history into one batch and integrate the
        ODE together through `vfp` (eager / compiled / cudagraph-full)."""
        from nanovllm_dots.models.dots.batched_fm import (
            batched_flow_matching, batched_meanflow,
        )
        core = self.core
        inp, cfg, mask, pos, gcond, p0, meanflow = self._pack_fm_inputs(payloads)
        if meanflow:
            # guidance_scale / ode_method are intentionally dropped: MeanFlow
            # distills CFG in and uses a fixed explicit-Euler few-step integrator.
            return batched_meanflow(
                core, input_sequence=inp, attn_mask=mask, pos_ids=pos,
                g_cond=gcond, num_steps=p0.num_steps, vfp=vfp,
            )
        return batched_flow_matching(
            core, input_sequence=inp, cfg_sequence=cfg, attn_mask=mask, pos_ids=pos,
            g_cond=gcond, num_steps=p0.num_steps, guidance_scale=p0.guidance_scale,
            ode_method=p0.ode_method, vfp=vfp,
        )

    def _mf_graph(self, payloads: list[DotsSeqState]) -> torch.Tensor:
        """Whole-patch MeanFlow: ONE CUDA graph captures the entire nfe-step solver
        (coordinate_proj + clones + DiT + z-update), so a patch is one replay
        instead of nfe replays glued by eager kernels."""
        inp, _, mask, pos, gcond, p0, _ = self._pack_fm_inputs(payloads)
        if self._mfgraph is None:
            from nanovllm_dots.models.dots.graphed_meanflow import GraphedMeanflow
            self._mfgraph = GraphedMeanflow(
                self.core, num_steps=p0.num_steps, dit=self._vfp)
        return self._mfgraph(
            input_sequence=inp, attn_mask=mask, pos_ids=pos,
            g_cond=gcond, num_steps=p0.num_steps,
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
        cap = st.span_count if st.max_patches is None else min(st.span_count, st.max_patches)
        if stop or schedule_exhausted or st.patches_emitted >= cap:
            seq.stoped = True

    def _finish_patches_flash(self, seqs, latents: torch.Tensor, stops) -> None:
        """Batched post-FM for flash_pe: per-seq FM-history append (cheap) + ONE
        flash patch_encoder call over all active seqs + per-seq emit/stop. `latents`
        is [N, patch_size, latent_dim] (normalized, as produced by the FM)."""
        core, dots = self.core, self.dots
        rows = []
        for seq, latent in zip(seqs, latents):
            st = seq.custom_payload
            dots._append_history_chunk(st.gen_state, latent.unsqueeze(0))  # normalized
            if st.pe_row is None:                       # lazily claim a cache row
                if not self._pe_free:
                    raise RuntimeError(
                        f"flash_pe: row pool exhausted ({self._pe_max_batch} rows); "
                        f"more sequences are generating concurrently. Increase "
                        f"pe_max_batch or cap concurrency (max_num_seqs).")
                st.pe_row = self._pe_free.pop()
                self._flash_pe.reset_rows(
                    torch.tensor([st.pe_row], device=self.device, dtype=torch.int32))
            rows.append(st.pe_row)
        rows_t = torch.tensor(rows, device=self.device, dtype=torch.int32)
        patches = core.io_helper.denormalize(latents)              # [N, ps, ld]
        embeds = self._flash_pe.decode_patch(patches, rows_t)      # [N, 1, H]
        for seq, latent, stop, emb in zip(seqs, latents, stops, embeds):
            self._emit_patch(seq, seq.custom_payload, latent.unsqueeze(0), emb, stop)

    def _emit_patch(self, seq: Sequence, st: DotsSeqState, patch: torch.Tensor,
                    emb: torch.Tensor, stop: bool) -> None:
        """Record one emitted patch (latent + LLM next-input) and apply the stop
        rule. `emb` is the precomputed patch_encoder embedding [1, H]."""
        core = self.core
        st.next_embed = emb
        latent_cpu = core.io_helper.denormalize(patch).detach().float().cpu()
        st.latents.append(latent_cpu)
        st.patches_emitted += 1
        st.position += 1
        seq.append_token(latent_cpu.reshape(-1).numpy().tobytes())
        schedule_exhausted = st.position >= st.schedule.size(1)
        cap = st.span_count if st.max_patches is None else min(st.span_count, st.max_patches)
        if stop or schedule_exhausted or st.patches_emitted >= cap:
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

    # --------------------------------------------------------------- vocoder
    @torch.no_grad()
    def vocode(self, latents: torch.Tensor) -> torch.Tensor:
        """One-shot latents [1, frames, latent_dim] -> wav [samples] (48 kHz).
        Mirrors the reference `_decode_latents` (do_sample=False)."""
        if latents.numel() == 0:
            return torch.zeros(0)
        wav = self.dots._decode_latents(latents.to(self.device))  # [1,1,samples]
        return wav.detach().float().cpu().reshape(-1)

    @torch.no_grad()
    def _vocode_stream_step(self, st: DotsSeqState, patch_cpu: torch.Tensor) -> torch.Tensor:
        """Stream ONE emitted patch [1, patch_size, latent_dim] through BigVGAN's
        streaming decoder (chunk_size == latent_patch_size, so 1 patch == 1 step)."""
        if st.vocoder_state is None:
            st.vocoder_state = self.dots._init_vocoder_stream_state()
        # _stream_vocoder_patch expects frames-first [1, ps, latent_dim] (same layout
        # as our stored latents) and transposes to [1, latent_dim, ps] internally.
        lat = patch_cpu.to(self.device)
        chunk = self.dots._stream_vocoder_patch(lat, stream_state=st.vocoder_state)
        return chunk.detach().float().cpu().reshape(-1)

    @torch.no_grad()
    def _vocode_flush(self, st: DotsSeqState) -> torch.Tensor:
        if st.vocoder_state is None:
            return torch.zeros(0)
        final = self.dots._flush_vocoder_stream(st.vocoder_state)
        return final.detach().float().cpu().reshape(-1)

    # ----------------------------------------------------------- streaming run
    @torch.no_grad()
    def step_stream(self, *, vocode: bool = True) -> list[tuple[str, str, torch.Tensor | None]]:
        """One scheduler step; vocode any newly-emitted patches and flush finished
        sequences. Returns events: (seq_id, "audio", wav_chunk) per produced chunk
        and (seq_id, "done", None) when a sequence finishes. This is the primitive
        the async server loop drives, interleaving add_request between calls.
        `vocode=False` emits silent chunks of the right length (profiling: isolates
        the vocoder cost from the LLM/FM/patch_encoder stepping)."""
        hop = self.dots.hop_size * self.core.latent_patch_size
        seqs = self.step()
        if not seqs and not self.scheduler.is_finished():
            raise RuntimeError("scheduler made no progress with pending requests "
                               "(KV exhaustion / preemption livelock).")
        events: list[tuple[str, str, torch.Tensor | None]] = []
        for seq in seqs:
            st: DotsSeqState = seq.custom_payload
            if st.stream:
                # low-latency: vocode each patch as it lands (per-4-frame stream_step)
                while st.vocoded < len(st.latents):
                    chunk = (self._vocode_stream_step(st, st.latents[st.vocoded])
                             if vocode else torch.zeros(hop))
                    st.vocoded += 1
                    if chunk.numel():
                        events.append((seq.seq_id, "audio", chunk))
                if seq.stoped:
                    final = self._vocode_flush(st)
                    if final.numel():
                        events.append((seq.seq_id, "audio", final))
                    events.append((seq.seq_id, "done", None))
            elif seq.stoped:
                # throughput: one-shot vocode the whole utterance at the end (the
                # streaming decoder's per-chunk context recompute is ~11x costlier)
                if vocode:
                    lat = torch.cat(st.latents, dim=1) if st.latents else None
                    wav = self.vocode(lat) if lat is not None else torch.zeros(0)
                else:
                    wav = torch.zeros(hop * len(st.latents))
                if wav.numel():
                    events.append((seq.seq_id, "audio", wav))
                events.append((seq.seq_id, "done", None))
        return events

    @torch.no_grad()
    def generate_stream(self) -> "Iterator[tuple[str, str, torch.Tensor | None]]":
        """Offline driver: run all queued requests to completion, yielding streaming
        events as they are produced (a thin wrapper over step_stream)."""
        while not self.scheduler.is_finished():
            for ev in self.step_stream():        # prefill steps just yield nothing
                yield ev
