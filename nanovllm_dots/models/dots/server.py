"""Async streaming server around DotsBatchEngine (Phase 3).

The engine is single-threaded and step-driven (step_stream: one continuous-
batching step -> per-request audio chunk events). This wraps it for concurrent
async serving:

  - ONE worker thread owns the engine and runs the continuous-batching loop:
    drain newly-submitted requests (thread-safe intake queue) -> engine.add_request,
    then engine.step_stream(), dispatching each audio/done/error event to the
    submitting coroutine. The heavy work is CUDA kernels (GIL released) + the
    engine's per-seq Python, so keeping it off the event loop keeps FastAPI
    responsive. The engine is touched ONLY by this thread.
  - Each `generate()` coroutine registers an asyncio.Queue; the worker pushes
    events to it via loop.call_soon_threadsafe. New requests can arrive mid-loop
    (continuous batching), so many generate() coroutines share one batched engine.

A thread (not a subprocess) loads the ~big model once in-process; a multi-GPU
pool would run one of these per process (future work).
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import uuid
from typing import Any, AsyncIterator

import torch


class DotsStreamServer:
    def __init__(self, engine):
        self.engine = engine
        self.vocode = True                  # set False to profile w/o the vocoder
        self.sample_rate = int(engine.dots.vocoder.sample_rate)
        self._intake: "queue.Queue[tuple]" = queue.Queue()
        self._subs: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="dots-engine", daemon=True)
        self._thread.start()

    # ----------------------------------------------------------- construction
    @classmethod
    def from_pretrained(cls, model_dir: str, *, max_num_seqs: int = 64,
                        num_kvcache_blocks: int = 512, block_size: int = 256,
                        fm_accel: str = "compile", flash_pe: bool = True,
                        graph_decode: bool = True, warmup: bool = True,
                        **engine_kwargs) -> "DotsStreamServer":
        # fm_accel="compile" (dynamic) is the SERVER default, not cudagraph: under
        # continuous batching the batch size changes whenever a request starts/
        # finishes, and cudagraph captures one graph per (batch_size, history_len)
        # -> constant capture churn. compile is dynamic (no per-shape capture) and
        # was slightly FASTER at high batch (60.7x vs 57.8x). graph_decode captures
        # only per batch-size (<=max_num_seqs, pre-captured at construction); flash_pe
        # is kernel-dynamic. So after warmup there is no capture in the serving path.
        import torch.distributed as dist
        from transformers import Qwen2Config
        from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
        from dots_tts.runtime import DotsTtsRuntime
        from nanovllm_dots.models.dots.loader import load_llm_weights
        from nanovllm_dots.models.dots.model_llm import QwenLLM
        from nanovllm_dots.models.dots.engine import DotsBatchEngine
        from nanovllm_dots.models.dots.cudagraph_dit import (
            CudaGraphRunner, make_dit_capture_safe,
        )

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29529")
            dist.init_process_group("nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)

        runtime = DotsTtsRuntime.from_pretrained(
            model_dir, precision="bfloat16", optimize=False, max_generate_length=256)
        ckpt = os.path.join(model_dir, "model.safetensors")
        cfg = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            paged = QwenLLM(cfg).eval()
        load_llm_weights(paged, ckpt)
        torch.set_default_dtype(torch.float32)

        vfp = None
        if fm_accel == "cudagraph":
            make_dit_capture_safe(runtime.model.core.velocity_field_predictor,
                                  torch.device("cuda"))
            vfp = CudaGraphRunner(runtime.model.core.velocity_field_predictor)
        eng = DotsBatchEngine(
            runtime, paged, model_dir=model_dir, num_kvcache_blocks=num_kvcache_blocks,
            block_size=block_size, max_num_seqs=max_num_seqs, fm_accel=fm_accel,
            fm_vfp=vfp, flash_pe=flash_pe, graph_decode=graph_decode,
            pe_max_batch=max_num_seqs, **engine_kwargs)
        srv = cls(eng)
        if warmup:
            srv.warmup()
        return srv

    def warmup(self, *, text: str = "warmup", num_steps: int = 4,
               batch_sizes=(1, 8, 32), patches: int = 8) -> None:
        """Trigger inductor compilation + decode-graph capture before serving by
        running short dummy batches synchronously. The worker thread owns the engine,
        so pause it for the duration."""
        # join WITHOUT a timeout: running the engine from two threads at once (the
        # warmup body + a still-alive worker) corrupts its single-threaded state.
        self._stop.set(); self._thread.join()
        assert not self._thread.is_alive()
        eng = self.engine
        cap = getattr(eng, "_pe_max_batch", 1 << 30)   # never warm past the row pool
        for bs in sorted({min(b, cap) for b in batch_sizes}):
            for i in range(bs):
                eng.add_request(f"_warm{bs}_{i}", text, num_steps=num_steps,
                                eos_threshold=2.0, max_patches=patches, stream=False)
            while not eng.scheduler.is_finished():
                eng.step_stream(vocode=True)
        torch.cuda.synchronize()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="dots-engine", daemon=True)
        self._thread.start()

    # ----------------------------------------------------------- worker loop
    def _loop(self) -> None:
        eng = self.engine
        while not self._stop.is_set():
            self._drain_intake(block=eng.scheduler.is_finished())
            if eng.scheduler.is_finished():
                continue
            try:
                events = eng.step_stream(vocode=self.vocode)
            except Exception as e:                       # fail every active stream
                self._fail_all(e)
                continue
            for sid, kind, data in events:
                self._dispatch(sid, (kind, data))

    def _drain_intake(self, *, block: bool) -> None:
        """Pull submitted requests into the engine. When the engine is idle, block
        briefly so a newly-arrived request wakes the loop without busy-spinning."""
        if block:
            try:
                self._handle(self._intake.get(timeout=0.05))
            except queue.Empty:
                return
        while True:
            try:
                self._handle(self._intake.get_nowait())
            except queue.Empty:
                return

    def _handle(self, item: tuple) -> None:
        if item[0] == "_stop":
            self._stop.set()
            return
        if item[0] == "_cancel":
            self.engine.cancel(item[1])      # client gone: free the batch slot
            return
        _, seq_id, text, params = item
        try:
            self.engine.add_request(seq_id, text, **params)
        except Exception as e:
            self._dispatch(seq_id, ("error", e))

    @staticmethod
    def _post(loop, q, payload) -> None:
        # the target loop may be closed (client gone / shutdown); a raised
        # RuntimeError here would otherwise kill the worker thread.
        try:
            loop.call_soon_threadsafe(q.put_nowait, payload)
        except RuntimeError:
            pass

    def _dispatch(self, seq_id: str, payload: tuple[str, Any]) -> None:
        with self._lock:
            sub = self._subs.get(seq_id)
            if payload[0] in ("done", "error"):
                self._subs.pop(seq_id, None)
        if sub is not None:
            self._post(sub[0], sub[1], payload)

    def _fail_all(self, exc: Exception) -> None:
        with self._lock:
            subs = list(self._subs.values())
            self._subs.clear()
        for loop, q in subs:
            self._post(loop, q, ("error", exc))

    # ----------------------------------------------------------- async client
    async def generate(self, text: str, *, num_steps: int = 4,
                       guidance_scale: float = 1.2, eos_threshold: float = 0.8,
                       max_patches: int | None = None,
                       stream: bool = True, prompt_audio_path: str | None = None,
                       speaker_scale: float = 1.5) -> AsyncIterator[torch.Tensor]:
        """Yield 48 kHz wav chunks (1-D float CPU tensors) for `text`. stream=True
        vocodes per-patch (low latency, many small chunks); stream=False vocodes
        once at the end (one big chunk, much higher throughput). Concurrent calls
        share one batched engine."""
        seq_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subs[seq_id] = (loop, q)
        self._intake.put(("req", seq_id, text, dict(
            num_steps=num_steps, guidance_scale=guidance_scale,
            eos_threshold=eos_threshold, max_patches=max_patches, stream=stream,
            prompt_audio_path=prompt_audio_path, speaker_scale=speaker_scale)))
        finished = False
        try:
            while True:
                kind, data = await q.get()
                if kind == "done":
                    finished = True
                    return
                if kind == "error":
                    finished = True
                    raise RuntimeError(f"generation failed: {data}") from data
                yield data
        finally:
            with self._lock:
                self._subs.pop(seq_id, None)
            if not finished:
                # client disconnected / cancelled mid-stream: tell the engine to
                # abort the seq so it doesn't run to completion holding a slot.
                self._intake.put(("_cancel", seq_id))

    async def generate_wav(self, text: str, **kw) -> torch.Tensor:
        """Non-streaming: the full waveform (one-shot vocode, high throughput)."""
        kw.setdefault("stream", False)
        chunks = [c async for c in self.generate(text, **kw)]
        return torch.cat(chunks) if chunks else torch.zeros(0)

    def close(self) -> None:
        self._intake.put(("_stop",))
        self._thread.join(timeout=5)
